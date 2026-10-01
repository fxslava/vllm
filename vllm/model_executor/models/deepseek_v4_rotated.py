# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prototype: DeepSeek-style MLA + MoE layers with offline-conjugated
randomized Hadamard rotations (R-FWHT), targeting 4-bit KV cache
(TurboQuant) + FP4 experts.

This module is a numerical prototype, not a registered vLLM model: it is
pure torch and intentionally free of vLLM runtime dependencies so the
tensor contracts can be validated on CPU in float64. Its weights must be
produced by ``tools/conjugate_deepseek.py``.

Conventions: storage ``(out, in)`` with ``y = x @ W.T``; math row-vector
``y = x @ W_math`` with ``W_math = W_storage.T``.

Design:

1. Residual stream is CANONICAL. The rotation exists only between the
   fused entry norm and the layer's weights. Every layer output is
   canonical by construction (attention output is de-rotated through the
   conjugated ``kv_b_proj`` value slice; MoE ``down_proj`` consumes
   canonical intermediates), so ``o_proj``, ``down_proj``, embeddings and
   ``lm_head`` are untouched.

2. Layer entry (invariant 1): ``h_rot = fused_rmsnorm_fwht(x, g_in, eps,
   R_in) = (g_in * x / rms(x)) @ R_in`` with ``R_in = H D`` orthogonal,
   hence ``||h_rot|| == ||rmsnorm(x)||``. This is the ONLY runtime FWHT
   site (one call per layer entry; ``input_layernorm`` and
   ``post_attention_layernorm`` share the same per-layer ``R_in``).

3. RoPE decoupling (invariant 2): the ``k_pe`` rows of
   ``kv_a_proj_with_mqa`` are NOT rotated (block-diag identity), and the
   per-head rope slice of ``q_b_proj`` lives in the canonical basis.

4. MLA absorption (invariant 3): ``R_kv`` is folded offline into both the
   latent rows of ``kv_a_proj_with_mqa`` and the input columns of
   ``kv_b_proj``. The decode fold ``w_qk[h] = W~_q_b[h].T @ W~_UK[h]``
   therefore yields queries already living in the rotated latent space
   (``q_abs = q_abs_orig @ R_kv``), so scores are computed directly
   against cached rotated latents. Zero runtime FWHT in any projection.

5. kv_a/q_a layernorm gains: the per-token scalar ``1/rms`` is rotation
   invariant and stays at runtime (applied to the rotated latent); the
   gains ``g_kv``/``g_q`` do NOT commute with the rotation and are folded
   offline into the input columns of ``kv_b_proj``/``q_b_proj``
   (algebraically equal to folding the dense ``R_kv.T @ diag(g) @ R_kv``
   into the consumers).

TP shard-safety: every conjugation acts on the INPUT dimension of
column-parallel weights (``q_a``, ``q_b``, ``kv_b``, MoE ``w13``) or on
replicated weights (``kv_a_proj_with_mqa``, router), so it commutes with
output-dim sharding; ``R_in``/``R_kv`` signs are identical across ranks;
absorbed folds are computed per-rank over local heads.

Cache layout contract (TurboQuant, MLA single KV head):
``kv_cache[num_blocks, block_size, 1, kv_lora_rank + qk_rope_head_dim]``
holding ``[int4(c_kv_rot) | fp8(k_pe)]`` per token in the real kernel;
the reference fallback below stores ``cache_dtype`` and can simulate
int4 on the latent slice to measure quantization robustness.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "BlockRandomizedHadamard",
    "fwht",
    "fused_rmsnorm_fwht",
    "turboquant_reshape_and_cache",
    "turboquant_paged_attention",
    "DeepSeekV4RotatedMLA",
    "DeepSeekV4RotatedMoE",
    "reset_fwht_counter",
]

# Runtime rotation-site counter: incremented once per
# ``fused_rmsnorm_fwht`` call (a block-diagonal R over non-power-of-two
# dims launches one small FWHT per power-of-two block inside a single
# site). Invariant: one decoder-layer forward must increment this exactly
# twice (the two fused entry norms) -- never inside a linear projection.
FWHT_SITE_COUNT = 0

_INT4_GROUP = 64


def fwht(x: torch.Tensor) -> torch.Tensor:
    """Normalized fast Walsh-Hadamard transform along the last dim.

    Returns ``x @ H`` for the symmetric orthogonal Sylvester matrix ``H``
    (``H @ H.T == I``); the last dim must be a power of two.
    """
    n = x.shape[-1]
    if n & (n - 1):
        raise ValueError(f"fwht needs a power-of-two last dim, got {n}")
    y = x.clone(memory_format=torch.contiguous_format)
    h = 1
    while h < n:
        y = y.view(*y.shape[:-1], -1, 2, h)
        a, b = y[..., 0, :], y[..., 1, :]
        y = torch.stack((a + b, a - b), dim=-2).reshape(x.shape)
        h *= 2
    return y * (n**-0.5)


class BlockRandomizedHadamard:
    """Block-diagonal randomized Hadamard ``R = blockdiag_b(H_b D_b)``.

    Block sizes are the binary decomposition of ``n`` (largest first),
    keeping ``R`` orthogonal for any ``n`` (e.g. 7168 = 1024 * 7). The
    ``D_b`` are independent Rademacher diagonals; only the signs are
    serialized since ``H`` is algorithmic.
    """

    def __init__(self, blocks: list[int], signs: torch.Tensor) -> None:
        self.blocks = list(blocks)
        self.signs = signs
        if sum(blocks) != signs.numel():
            raise ValueError("blocks/signs size mismatch")

    @classmethod
    def create(cls, n: int, seed: int) -> BlockRandomizedHadamard:
        blocks = [1 << b for b in range(n.bit_length() - 1, -1, -1) if n >> b & 1]
        g = torch.Generator().manual_seed(seed)
        signs = torch.where(
            torch.rand(sum(blocks), generator=g) < 0.5, -1.0, 1.0
        ).double()
        return cls(blocks, signs)

    def state(self) -> dict:
        bits = "".join("1" if s > 0 else "0" for s in self.signs.tolist())
        return {"blocks": self.blocks, "signs_hex": f"{int(bits, 2):x}"}

    @classmethod
    def from_state(
        cls, st: dict, dtype: torch.dtype, device: torch.device | None = None
    ) -> BlockRandomizedHadamard:
        n = sum(st["blocks"])
        bits = f"{int(st['signs_hex'], 16):0{n}b}"
        signs = torch.tensor(
            [-1.0 if b == "0" else 1.0 for b in bits], dtype=torch.float64
        )
        return cls(st["blocks"], signs.to(device=device, dtype=dtype))

    def _to(self, dtype: torch.dtype, device: torch.device | None = None):
        self.signs = self.signs.to(device=device, dtype=dtype)
        return self

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """``x @ R`` (canonical -> rotated), along the last dim."""
        y = x.clone()
        off = 0
        for b in self.blocks:
            d = self.signs[off : off + b].to(x.dtype)
            y[..., off : off + b] = fwht(x[..., off : off + b]) * d
            off += b
        return y

    def apply_t(self, x: torch.Tensor) -> torch.Tensor:
        """``x @ R.T`` (rotated -> canonical), along the last dim."""
        y = x.clone()
        off = 0
        for b in self.blocks:
            d = self.signs[off : off + b].to(x.dtype)
            y[..., off : off + b] = fwht(x[..., off : off + b] * d)
            off += b
        return y

    def right_mult_(self, w: torch.Tensor) -> None:
        """In-place ``w <- w @ R``: conjugate a weight's input dim."""
        off = 0
        for b in self.blocks:
            d = self.signs[off : off + b].to(w.dtype)
            w[:, off : off + b] = fwht(w[:, off : off + b]) * d
            off += b

    def left_mult_t_(self, w: torch.Tensor, row_offset: int = 0) -> None:
        """In-place ``w <- blockdiag(R, I) @ w`` on the row block starting
        at ``row_offset``: fold an output rotation into a weight."""
        off = 0
        for b in self.blocks:
            d = self.signs[off : off + b].to(w.dtype)
            rows = w[row_offset + off : row_offset + off + b, :]
            w[row_offset + off : row_offset + off + b, :] = (
                d[:, None] * fwht(rows.t()).t()
            )
            off += b


# ---------------------------------------------------------------------------
# Kernel stubs: contracts + exact reference fallbacks
# ---------------------------------------------------------------------------


def fused_rmsnorm_fwht(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    rot: BlockRandomizedHadamard,
) -> torch.Tensor:
    """STUB: fused RMSNorm + randomized FWHT, SRAM-resident.

    Computes ``((weight * x) * rsqrt(mean(x^2) + eps)) @ R`` in a single
    pass with fp32 (fp64 in tests) accumulation, returning ``x.dtype``.
    Since ``R`` is orthogonal, the output preserves the Euclidean norm of
    the RMSNorm output (invariant 1).

    Args:
        x: (..., n) canonical activations entering the layer.
        weight: (n,) RMSNorm gain in the canonical basis.
        eps: variance epsilon.
        rot: per-layer randomized Hadamard for this entry point.

    Returns:
        (..., n) rotated activations feeding the layer's projections.

    """
    global FWHT_SITE_COUNT
    FWHT_SITE_COUNT += 1
    acc = x.double() if x.dtype == torch.float64 else x.float()
    normed = acc * torch.rsqrt(acc.pow(2).mean(-1, keepdim=True) + eps)
    return rot.apply(normed * weight.to(acc.dtype)).to(x.dtype)


def _sim_int4(x: torch.Tensor, group: int = _INT4_GROUP) -> torch.Tensor:
    """Symmetric int4 round-trip with per-(row, group) scales."""
    n = x.shape[-1]
    if n % group:
        raise ValueError(f"last dim {n} not divisible by group {group}")
    xv = x.reshape(-1, n // group, group)
    scale = xv.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 7.0
    q = (xv / scale).round().clamp_(-7, 7)
    return (q * scale).reshape(x.shape)


def turboquant_reshape_and_cache(
    c_kv_rot: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    sim_int4: bool = False,
) -> None:
    """STUB: TurboQuant cache write for rotated MLA latents.

    Contract (real kernel): quantizes ``c_kv_rot`` to packed int4 with
    per-(token, group=64) fp8 scales and stores ``k_pe`` as fp8 into
    ``kv_cache[num_blocks, block_size, 1, kv_lora + rope]`` (layout
    ``[int4 c_kv_rot | fp8 k_pe]``), one write per token.

    Args:
        c_kv_rot: (T, kv_lora_rank) RMS-normalized, R_kv-rotated latents.
            The per-token ``1/rms`` scalar was already applied upstream
            (rotation invariant); the ``kv_a_layernorm`` gain was folded
            offline into ``kv_b_proj``.
        k_pe: (T, 1, qk_rope_head_dim) RoPE keys, canonical basis
            (invariant 2).
        kv_cache: paged cache tensor of dtype ``cache_dtype``.
        slot_mapping: (T,) int64 flat slot ids.
        sim_int4: reference-fallback only: simulate int4 on the latent
            slice to measure quantization robustness.

    """
    if sim_int4:
        c_kv_rot = _sim_int4(c_kv_rot)
    packed = torch.cat([c_kv_rot, k_pe.squeeze(1)], dim=-1).unsqueeze(1)
    kv_cache.view(-1, 1, kv_cache.shape[-1])[slot_mapping] = packed


def turboquant_paged_attention(
    q_rot: torch.Tensor,
    k_pe: torch.Tensor | None,
    kv_cache: torch.Tensor,
    scale: float,
    *,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    kv_lora_rank: int,
) -> torch.Tensor:
    """STUB: TurboQuant paged attention over rotated MLA latents.

    Contract (real kernel): for each query head ``h`` and cached token
    ``t``::

        score[h, t] = scale * (<q_rot[h, :kv_lora], deint4(c_rot[t])>
                              + <q_rot[h, kv_lora:], fp8(k_pe[t])>)

    Scores are computed directly in the rotated latent space (invariant
    3): the query was absorbed with ``W~_UK`` which already carries
    ``R_kv``. The kernel returns the softmax-weighted sum of the ROTATED
    latents (not reconstructed values); ``W~_UV``/``o_proj`` are applied
    afterwards via the per-head ``w_vo`` fold, which is exact because
    attention is linear in the cached latents.

    Args:
        q_rot: (T, H, kv_lora_rank + qk_rope_head_dim) absorbed decode
            queries ``[q_abs (rotated latent space) | q_rope (canonical)]``.
        k_pe: (T, 1, rope) current-token RoPE keys; reserved for fused
            kernels that keep them in registers. The reference fallback
            reads every rope key from ``kv_cache`` and ignores this.
        kv_cache: (num_blocks, block_size, 1, kv_lora + rope).
        scale: softmax scale ``(qk_nope + qk_rope) ** -0.5``.
        block_table: (T, max_blocks_per_seq) block ids per query.
        seq_lens: (T,) number of cached tokens visible to each query
            (inclusive of the token just written).
        kv_lora_rank: splits the cache head into latent/rope slices.

    Returns:
        context: (T, H, kv_lora_rank) attention-weighted rotated latents.

    """
    num_tokens, num_heads, _ = q_rot.shape
    block_size = kv_cache.shape[1]
    flat = kv_cache.view(-1, kv_cache.shape[-1])
    out = torch.empty(num_tokens, num_heads, kv_lora_rank, dtype=q_rot.dtype)
    for t in range(num_tokens):
        seq_len = int(seq_lens[t])
        n_blocks = (seq_len + block_size - 1) // block_size
        rows = torch.cat(
            [
                flat[int(b) * block_size : (int(b) + 1) * block_size]
                for b in block_table[t, :n_blocks]
            ],
            dim=0,
        )[:seq_len]
        c_rot, k_pe_cached = rows[:, :kv_lora_rank], rows[:, kv_lora_rank:]
        q = q_rot[t]
        scores = (q[:, None, :kv_lora_rank] * c_rot[None]).sum(-1) + (
            q[:, None, kv_lora_rank:] * k_pe_cached[None]
        ).sum(-1)
        alpha = _softmax_precise(scores * scale)
        out[t] = alpha @ c_rot
    return out


def _rope_interleaved(
    x: torch.Tensor, positions: torch.Tensor, base: float
) -> torch.Tensor:
    """GPT-NeoX-style=False (interleaved pairs) RoPE; DeepSeek's mode.

    Args:
        x: (T, H, d) canonical rope slice, d even.
        positions: (T,) int64 absolute positions.
        base: rope theta.

    Returns:
        (T, H, d) rope-embedded slice, same dtype/basis.

    """
    d = x.shape[-1]
    inv_freq = base ** (
        -torch.arange(0, d, 2, dtype=torch.float64, device=x.device) / d
    )
    ang = positions.to(torch.float64)[:, None] * inv_freq[None, :]
    cos, sin = ang.cos(), ang.sin()
    x1 = x[..., 0::2].to(torch.float64)
    x2 = x[..., 1::2].to(torch.float64)
    o1 = x1 * cos[:, None, :] - x2 * sin[:, None, :]
    o2 = x1 * sin[:, None, :] + x2 * cos[:, None, :]
    return torch.stack((o1, o2), dim=-1).flatten(-2).to(x.dtype)


def _softmax_precise(scores: torch.Tensor) -> torch.Tensor:
    """Softmax in the widest sane precision (fp32 for bf16/fp16 inputs,
    fp64 in the exactness self-tests), cast back to the input dtype."""
    wide = scores.double() if scores.dtype == torch.float64 else scores.float()
    return torch.softmax(wide, dim=-1).to(scores.dtype)


def _causal_mha(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float
) -> torch.Tensor:
    """Reference causal attention. q/k: (T, H, dk), v: (T, H, dv)."""
    scores = torch.einsum("thd,shd->hts", q, k) * scale
    T = q.shape[0]
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), diagonal=1)
    scores = scores.masked_fill(mask[None], float("-inf"))
    alpha = _softmax_precise(scores)
    return torch.einsum("hts,shd->thd", alpha, v)


def _rms_scalar(x: torch.Tensor, eps: float) -> torch.Tensor:
    """``x * rsqrt(mean(x^2) + eps)`` over the last dim (no gain)."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


# ---------------------------------------------------------------------------
# Rotated MLA layer
# ---------------------------------------------------------------------------


@dataclass
class RotatedMLAConfig:
    hidden_size: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    num_heads: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    dtype: torch.dtype = torch.float64


class DeepSeekV4RotatedMLA(nn.Module):
    """MLA with offline-conjugated ``R_in``/``R_kv`` rotations.

    Weight contracts (all produced by ``tools/conjugate_deepseek.py``;
    math-convention formulas in the module docstring of that script):

    - ``w_q_a``  (q_lora, hidden):           ``W_q_a @ R_in``
    - ``w_kv_a`` (kv_lora + rope, hidden):
        ``blockdiag(R_kv, I_rope).T @ W_kv_a @ R_in`` -- latent rows come
        out pre-rotated, rope rows canonical (invariant 2)
    - ``w_q_b``  (H*(nope+rope), q_lora):    columns scaled by ``g_q``
    - ``w_kv_b`` (H*(nope+v), kv_lora):      ``(W_kv_b * g_kv) @ R_kv``
    - ``w_o``    (hidden, H*v):              untouched (canonical input)

    Per-head row layout follows vLLM's MLA modules: ``q_b_proj`` heads are
    ``[nope | rope]``; ``kv_b_proj`` heads are ``[K_nope | V]``.

    Derived decode folds (per local head, TP-safe):
    - ``w_qk[h]`` (q_lora, kv_lora) = ``W~_q_b[h].T @ W~_UK[h]``
    - ``w_vo[h]`` (kv_lora, hidden) = ``W~_UV[h].T @ W_o[h].T``
    """

    def __init__(
        self,
        cfg: RotatedMLAConfig,
        rot_in: BlockRandomizedHadamard,
        rot_kv: BlockRandomizedHadamard,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.rot_in = rot_in._to(cfg.dtype)
        self.rot_kv = rot_kv._to(cfg.dtype)
        self.scale = (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim) ** -0.5
        d, dt = cfg.hidden_size, cfg.dtype
        self.g_in = nn.Parameter(torch.ones(d, dtype=dt))
        self.w_q_a = nn.Parameter(torch.empty(cfg.q_lora_rank, d, dtype=dt))
        self.w_kv_a = nn.Parameter(
            torch.empty(cfg.kv_lora_rank + cfg.qk_rope_head_dim, d, dtype=dt)
        )
        self.w_q_b = nn.Parameter(
            torch.empty(
                cfg.num_heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim),
                cfg.q_lora_rank,
                dtype=dt,
            )
        )
        self.w_kv_b = nn.Parameter(
            torch.empty(
                cfg.num_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim),
                cfg.kv_lora_rank,
                dtype=dt,
            )
        )
        self.w_o = nn.Parameter(
            torch.empty(d, cfg.num_heads * cfg.v_head_dim, dtype=dt)
        )
        self.w_qk: torch.Tensor
        self.w_vo: torch.Tensor

    def load_conjugated(self, sd: dict[str, torch.Tensor], prefix: str) -> None:
        """Load a conjugated HF-layout state dict slice for one layer.

        Keys follow the HF DeepSeek layout (``mlp.experts.{e}``,
        ``mlp.shared_experts``, ``mlp.gate``); vLLM's own remapping to
        ``experts.*`` applies afterwards as usual.
        """

        def g(name: str) -> torch.Tensor:
            return sd[f"{prefix}.{name}"]

        with torch.no_grad():
            self.g_in.copy_(g("input_layernorm.weight"))
            self.w_q_a.copy_(g("self_attn.q_a_proj.weight"))
            self.w_kv_a.copy_(g("self_attn.kv_a_proj_with_mqa.weight"))
            self.w_q_b.copy_(g("self_attn.q_b_proj.weight"))
            self.w_kv_b.copy_(g("self_attn.kv_b_proj.weight"))
            self.w_o.copy_(g("self_attn.o_proj.weight"))
        self.prepare_decode_weights()

    def prepare_decode_weights(self) -> None:
        """Absorb ``W~_UK``/``W~_UV`` (rotation included) into the query
        and output projections (invariant 3)."""
        cfg = self.cfg
        h, dh, dv = cfg.num_heads, cfg.qk_nope_head_dim, cfg.v_head_dim
        w_qk = torch.empty(h, cfg.q_lora_rank, cfg.kv_lora_rank, dtype=self.w_q_b.dtype)
        w_vo = torch.empty(h, cfg.kv_lora_rank, cfg.hidden_size, dtype=self.w_o.dtype)
        per_kv, per_q = dh + dv, dh + cfg.qk_rope_head_dim
        for i in range(h):
            q_nope = self.w_q_b[i * per_q : i * per_q + dh]
            w_k = self.w_kv_b[i * per_kv : i * per_kv + dh]
            w_v = self.w_kv_b[i * per_kv + dh : (i + 1) * per_kv]
            w_o_h = self.w_o[:, i * dv : (i + 1) * dv]
            w_qk[i] = q_nope.t() @ w_k
            w_vo[i] = w_v.t() @ w_o_h.t()
        self.w_qk, self.w_vo = w_qk, w_vo

    def _project(self, hidden_states: torch.Tensor):
        """Shared projections. Returns ``(q_c, c_kv_rot, k_pe_raw)``.

        h_rot (T, d) -> q_c (T, q_lora), c_kv_rot (T, kv_lora) [normed,
        rotated], k_pe_raw (T, rope) [canonical]. No FWHT is launched
        here: ``R_in``/``R_kv`` live inside the weights.
        """
        h_rot = fused_rmsnorm_fwht(
            hidden_states, self.g_in, self.cfg.rms_norm_eps, self.rot_in
        )
        q_c_raw = h_rot @ self.w_q_a.t()
        kv = h_rot @ self.w_kv_a.t()
        c_kv_rot_raw, k_pe_raw = kv.split(
            [self.cfg.kv_lora_rank, self.cfg.qk_rope_head_dim], dim=-1
        )
        # 1/rms is rotation invariant, so it is applied to the rotated
        # latent; the layernorm gains were folded into w_q_b/w_kv_b.
        q_c = _rms_scalar(q_c_raw, self.cfg.rms_norm_eps)
        c_kv_rot = _rms_scalar(c_kv_rot_raw, self.cfg.rms_norm_eps)
        return q_c, c_kv_rot, k_pe_raw

    def forward_prefill(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Non-absorbed prefill: materializes canonical K/V from the
        conjugated ``w_kv_b``, then runs standard causal MHA.

        Args:
            positions: (T,) int64.
            hidden_states: (T, hidden) CANONICAL residual-stream input.

        Returns:
            (T, hidden) canonical attention output (post ``o_proj``).

        """
        cfg = self.cfg
        T = hidden_states.shape[0]
        q_c, c_kv_rot, k_pe_raw = self._project(hidden_states)
        q = (q_c @ self.w_q_b.t()).view(
            T, cfg.num_heads, cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
        )
        q_pe = _rope_interleaved(
            q[..., cfg.qk_nope_head_dim :], positions, cfg.rope_theta
        )
        q = torch.cat([q[..., : cfg.qk_nope_head_dim], q_pe], dim=-1)
        k_pe = _rope_interleaved(k_pe_raw.unsqueeze(1), positions, cfg.rope_theta)
        kv = (c_kv_rot @ self.w_kv_b.t()).view(
            T, cfg.num_heads, cfg.qk_nope_head_dim + cfg.v_head_dim
        )
        k_nope = kv[..., : cfg.qk_nope_head_dim]
        v = kv[..., cfg.qk_nope_head_dim :]
        k = torch.cat(
            [k_nope, k_pe.expand(T, cfg.num_heads, cfg.qk_rope_head_dim)], dim=-1
        )
        attn = _causal_mha(q, k, v, self.scale)
        return attn.reshape(T, -1) @ self.w_o.t()

    def forward_decode(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        slot_mapping: torch.Tensor,
        *,
        sim_int4: bool = False,
    ) -> torch.Tensor:
        """Absorbed decode: scores in rotated latent space, no FWHT.

        Args:
            positions: (T,) int64 positions of the decoded tokens.
            hidden_states: (T, hidden) CANONICAL residual-stream input.
            kv_cache: (num_blocks, block_size, 1, kv_lora + rope).
            block_table: (T, max_blocks) paged block ids.
            seq_lens: (T,) visible cached length per query (inclusive).
            slot_mapping: (T,) flat slots for the new tokens.
            sim_int4: forwarded to the cache-write stub.

        Returns:
            (T, hidden) canonical attention output (post ``o_proj``).

        """
        cfg = self.cfg
        q_c, c_kv_rot, k_pe_raw = self._project(hidden_states)
        k_pe = _rope_interleaved(k_pe_raw.unsqueeze(1), positions, cfg.rope_theta)
        turboquant_reshape_and_cache(
            c_kv_rot, k_pe, kv_cache, slot_mapping, sim_int4=sim_int4
        )
        q = (q_c @ self.w_q_b.t()).view(
            -1, cfg.num_heads, cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
        )
        q_rope = _rope_interleaved(
            q[..., cfg.qk_nope_head_dim :], positions, cfg.rope_theta
        )
        q_abs = torch.einsum("tc,hcd->thd", q_c, self.w_qk)
        q_rot = torch.cat([q_abs, q_rope], dim=-1)
        context = turboquant_paged_attention(
            q_rot,
            k_pe,
            kv_cache,
            self.scale,
            block_table=block_table,
            seq_lens=seq_lens,
            kv_lora_rank=cfg.kv_lora_rank,
        )
        return torch.einsum("thd,hde->te", context, self.w_vo)


# ---------------------------------------------------------------------------
# Rotated MoE layer
# ---------------------------------------------------------------------------


@dataclass
class RotatedMoEConfig:
    hidden_size: int
    moe_intermediate_size: int
    n_routed_experts: int
    num_experts_per_tok: int
    n_shared_experts: int = 1
    renormalize: bool = True
    routed_scaling_factor: float = 1.0
    rms_norm_eps: float = 1e-6
    dtype: torch.dtype = torch.float64


class DeepSeekV4RotatedMoE(nn.Module):
    """MoE with ``R_in``-conjugated router/gate/up (invariant 4).

    Weight contracts (offline conjugation):
    - ``gate`` (E, hidden): ``W_gate @ R_in`` -> EXACT router logits
    - ``w13[e]`` (2I, hidden): ``[W_gate_e; W_up_e] @ R_in`` -> gate/up
      outputs canonical, so ``SiLU(gate) * up`` is elementwise-exact
    - ``w2[e]`` (hidden, moe_intermediate): untouched (canonical in)
    - shared expert ``sh_g``/``sh_u`` conjugated, ``sh_d`` untouched
    """

    def __init__(
        self,
        cfg: RotatedMoEConfig,
        rot_in: BlockRandomizedHadamard,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.rot_in = rot_in._to(cfg.dtype)
        d, inter, dt = cfg.hidden_size, cfg.moe_intermediate_size, cfg.dtype
        self.g_post = nn.Parameter(torch.ones(d, dtype=dt))
        self.gate = nn.Parameter(torch.empty(cfg.n_routed_experts, d, dtype=dt))
        self.w13 = nn.Parameter(
            torch.empty(cfg.n_routed_experts, 2 * inter, d, dtype=dt)
        )
        self.w2 = nn.Parameter(torch.empty(cfg.n_routed_experts, d, inter, dtype=dt))
        sh_I = inter * cfg.n_shared_experts
        self.sh_g = nn.Parameter(torch.empty(sh_I, d, dtype=dt))
        self.sh_u = nn.Parameter(torch.empty(sh_I, d, dtype=dt))
        self.sh_d = nn.Parameter(torch.empty(d, sh_I, dtype=dt))

    def load_conjugated(self, sd: dict[str, torch.Tensor], prefix: str) -> None:
        """Load a conjugated HF-layout state dict slice for one layer."""
        cfg = self.cfg
        inter = cfg.moe_intermediate_size
        with torch.no_grad():
            self.g_post.copy_(sd[f"{prefix}.post_attention_layernorm.weight"])
            self.gate.copy_(sd[f"{prefix}.mlp.gate.weight"])
            for e in range(cfg.n_routed_experts):
                g13 = sd[f"{prefix}.mlp.experts.{e}.gate_proj.weight"]
                u13 = sd[f"{prefix}.mlp.experts.{e}.up_proj.weight"]
                self.w13[e, :inter].copy_(g13)
                self.w13[e, inter:].copy_(u13)
                self.w2[e].copy_(sd[f"{prefix}.mlp.experts.{e}.down_proj.weight"])
            self.sh_g.copy_(sd[f"{prefix}.mlp.shared_experts.gate_proj.weight"])
            self.sh_u.copy_(sd[f"{prefix}.mlp.shared_experts.up_proj.weight"])
            self.sh_d.copy_(sd[f"{prefix}.mlp.shared_experts.down_proj.weight"])

    @staticmethod
    def _softmax_topk(
        logits: torch.Tensor, k: int, renormalize: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Softmax top-k routing (DeepSeek ``scoring_func="softmax"``)."""
        probs = _softmax_precise(logits)
        weights, ids = torch.topk(probs, k, dim=-1)
        if renormalize:
            weights = weights / weights.sum(-1, keepdim=True)
        return weights, ids

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Args:
            hidden_states: (T, hidden) CANONICAL residual-stream input.

        Returns:
            (T, hidden) canonical MoE output (residual update).

        """
        cfg = self.cfg
        h_rot = fused_rmsnorm_fwht(
            hidden_states, self.g_post, cfg.rms_norm_eps, self.rot_in
        )
        router_logits = h_rot @ self.gate.t()
        weights, ids = self._softmax_topk(
            router_logits, cfg.num_experts_per_tok, cfg.renormalize
        )
        inter = cfg.moe_intermediate_size
        y = torch.zeros_like(h_rot)
        for e in range(cfg.n_routed_experts):
            tok, slot = torch.where(ids == e)
            if tok.numel() == 0:
                continue
            gate_out = h_rot[tok] @ self.w13[e, :inter].t()
            up_out = h_rot[tok] @ self.w13[e, inter:].t()
            expert = (F.silu(gate_out) * up_out) @ self.w2[e].t()
            y.index_add_(0, tok, expert * weights[tok, slot, None])
        y = y * cfg.routed_scaling_factor
        shared = (
            F.silu(h_rot @ self.sh_g.t()) * (h_rot @ self.sh_u.t())
        ) @ self.sh_d.t()
        return y + shared


def reset_fwht_counter() -> None:
    """Zero the runtime rotation-site counter (test hook)."""
    global FWHT_SITE_COUNT
    FWHT_SITE_COUNT = 0


if __name__ == "__main__":
    # Shape/dtype contract smoke test with random "already conjugated"
    # weights (exactness is proven by tools/conjugate_deepseek.py
    # --self-test, which checks against an independent reference).
    torch.manual_seed(0)
    dtype = torch.float64
    cfg = RotatedMLAConfig(
        hidden_size=1536,
        q_lora_rank=384,
        kv_lora_rank=256,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        num_heads=4,
        dtype=dtype,
    )
    rot_in = BlockRandomizedHadamard.create(cfg.hidden_size, seed=1)
    rot_kv = BlockRandomizedHadamard.create(cfg.kv_lora_rank, seed=2)
    mla = DeepSeekV4RotatedMLA(cfg, rot_in, rot_kv).double()
    with torch.no_grad():
        for p in mla.parameters():
            p.normal_(0, 0.02)
        mla.prepare_decode_weights()

    T = 7
    x = torch.randn(T, cfg.hidden_size, dtype=dtype)
    pos = torch.arange(T)
    reset_fwht_counter()
    out = mla.forward_prefill(pos, x)
    assert out.shape == x.shape and out.dtype == dtype
    assert FWHT_SITE_COUNT == 1, "prefill must rotate exactly once (entry norm)"

    num_blocks, block_size = 4, 16
    head_kv = cfg.kv_lora_rank + cfg.qk_rope_head_dim
    kv_cache = torch.zeros(num_blocks, block_size, 1, head_kv, dtype=dtype)
    slots = torch.arange(T)
    seq_lens = torch.tensor([T])
    block_table = torch.arange(num_blocks).view(1, num_blocks)
    reset_fwht_counter()
    out_d = mla.forward_decode(
        pos[-1:], x[-1:], kv_cache, block_table, seq_lens, slots[-1:]
    )
    assert out_d.shape == (1, cfg.hidden_size) and out_d.dtype == dtype
    assert FWHT_SITE_COUNT == 1, "decode must rotate exactly once (entry norm)"

    moe_cfg = RotatedMoEConfig(
        hidden_size=cfg.hidden_size,
        moe_intermediate_size=512,
        n_routed_experts=8,
        num_experts_per_tok=2,
        dtype=dtype,
    )
    moe = DeepSeekV4RotatedMoE(moe_cfg, rot_in).double()
    with torch.no_grad():
        for p in moe.parameters():
            p.normal_(0, 0.02)
    reset_fwht_counter()
    moe_out = moe(x)
    assert moe_out.shape == x.shape and moe_out.dtype == dtype
    assert FWHT_SITE_COUNT == 1, "moe must rotate exactly once (entry norm)"

    print("prefill out:", tuple(out.shape), "| decode out:", tuple(out_d.shape))
    print("moe out:", tuple(moe_out.shape), "| dtype:", moe_out.dtype)
    print("rotation sites per layer forward: 1 (entry norm only)")
    print("shape/dtype contract smoke test passed")
