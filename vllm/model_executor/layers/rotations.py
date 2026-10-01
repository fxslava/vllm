# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Orthogonal-rotation primitives for the conjugated DeepSeek prototype
(R-FWHT residual stream + TurboQuant 4-bit MLA latents).

This module is the single source of truth for the rotation math and is
intentionally pure torch (no vllm._C / platform deps) so that
``tools/conjugate_deepseek.py`` can import it standalone and validate
the algebra in float64 on any machine.

Design summary (full derivation in the model and tools docstrings):

- ``R = blockdiag_b(H_b D_b)``: randomized Hadamard over the binary
  decomposition of the dimension; orthogonal for any ``n`` (e.g. 7168 =
  1024 * 7), so RMSNorm outputs keep their Euclidean norm (invariant 1).
- The residual stream stays CANONICAL; rotations exist only between the
  fused entry norm and each layer's (offline-conjugated) weights.
- RoPE slices are never rotated (invariant 2).
- ``R_kv`` is folded offline into both ``kv_a_proj_with_mqa`` (latent
  rows) and ``kv_b_proj`` (input columns), so absorbed decode queries
  live in the rotated latent space and no projection ever needs an
  FWHT (invariant 3).

The runtime FWHT site counter enforces invariant 3's flip side: each
decoder-layer forward must rotate exactly twice (the two fused entry
norms) and never inside a linear layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from vllm.model_executor.custom_op import CustomOp

__all__ = [
    "BlockRandomizedHadamard",
    "fwht",
    "fused_rmsnorm_fwht",
    "turboquant_reshape_and_cache",
    "turboquant_paged_attention",
    "RotatedRMSNorm",
    "simulate_int4",
    "reset_fwht_counter",
    "FWHT_SITE_COUNT",
]

# Runtime rotation-site counter: incremented once per
# ``fused_rmsnorm_fwht`` call (a block-diagonal R over non-power-of-two
# dims launches one small FWHT per power-of-two block inside one site).
FWHT_SITE_COUNT = 0

_INT4_GROUP = 64


def reset_fwht_counter() -> None:
    """Zero the runtime rotation-site counter (test hook)."""
    global FWHT_SITE_COUNT
    FWHT_SITE_COUNT = 0


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


def binary_decomposition(n: int) -> list[int]:
    """Power-of-two blocks (largest first) summing to ``n``."""
    return [1 << b for b in range(n.bit_length() - 1, -1, -1) if n >> b & 1]


class BlockRandomizedHadamard:
    """Block-diagonal randomized Hadamard ``R = blockdiag_b(H_b D_b)``.

    The ``D_b`` are independent Rademacher diagonals; only the signs are
    serialized (``state()``/``from_state()``) since ``H`` is algorithmic.
    """

    def __init__(self, blocks: list[int], signs: torch.Tensor) -> None:
        self.blocks = list(blocks)
        self.signs = signs
        if sum(blocks) != signs.numel():
            raise ValueError("blocks/signs size mismatch")

    @classmethod
    def create(cls, n: int, seed: int) -> BlockRandomizedHadamard:
        blocks = binary_decomposition(n)
        g = torch.Generator().manual_seed(seed)
        signs = torch.where(
            torch.rand(sum(blocks), generator=g) < 0.5, -1.0, 1.0
        ).double()
        return cls(blocks, signs)

    @classmethod
    def identity(cls, n: int) -> BlockRandomizedHadamard:
        return cls([1] * n, torch.ones(n, dtype=torch.float64))

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
            w[row_offset + off : row_offset + off + b, :] = d[:, None] * fwht(
                rows.t()
            ).t()
            off += b


def fused_rmsnorm_fwht(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    rot: BlockRandomizedHadamard,
) -> torch.Tensor:
    """Reference implementation of the fused RMSNorm + randomized FWHT.

    Computes ``((weight * x) * rsqrt(mean(x^2) + eps)) @ R`` in one
    pass with fp32 (fp64 in float64 tests) accumulation, returning
    ``x.dtype``. Orthogonality of ``R`` preserves the Euclidean norm of
    the RMSNorm output (invariant 1). The production kernel keeps this
    resident in SRAM; this reference exists for eager CPU execution and
    exactness tests.

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


def simulate_int4(x: torch.Tensor, group: int = _INT4_GROUP) -> torch.Tensor:
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
    """Reference TurboQuant cache write for rotated MLA latents.

    Contract (real kernel): quantizes ``c_kv_rot`` to packed int4 with
    per-(token, group=64) fp8 scales and stores ``k_pe`` as fp8 into
    ``kv_cache[..., 1, kv_lora + rope]`` (layout
    ``[int4 c_kv_rot | fp8 k_pe]``), one write per token.

    Args:
        c_kv_rot: (T, kv_lora_rank) RMS-normalized, R_kv-rotated latents.
            The per-token ``1/rms`` scalar was applied upstream (rotation
            invariant); the ``kv_a_layernorm`` gain was folded offline
            into ``kv_b_proj``.
        k_pe: (T, 1, qk_rope_head_dim) RoPE keys, canonical basis
            (invariant 2).
        kv_cache: paged cache tensor of the backend's storage dtype.
        slot_mapping: (T,) flat slot ids; negative slots are skipped.
        sim_int4: simulate groupwise int4 on the latent slice.
    """
    slots = slot_mapping.flatten().to(torch.long)
    valid = slots >= 0
    if not valid.all():
        slots = slots[valid]
        c_kv_rot = c_kv_rot[valid]
        k_pe = k_pe[valid]
    if slots.numel() == 0:
        return
    if sim_int4:
        c_kv_rot = simulate_int4(c_kv_rot)
    kv_lora = c_kv_rot.shape[-1]
    flat = kv_cache.view(-1, 1, kv_cache.shape[-1])
    packed = torch.cat([c_kv_rot, k_pe.squeeze(1).to(flat.dtype)], dim=-1)
    flat[slots] = packed.unsqueeze(1).to(flat.dtype)


def turboquant_paged_attention(
    q_rot: torch.Tensor,
    kv_cache: torch.Tensor,
    scale: float,
    *,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    kv_lora_rank: int,
) -> torch.Tensor:
    """Reference TurboQuant paged attention over rotated MLA latents.

    Contract (real kernel): for each query head ``h`` and cached token
    ``t``::

        score[h, t] = scale * (<q_rot[h, :kv_lora], deint4(c_rot[t])>
                              + <q_rot[h, kv_lora:], fp8(k_pe[t])>)

    Scores are computed directly in the rotated latent space (invariant
    3): the query was absorbed with ``W~_UK`` which already carries
    ``R_kv``. The kernel returns the softmax-weighted sum of the ROTATED
    latents (not reconstructed values); ``W~_UV``/``o_proj`` are applied
    afterwards via the per-head ``w_vo`` fold, exact because attention
    is linear in the cached latents.

    Args:
        q_rot: (T, H, kv_lora_rank + qk_rope_head_dim) absorbed decode
            queries ``[q_abs (rotated latent space) | q_rope (canonical)]``.
        kv_cache: (num_blocks, block_size, 1, kv_lora + rope).
        scale: softmax scale ``(qk_nope + qk_rope) ** -0.5``.
        block_table: (T, max_blocks) block ids per query.
        seq_lens: (T,) visible cached tokens per query (inclusive).
        kv_lora_rank: splits the cache head into latent/rope slices.

    Returns:
        context: (T, H, kv_lora_rank) attention-weighted rotated latents.
    """
    num_tokens, num_heads, _ = q_rot.shape
    block_size = kv_cache.shape[1]
    flat = kv_cache.view(-1, kv_cache.shape[-1]).to(q_rot.dtype)
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


def _softmax_precise(scores: torch.Tensor) -> torch.Tensor:
    """Softmax in the widest sane precision, cast back to input dtype."""
    wide = scores.double() if scores.dtype == torch.float64 else scores.float()
    return torch.softmax(wide, dim=-1).to(scores.dtype)


@CustomOp.register("rotated_rms_norm")
class RotatedRMSNorm(CustomOp):
    """Drop-in replacement for ``RMSNorm`` that appends the layer-entry
    randomized Hadamard rotation.

    The rotation is configured from checkpoint-embedded sign tensors
    (``model.layers.N.rot_in_signs``) by :meth:`set_rotation`; both entry
    norms of a layer share one ``R_in``. Until signs are loaded the
    module raises on forward: a rotated architecture requires a
    conjugated checkpoint (see ``tools/conjugate_deepseek.py``).
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self._rotation: BlockRandomizedHadamard | None = None

    @classmethod
    def from_rmsnorm(cls, norm) -> RotatedRMSNorm:
        rotated = cls(norm.weight.numel(), norm.variance_epsilon)
        rotated.weight = norm.weight
        return rotated

    def set_rotation(self, signs: torch.Tensor) -> None:
        """Configure ``R_in`` from a (n,) tensor of +/-1 signs."""
        signs = signs.to(device=self.weight.device, dtype=torch.float32)
        if not torch.all(torch.abs(signs) == 1):
            raise ValueError("rotation signs must be exactly +/-1")
        self._rotation = BlockRandomizedHadamard(
            binary_decomposition(signs.numel()), signs.double()
        )

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self._rotation is None:
            raise RuntimeError(
                "RotatedRMSNorm used without rotation signs; load a "
                "checkpoint produced by tools/conjugate_deepseek.py"
            )
        if residual is not None:
            x = x + residual
            residual = x
        out = fused_rmsnorm_fwht(x, self.weight, self.variance_epsilon, self._rotation)
        return out if residual is None else (out, residual)
