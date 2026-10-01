#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline weight conjugation for the rotated DeepSeek prototype.

Folds randomized Hadamard rotations ``R = H D`` (block-diagonal over the
binary decomposition of the dimension) into an HF-layout DeepSeek
checkpoint so that ``DeepSeekV4RotatedForCausalLM``
(vllm/model_executor/models/deepseek_v4_rotated.py) runs with zero
runtime FWHT inside any linear projection.

Per-layer transforms (math row-vector convention, weights ``W_math`` of
shape ``(in, out)``; PyTorch storage is the transpose):

    W~_DKV   = R_in^T @ W^DKV @ blkdiag(R_kv, I_rope)
    W~_KR    = R_in^T @ W^KR                     (rope rows: canonical)
    W~_UK    = R_kv^T @ diag(g_kv) @ W^UK
    W~_UV    = R_kv^T @ diag(g_kv) @ W^UV
    W~_DQ    = R_in^T @ W^DQ
    W~_UQ    = diag(g_q) @ W^UQ
    W~_gate  = R_in^T @ W^gate_moe               (exact top-k logits)
    W~_gate/up = R_in^T @ W^gate/up               (experts/shared/dense)

Notes:
- ``W~_DQ`` is not in the task's formula list but is forced by invariant
  1: ``q_a_proj`` consumes the rotated entry norm, exactly like DKV's.
- ``g_q``/``g_kv`` (q_a/kv_a layernorm gains) do not commute with the
  rotation; the per-token ``1/rms`` scalar does (norm invariance), so
  the gains are folded into the consumers' input columns and only the
  scalar normalization remains at runtime. The conjugated checkpoint
  therefore stores ones for both a-layernorms.
- ``o_proj``/``down_proj``/embeddings/``lm_head`` are copied verbatim:
  the residual stream stays canonical.
- TP shard-safety: every conjugation acts on a weight's INPUT dim
  (column-parallel sharding is on the OUTPUT dim) or on replicated
  weights (kv_a, router), so sharding may proceed unchanged afterwards.
- Each layer additionally emits ``model.layers.N.rot_in_signs`` /
  ``rot_kv_signs`` (float32 +/-1) consumed by ``RotatedRMSNorm`` /
  the TurboQuant MLA backend; only signs are serialized since ``H`` is
  algorithmic.

Modes:
    --input/--output      conjugate a real HF checkpoint (safetensors)
    --make-tiny-mock DIR  write a small random canonical checkpoint and
                          its conjugated twin (DIR/canonical,
                          DIR/rotated) for end-to-end engine tests
    --self-test           float64 exactness proof against an independent
                          reference forward + int4 outlier demo

Usage:
    python tools/conjugate_deepseek.py --input <hf_ckpt> --output <dir> \
        [--seed 0] [--dtype bfloat16]
    python tools/conjugate_deepseek.py --make-tiny-mock /tmp/mock [--moe]
    python tools/conjugate_deepseek.py --self-test
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]

META_FORMAT = 1


def _import_rotations():
    """Import the rotation primitives from this checkout."""
    import sys as _sys

    if str(REPO_ROOT) not in _sys.path:
        _sys.path.insert(0, str(REPO_ROOT))
    try:
        from vllm.model_executor.layers import rotations as mod

        return mod
    except ImportError as e:  # pragma: no cover - environment issue
        raise SystemExit(
            "cannot import vllm.model_executor.layers.rotations from "
            f"{REPO_ROOT}: {e}"
        ) from e


def dense_block_hadamard(n: int) -> torch.Tensor:
    """Independent Sylvester construction (self-test ground truth)."""
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h]), torch.cat([h, -h])], dim=1)
    return h * (n**-0.5)


def dense_from_rot(rot) -> torch.Tensor:
    """Materialize ``R = blockdiag(H_b D_b)`` for ground-truth checks."""
    n = sum(rot.blocks)
    r = torch.zeros(n, n, dtype=torch.float64)
    off = 0
    for b in rot.blocks:
        r[off : off + b, off : off + b] = dense_block_hadamard(b)
        off += b
    return r @ torch.diag(rot.signs.double())


# ---------------------------------------------------------------------------
# Offline conjugation
# ---------------------------------------------------------------------------


def conjugate_layer(sd, prefix, rot_in, rot_kv, m) -> dict[str, torch.Tensor]:
    """Conjugate one layer's weights. Returns new tensors; input intact.

    Raises:
        NotImplementedError: for non-MLA (dense) self-attention layers,
        whose RoPE-constrained key projections need a per-layer design
        decision that this prototype intentionally does not guess.
    """
    out: dict[str, torch.Tensor] = {}

    def src(name: str) -> torch.Tensor:
        return sd[f"{prefix}.{name}"].double()

    def conj_in(w: torch.Tensor) -> torch.Tensor:
        w = w.clone()
        rot_in.right_mult_(w)
        return w

    if f"{prefix}.self_attn.q_a_proj.weight" in sd:
        g_q = src("self_attn.q_a_layernorm.weight")
        g_kv = src("self_attn.kv_a_layernorm.weight")
        # DQ: consume the rotated entry norm.
        out["self_attn.q_a_proj.weight"] = conj_in(src("self_attn.q_a_proj.weight"))
        # DKV: latent rows rotated, rope rows canonical (invariant 2).
        w_kv_a = src("self_attn.kv_a_proj_with_mqa.weight").clone()
        rot_kv.left_mult_t_(w_kv_a)
        rot_in.right_mult_(w_kv_a)
        out["self_attn.kv_a_proj_with_mqa.weight"] = w_kv_a
        # UQ: fold g_q into the input columns.
        w_q_b = src("self_attn.q_b_proj.weight").clone()
        out["self_attn.q_b_proj.weight"] = w_q_b * g_q[None, :]
        # UK/UV: fold g_kv into the input columns, then R_kv.
        w_kv_b = src("self_attn.kv_b_proj.weight").clone()
        w_kv_b = w_kv_b * g_kv[None, :]
        rot_kv.right_mult_(w_kv_b)
        out["self_attn.kv_b_proj.weight"] = w_kv_b
        # Gains were folded; keep ones for structural compatibility.
        out["self_attn.q_a_layernorm.weight"] = torch.ones_like(g_q)
        out["self_attn.kv_a_layernorm.weight"] = torch.ones_like(g_kv)
    elif (
        f"{prefix}.self_attn.q_proj.weight" in sd
        or f"{prefix}.self_attn.k_proj.weight" in sd
    ):
        raise NotImplementedError(
            f"{prefix}: dense (non-MLA) self-attention is not covered by "
            "this prototype; DeepSeek-V3-style all-MLA checkpoints have "
            "no such layers."
        )
    else:
        raise KeyError(f"{prefix}: no recognized self-attention weights")

    out["self_attn.o_proj.weight"] = src("self_attn.o_proj.weight").clone()

    # MoE router (exact logits, invariant 4) and expert input weights.
    if f"{prefix}.mlp.gate.weight" in sd:
        out["mlp.gate.weight"] = conj_in(src("mlp.gate.weight"))
    keep = {
        "self_attn.o_proj.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "mlp.gate.weight",
    }
    for key in sd:
        if not key.startswith(f"{prefix}."):
            continue
        suffix = key[len(prefix) + 1 :]
        if suffix in keep:
            out.setdefault(suffix, src(suffix).clone())
        elif suffix.endswith("down_proj.weight"):
            out[suffix] = src(suffix).clone()
        elif suffix.endswith(("gate_proj.weight", "up_proj.weight")):
            out[suffix] = conj_in(src(suffix))
    return out


def conjugate_state_dict(
    sd: dict[str, torch.Tensor],
    *,
    num_layers: int,
    seed: int = 0,
    make_rot_in=None,
    make_rot_kv=None,
    m=None,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Conjugate a full HF-layout state dict.

    Args:
        sd: canonical checkpoint tensors (any dtype; math in float64).
        num_layers: number of ``model.layers.{i}`` entries.
        seed: base seed for the per-layer Rademacher diagonals.
        make_rot_in: optional ``layer_idx -> R_in`` override (tests).
        make_rot_kv: optional ``layer_idx -> R_kv`` override (tests).
        m: preimported rotations module (tests).

    Returns:
        (conjugated state dict with embedded sign tensors, metadata).
    """
    m = m or _import_rotations()
    meta = {
        "format": META_FORMAT,
        "generator": "blockdiag(H_b) @ diag(rademacher)",
        "seed": seed,
        "folded": {
            "q_a_layernorm": "q_b_proj input columns",
            "kv_a_layernorm": "kv_b_proj input columns",
        },
        "layers": {},
    }
    out: dict[str, torch.Tensor] = {}
    layer_prefixes = set()
    for key in sd:
        if key.startswith("model.layers."):
            layer_prefixes.add(key.rsplit(".", 2)[0])
    for i in range(num_layers):
        p = f"model.layers.{i}"
        if p not in layer_prefixes:
            continue
        rot_in = (
            make_rot_in(i)
            if make_rot_in
            else m.BlockRandomizedHadamard.create(
                sd[f"{p}.input_layernorm.weight"].numel(), seed + 2 * i
            )
        )
        rot_kv = (
            make_rot_kv(i)
            if make_rot_kv
            else m.BlockRandomizedHadamard.create(
                sd[f"{p}.self_attn.kv_a_layernorm.weight"].numel(),
                seed + 2 * i + 1,
            )
        )
        for name, w in conjugate_layer(sd, p, rot_in, rot_kv, m).items():
            out[f"{p}.{name}"] = w
        out[f"{p}.rot_in_signs"] = rot_in.signs.float()
        out[f"{p}.rot_kv_signs"] = rot_kv.signs.float()
        meta["layers"][str(i)] = {"in": rot_in.state(), "kv": rot_kv.state()}
    for key, w in sd.items():
        if not key.startswith("model.layers."):
            out[key] = w.clone()
    return out, meta


# ---------------------------------------------------------------------------
# Tiny mock checkpoints for engine-level e2e tests
# ---------------------------------------------------------------------------


def _mock_config(num_layers, moe, rotated):
    return {
        "architectures": [
            "DeepSeekV4RotatedForCausalLM" if rotated else "DeepseekV3ForCausalLM"
        ],
        "model_type": "deepseek_v3",
        "hidden_size": 1024,
        "q_lora_rank": 512,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "num_attention_heads": 8,
        "num_hidden_layers": num_layers,
        "first_k_dense_replace": 1 if moe else num_layers,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 256,
        "intermediate_size": 512,
        "rms_norm_eps": 1e-6,
        "vocab_size": 128,
        "rope_theta": 10000.0,
        "max_position_embeddings": 512,
        "tie_word_embeddings": False,
        "scoring_func": "softmax",
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.0,
        "moe_layer_freq": 1,
    }


def build_mock_state_dict(num_layers: int, moe: bool, seed: int = 0):
    hidden, q_lora, kv_lora = 1024, 512, 512
    nope, rope, v, heads = 128, 64, 128, 8
    experts, inter, dense_inter = 8, 256, 512
    g = torch.Generator().manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float32) * 0.02

    def ones_gain(n):
        return 1.0 + torch.rand(n, generator=g, dtype=torch.float32).abs() * 0.1

    sd: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": randn(128, hidden),
        "model.norm.weight": ones_gain(hidden),
    }
    # Untied lm_head (construction-level tying only applies when the
    # checkpoint ships no lm_head tensor; real V3 checkpoints ship one).
    g2 = torch.Generator().manual_seed(seed + 999)
    sd["lm_head.weight"] = (
        torch.randn(128, hidden, generator=g2, dtype=torch.float32) * 0.02
    )
    for i in range(num_layers):
        p = f"model.layers.{i}"
        sd[f"{p}.input_layernorm.weight"] = ones_gain(hidden)
        sd[f"{p}.post_attention_layernorm.weight"] = ones_gain(hidden)
        sd[f"{p}.self_attn.q_a_proj.weight"] = randn(q_lora, hidden)
        sd[f"{p}.self_attn.q_a_layernorm.weight"] = ones_gain(q_lora)
        sd[f"{p}.self_attn.q_b_proj.weight"] = randn(heads * (nope + rope), q_lora)
        sd[f"{p}.self_attn.kv_a_proj_with_mqa.weight"] = randn(kv_lora + rope, hidden)
        sd[f"{p}.self_attn.kv_a_layernorm.weight"] = ones_gain(kv_lora)
        sd[f"{p}.self_attn.kv_b_proj.weight"] = randn(heads * (nope + v), kv_lora)
        sd[f"{p}.self_attn.o_proj.weight"] = randn(hidden, heads * v)
        if moe and i >= 1:
            sd[f"{p}.mlp.gate.weight"] = randn(experts, hidden)
            for e in range(experts):
                sd[f"{p}.mlp.experts.{e}.gate_proj.weight"] = randn(inter, hidden)
                sd[f"{p}.mlp.experts.{e}.up_proj.weight"] = randn(inter, hidden)
                sd[f"{p}.mlp.experts.{e}.down_proj.weight"] = randn(hidden, inter)
            sd[f"{p}.mlp.shared_experts.gate_proj.weight"] = randn(inter, hidden)
            sd[f"{p}.mlp.shared_experts.up_proj.weight"] = randn(inter, hidden)
            sd[f"{p}.mlp.shared_experts.down_proj.weight"] = randn(hidden, inter)
        else:
            sd[f"{p}.mlp.gate_proj.weight"] = randn(dense_inter, hidden)
            sd[f"{p}.mlp.up_proj.weight"] = randn(dense_inter, hidden)
            sd[f"{p}.mlp.down_proj.weight"] = randn(hidden, dense_inter)
    return sd


def make_tiny_mock(out_dir: Path, *, moe: bool, seed: int = 0) -> None:
    from safetensors.torch import save_file

    m = _import_rotations()
    num_layers = 4
    sd = build_mock_state_dict(num_layers, moe, seed)
    conj, meta = conjugate_state_dict(sd, num_layers=num_layers, seed=seed + 1, m=m)
    for sub, tensors, rotated in (
        ("canonical", sd, False),
        ("rotated", conj, True),
    ):
        d = out_dir / sub
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(
            json.dumps(_mock_config(num_layers, moe, rotated), indent=1)
        )
        save_file(
            {k: v.contiguous() for k, v in tensors.items()},
            str(d / "model.safetensors"),
        )
    (out_dir / "rotations.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote mock checkpoint twins (moe={moe}) under {out_dir}")


# ---------------------------------------------------------------------------
# Standalone prototype tier: torch modules used by --self-test
# ---------------------------------------------------------------------------


def _rope_interleaved(
    x: torch.Tensor, positions: torch.Tensor, base: float
) -> torch.Tensor:
    """GPT-NeoX-style=False (interleaved pairs) RoPE; DeepSeek's mode."""
    d = x.shape[-1]
    inv_freq = base ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)
    ang = positions.to(torch.float64)[:, None] * inv_freq[None, :]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2].double(), x[..., 1::2].double()
    o1 = x1 * cos[:, None, :] - x2 * sin[:, None, :]
    o2 = x1 * sin[:, None, :] + x2 * cos[:, None, :]
    return torch.stack((o1, o2), dim=-1).flatten(-2).to(x.dtype)


def _causal_mha(q, k, v, scale):
    """Reference causal attention. q/k: (T, H, dk), v: (T, H, dv)."""
    scores = torch.einsum("thd,shd->hts", q, k) * scale
    T = q.shape[0]
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask[None], float("-inf"))
    alpha = _softmax_precise(scores)
    return torch.einsum("hts,shd->thd", alpha, v)


def _softmax_precise(scores: torch.Tensor) -> torch.Tensor:
    wide = scores.double() if scores.dtype == torch.float64 else scores.float()
    return torch.softmax(wide, dim=-1).to(scores.dtype)


def _rms_scalar(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


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
    """Standalone rotated MLA layer (exact math, torch only).

    Mirrors the integrated module's contracts; per-head row layouts
    follow vLLM's MLA modules: ``q_b_proj`` heads are ``[nope | rope]``,
    ``kv_b_proj`` heads are ``[K_nope | V]``.
    """

    def __init__(self, cfg: RotatedMLAConfig, rot_in, rot_kv) -> None:
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
        self.w_o = nn.Parameter(torch.empty(d, cfg.num_heads * cfg.v_head_dim, dtype=dt))
        self.w_qk: torch.Tensor
        self.w_vo: torch.Tensor

    def load_conjugated(self, sd, prefix: str) -> None:
        def g(name):
            return sd[f"{prefix}.{name}"].to(self.cfg.dtype)

        with torch.no_grad():
            self.g_in.copy_(g("input_layernorm.weight"))
            self.w_q_a.copy_(g("self_attn.q_a_proj.weight"))
            self.w_kv_a.copy_(g("self_attn.kv_a_proj_with_mqa.weight"))
            self.w_q_b.copy_(g("self_attn.q_b_proj.weight"))
            self.w_kv_b.copy_(g("self_attn.kv_b_proj.weight"))
            self.w_o.copy_(g("self_attn.o_proj.weight"))
        self.prepare_decode_weights()

    def prepare_decode_weights(self) -> None:
        """Absorb W~_UK/W~_UV (rotation included) per head (invariant 3)."""
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

    def _project(self, hidden_states):
        """Returns (q_c, c_kv_rot, k_pe_raw); no FWHT inside projections."""
        m = _import_rotations()
        cfg = self.cfg
        h_rot = m.fused_rmsnorm_fwht(
            hidden_states, self.g_in, cfg.rms_norm_eps, self.rot_in
        )
        q_c_raw = h_rot @ self.w_q_a.t()
        kv = h_rot @ self.w_kv_a.t()
        c_kv_rot_raw, k_pe_raw = kv.split(
            [cfg.kv_lora_rank, cfg.qk_rope_head_dim], dim=-1
        )
        return (
            _rms_scalar(q_c_raw, cfg.rms_norm_eps),
            _rms_scalar(c_kv_rot_raw, cfg.rms_norm_eps),
            k_pe_raw,
        )

    def forward_prefill(self, positions, hidden_states):
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
        k = torch.cat(
            [kv[..., : cfg.qk_nope_head_dim], k_pe.expand(T, cfg.num_heads, -1)],
            dim=-1,
        )
        attn = _causal_mha(q, k, kv[..., cfg.qk_nope_head_dim :], self.scale)
        return attn.reshape(T, -1) @ self.w_o.t()

    def forward_decode(
        self,
        positions,
        hidden_states,
        kv_cache,
        block_table,
        seq_lens,
        slot_mapping,
        *,
        sim_int4: bool = False,
    ):
        m = _import_rotations()
        cfg = self.cfg
        q_c, c_kv_rot, k_pe_raw = self._project(hidden_states)
        k_pe = _rope_interleaved(k_pe_raw.unsqueeze(1), positions, cfg.rope_theta)
        m.turboquant_reshape_and_cache(
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
        context = m.turboquant_paged_attention(
            q_rot,
            kv_cache,
            self.scale,
            block_table=block_table,
            seq_lens=seq_lens,
            kv_lora_rank=cfg.kv_lora_rank,
        )
        return torch.einsum("thd,hde->te", context, self.w_vo)


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
    """Standalone rotated MoE layer (exact math, torch only)."""

    def __init__(self, cfg: RotatedMoEConfig, rot_in) -> None:
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
        sh_inter = inter * cfg.n_shared_experts
        self.sh_g = nn.Parameter(torch.empty(sh_inter, d, dtype=dt))
        self.sh_u = nn.Parameter(torch.empty(sh_inter, d, dtype=dt))
        self.sh_d = nn.Parameter(torch.empty(d, sh_inter, dtype=dt))

    def load_conjugated(self, sd, prefix: str) -> None:
        cfg = self.cfg
        inter = cfg.moe_intermediate_size

        def g(name):
            return sd[f"{prefix}.{name}"].to(cfg.dtype)

        with torch.no_grad():
            self.g_post.copy_(g("post_attention_layernorm.weight"))
            self.gate.copy_(g("mlp.gate.weight"))
            for e in range(cfg.n_routed_experts):
                self.w13[e, :inter].copy_(g(f"mlp.experts.{e}.gate_proj.weight"))
                self.w13[e, inter:].copy_(g(f"mlp.experts.{e}.up_proj.weight"))
                self.w2[e].copy_(g(f"mlp.experts.{e}.down_proj.weight"))
            self.sh_g.copy_(g("mlp.shared_experts.gate_proj.weight"))
            self.sh_u.copy_(g("mlp.shared_experts.up_proj.weight"))
            self.sh_d.copy_(g("mlp.shared_experts.down_proj.weight"))

    @staticmethod
    def _softmax_topk(logits, k, renormalize):
        probs = _softmax_precise(logits)
        weights, ids = torch.topk(probs, k, dim=-1)
        if renormalize:
            weights = weights / weights.sum(-1, keepdim=True)
        return weights, ids

    def forward(self, hidden_states):
        m = _import_rotations()
        cfg = self.cfg
        h_rot = m.fused_rmsnorm_fwht(
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


# ---------------------------------------------------------------------------
# Independent canonical references (self-test ground truth)
# ---------------------------------------------------------------------------


class ReferenceMLA:
    """Independent canonical MLA forward (HF math, no rotations)."""

    def __init__(self, sd, prefix, *, nope, rope, v, heads, eps):
        self.eps, self.num_heads, self.dims = eps, heads, (nope, rope, v)
        self.scale = (nope + rope) ** -0.5

        def g(n):
            return sd[f"{prefix}.{n}"].double()

        self.g_in = g("input_layernorm.weight")
        self.w_q_a, self.g_q = g("self_attn.q_a_proj.weight"), g(
            "self_attn.q_a_layernorm.weight"
        )
        self.w_q_b = g("self_attn.q_b_proj.weight")
        self.w_kv_a, self.g_kv = g("self_attn.kv_a_proj_with_mqa.weight"), g(
            "self_attn.kv_a_layernorm.weight"
        )
        self.w_kv_b, self.w_o = g("self_attn.kv_b_proj.weight"), g(
            "self_attn.o_proj.weight"
        )

    @staticmethod
    def _rms(x, gamma, eps):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gamma

    def heads(self, positions, x):
        nope, rope, v = self.dims
        T, heads = x.shape[0], self.num_heads
        hn = self._rms(x, self.g_in, self.eps)
        q_c = self._rms(hn @ self.w_q_a.t(), self.g_q, self.eps)
        kv = hn @ self.w_kv_a.t()
        c_raw, k_pe_raw = kv.split([kv.shape[-1] - rope, rope], dim=-1)
        c = self._rms(c_raw, self.g_kv, self.eps)
        k_pe = _rope_interleaved(k_pe_raw.unsqueeze(1), positions, 10000.0)
        q = (q_c @ self.w_q_b.t()).view(T, heads, nope + rope)
        q = torch.cat(
            [q[..., :nope], _rope_interleaved(q[..., nope:], positions, 10000.0)],
            dim=-1,
        )
        kvb = (c @ self.w_kv_b.t()).view(T, heads, nope + v)
        k = torch.cat([kvb[..., :nope], k_pe.expand(T, heads, rope)], dim=-1)
        return q, k, kvb[..., nope:]

    def prefill(self, positions, x):
        q, k, v = self.heads(positions, x)
        attn = _causal_mha(q, k, v, self.scale)
        return attn.reshape(x.shape[0], -1) @ self.w_o.t()

    def decode_row(self, positions_all, x_all, t):
        q, k, v = self.heads(positions_all, x_all)
        scores = torch.einsum("hd,shd->hs", q[t], k[: t + 1]) * self.scale
        alpha = torch.softmax(scores, dim=-1)
        return torch.einsum("hs,shd->hd", alpha, v[: t + 1]).reshape(-1) @ self.w_o.t()


class ReferenceMoE:
    """Independent canonical MoE forward (softmax top-k)."""

    def __init__(self, sd, prefix, *, experts, k, inter, eps, scaling=1.0):
        self.E = experts
        self.K = k
        self.inter = inter
        self.eps = eps
        self.scaling = scaling

        def g(n):
            return sd[f"{prefix}.{n}"].double()

        self.g_post = g("post_attention_layernorm.weight")
        self.gate = g("mlp.gate.weight")
        self.e_gate = torch.stack(
            [g(f"mlp.experts.{e}.gate_proj.weight") for e in range(experts)]
        )
        self.e_up = torch.stack(
            [g(f"mlp.experts.{e}.up_proj.weight") for e in range(experts)]
        )
        self.e_down = torch.stack(
            [g(f"mlp.experts.{e}.down_proj.weight") for e in range(experts)]
        )
        self.sh_g = g("mlp.shared_experts.gate_proj.weight")
        self.sh_u = g("mlp.shared_experts.up_proj.weight")
        self.sh_d = g("mlp.shared_experts.down_proj.weight")

    def forward(self, x):
        inter = self.inter
        hn = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.g_post
        logits = hn @ self.gate.t()
        probs = torch.softmax(logits, dim=-1)
        w, ids = torch.topk(probs, self.K, dim=-1)
        w = w / w.sum(-1, keepdim=True)
        y = torch.zeros_like(hn)
        for e in range(self.E):
            tok, slot = torch.where(ids == e)
            if tok.numel() == 0:
                continue
            expert = (
                F.silu(hn[tok] @ self.e_gate[e].t()) * (hn[tok] @ self.e_up[e].t())
            ) @ self.e_down[e].t()
            y.index_add_(0, tok, expert * w[tok, slot, None])
        shared = (
            F.silu(hn @ self.sh_g.t()) * (hn @ self.sh_u.t())
        ) @ self.sh_d.t()
        return y * self.scaling + shared, logits


# ---------------------------------------------------------------------------
# Self-test: exactness against the independent reference
# ---------------------------------------------------------------------------


def _build_tiny_state_dict(*, hidden, q_lora, kv_lora, nope, rope, v, heads, E, inter):
    torch.manual_seed(7)
    sd: dict[str, torch.Tensor] = {}
    p = "model.layers.0"
    g = torch.Generator().manual_seed(11)

    def randn(*shape, scale=0.02):
        return torch.randn(*shape, generator=g, dtype=torch.float64) * scale

    sd[f"{p}.input_layernorm.weight"] = 1.0 + randn(hidden, scale=0.1).abs()
    sd[f"{p}.post_attention_layernorm.weight"] = 1.0 + randn(hidden, scale=0.1).abs()
    sd[f"{p}.self_attn.q_a_proj.weight"] = randn(q_lora, hidden)
    sd[f"{p}.self_attn.q_a_layernorm.weight"] = 1.0 + randn(q_lora, scale=0.1).abs()
    sd[f"{p}.self_attn.q_b_proj.weight"] = randn(heads * (nope + rope), q_lora)
    sd[f"{p}.self_attn.kv_a_proj_with_mqa.weight"] = randn(kv_lora + rope, hidden)
    sd[f"{p}.self_attn.kv_a_layernorm.weight"] = 1.0 + randn(kv_lora, scale=0.1).abs()
    sd[f"{p}.self_attn.kv_b_proj.weight"] = randn(heads * (nope + v), kv_lora)
    sd[f"{p}.self_attn.o_proj.weight"] = randn(hidden, heads * v)
    sd[f"{p}.mlp.gate.weight"] = randn(E, hidden)
    for e in range(E):
        sd[f"{p}.mlp.experts.{e}.gate_proj.weight"] = randn(inter, hidden)
        sd[f"{p}.mlp.experts.{e}.up_proj.weight"] = randn(inter, hidden)
        sd[f"{p}.mlp.experts.{e}.down_proj.weight"] = randn(hidden, inter)
    sd[f"{p}.mlp.shared_experts.gate_proj.weight"] = randn(inter, hidden)
    sd[f"{p}.mlp.shared_experts.up_proj.weight"] = randn(inter, hidden)
    sd[f"{p}.mlp.shared_experts.down_proj.weight"] = randn(hidden, inter)
    sd["model.embed_tokens.weight"] = randn(64, hidden)
    sd["lm_head.weight"] = randn(64, hidden)
    return sd


def run_self_test() -> int:
    m = _import_rotations()
    hidden, q_lora, kv_lora = 1536, 384, 256  # hidden = 1024 + 512 blocks
    nope, rope, v, heads = 128, 64, 128, 4
    E, K, inter = 8, 2, 512
    eps, T, block_size = 1e-6, 13, 4
    cfg_mla = RotatedMLAConfig(
        hidden_size=hidden,
        q_lora_rank=q_lora,
        kv_lora_rank=kv_lora,
        qk_nope_head_dim=nope,
        qk_rope_head_dim=rope,
        v_head_dim=v,
        num_heads=heads,
        rms_norm_eps=eps,
        dtype=torch.float64,
    )
    cfg_moe = RotatedMoEConfig(
        hidden_size=hidden,
        moe_intermediate_size=inter,
        n_routed_experts=E,
        num_experts_per_tok=K,
        n_shared_experts=1,
        rms_norm_eps=eps,
        dtype=torch.float64,
    )
    sd = _build_tiny_state_dict(
        hidden=hidden,
        q_lora=q_lora,
        kv_lora=kv_lora,
        nope=nope,
        rope=rope,
        v=v,
        heads=heads,
        E=E,
        inter=inter,
    )
    # Moderate systematic latent outliers (rank-1 weight spikes spread
    # across all int4 groups): bulk coordinates still carry most of the
    # energy, which is the regime where max-scaled int4 actually degrades.
    kv_a = sd["model.layers.0.self_attn.kv_a_proj_with_mqa.weight"]
    for r in (0, 5, 70, 130, 200, 250):
        kv_a[r, :] *= 10.0

    # Anchor: block ops vs an independent dense Sylvester construction.
    rot = m.BlockRandomizedHadamard.create(hidden, seed=99)
    r = dense_from_rot(rot)
    x_probe = torch.randn(5, hidden, dtype=torch.float64)
    w_probe = torch.randn(64, hidden, dtype=torch.float64)
    v_probe = torch.randn(hidden, 64, dtype=torch.float64)
    assert torch.allclose(rot.apply(x_probe), x_probe @ r, atol=1e-12)
    assert torch.allclose(rot.apply_t(rot.apply(x_probe)), x_probe, atol=1e-12)
    w_t = w_probe.clone()
    m.BlockRandomizedHadamard.create(hidden, seed=99).right_mult_(w_t)
    assert torch.allclose(w_t, w_probe @ r, atol=1e-12), "right_mult_ mismatch"
    v_t = v_probe.clone()
    m.BlockRandomizedHadamard.create(hidden, seed=99).left_mult_t_(v_t)
    assert torch.allclose(v_t, r.t() @ v_probe, atol=1e-12), "left_mult_t_ mismatch"
    assert torch.allclose(
        rot.apply(x_probe).norm(dim=-1), x_probe.norm(dim=-1), atol=1e-9
    ), "norm not preserved"
    print("anchor: apply/apply_t/right_mult_/left_mult_t_ == dense H @ D")

    ref_mla = ReferenceMLA(
        sd, "model.layers.0", nope=nope, rope=rope, v=v, heads=heads, eps=eps
    )
    ref_moe = ReferenceMoE(sd, "model.layers.0", experts=E, k=K, inter=inter, eps=eps)
    positions = torch.arange(T)
    g = torch.Generator().manual_seed(5)
    x = torch.randn(T, hidden, generator=g, dtype=torch.float64)
    x[:, 3] *= 80.0  # activation outlier for the int4 demo

    def run_decode(mla, sim_int4, track_diffs):
        num_blocks = (T + block_size - 1) // block_size
        head_kv = kv_lora + rope
        cache = torch.zeros(num_blocks, block_size, 1, head_kv, dtype=torch.float64)
        block_table = torch.arange(num_blocks).view(1, num_blocks)
        last, worst = None, 0.0
        for t in range(T):
            got = mla.forward_decode(
                positions[t : t + 1],
                x[t : t + 1],
                cache,
                block_table,
                torch.tensor([t + 1]),
                torch.tensor([t]),
                sim_int4=sim_int4,
            )[0]
            last = got
            if track_diffs:
                ref = ref_mla.decode_row(positions, x, t)
                worst = max(worst, (got - ref).abs().max().item())
        return last, worst

    rows = []
    for tag, ident in (("identity R", True), ("randomized R", False)):
        if ident:
            mk_in = lambda i: m.BlockRandomizedHadamard.identity(hidden)  # noqa: E731
            mk_kv = lambda i: m.BlockRandomizedHadamard.identity(kv_lora)  # noqa: E731
        else:
            mk_in = lambda i: m.BlockRandomizedHadamard.create(  # noqa: E731
                hidden, seed=1000 + 2 * i
            )
            mk_kv = lambda i: m.BlockRandomizedHadamard.create(  # noqa: E731
                kv_lora, seed=1001 + 2 * i
            )
        conj, meta = conjugate_state_dict(
            sd, num_layers=1, make_rot_in=mk_in, make_rot_kv=mk_kv, m=m
        )
        rot_in = m.BlockRandomizedHadamard.from_state(
            meta["layers"]["0"]["in"], torch.float64
        )
        rot_kv = m.BlockRandomizedHadamard.from_state(
            meta["layers"]["0"]["kv"], torch.float64
        )
        mla = DeepSeekV4RotatedMLA(cfg_mla, rot_in, rot_kv)
        mla.load_conjugated(conj, "model.layers.0")
        moe = DeepSeekV4RotatedMoE(cfg_moe, rot_in)
        moe.load_conjugated(conj, "model.layers.0")

        d_prefill = (
            mla.forward_prefill(positions, x) - ref_mla.prefill(positions, x)
        ).abs().max().item()

        _, d_decode = run_decode(mla, sim_int4=False, track_diffs=True)

        x_res = x + ref_mla.prefill(positions, x)
        got_moe = moe(x_res)
        want_moe, ref_logits = ref_moe.forward(x_res)
        d_moe = (got_moe - want_moe).abs().max().item()
        h_rot = m.fused_rmsnorm_fwht(x_res, moe.g_post, eps, rot_in)
        d_router = (h_rot @ moe.gate.t() - ref_logits).abs().max().item()

        ref_last = ref_mla.decode_row(positions, x, T - 1)
        out_int4, _ = run_decode(mla, sim_int4=True, track_diffs=False)
        rel_int4 = (
            (out_int4 - ref_last).abs().max() / ref_last.abs().max().clamp_min(1e-9)
        ).item()
        c_kv_rot = mla._project(x)[1]
        concentration = (
            c_kv_rot.abs().max() / c_kv_rot.pow(2).mean().sqrt().clamp_min(1e-12)
        ).item()
        rows.append(
            (tag, d_prefill, d_decode, d_moe, d_router, rel_int4, concentration)
        )

    tol = 1e-9
    for tag, d_prefill, d_decode, d_moe, d_router, _, _ in rows:
        assert d_prefill < tol, f"{tag}: MLA prefill mismatch {d_prefill:.3e}"
        assert d_decode < tol, f"{tag}: absorbed decode mismatch {d_decode:.3e}"
        assert d_moe < tol, f"{tag}: MoE mismatch {d_moe:.3e}"
        assert d_router < tol, f"{tag}: router logits mismatch {d_router:.3e}"

    print(f"\nfloat64 exactness vs independent reference (tol {tol:.0e}):")
    for tag, d_prefill, d_decode, d_moe, d_router, rel_int4, conc in rows:
        print(
            f"  {tag:<13} prefill {d_prefill:.2e}  decode {d_decode:.2e}"
            f"  moe {d_moe:.2e}  router {d_router:.2e}"
            f"  | latent max/rms {conc:6.1f}  int4-rel-err {rel_int4:.3f}"
        )
    print(
        "\nint4 demo: relative decode error with simulated groupwise-int4"
        " KV latents,\nwith vs without the randomized rotation (same"
        " moderate weight spikes and activation outlier)."
    )
    print("self-test passed")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", help="HF checkpoint directory (*.safetensors)")
    ap.add_argument("--output", help="output directory for conjugated ckpt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16", help="save dtype")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument(
        "--make-tiny-mock",
        metavar="DIR",
        help="write canonical + conjugated tiny mock checkpoints",
    )
    ap.add_argument(
        "--moe",
        action="store_true",
        help="with --make-tiny-mock: include MoE layers",
    )
    args = ap.parse_args(argv)

    if args.self_test:
        return run_self_test()
    if args.make_tiny_mock:
        make_tiny_mock(Path(args.make_tiny_mock), moe=args.moe, seed=args.seed)
        return 0
    if not args.input or not args.output:
        ap.error(
            "--input and --output are required (or use --self-test / "
            "--make-tiny-mock)"
        )

    from safetensors.torch import load_file, save_file

    in_dir, out_dir = Path(args.input), Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((in_dir / "config.json").read_text())
    num_layers = cfg["num_hidden_layers"]
    sd = {}
    for f in sorted(in_dir.glob("*.safetensors")):
        sd.update(load_file(str(f)))
    if not sd:
        raise SystemExit(f"no safetensors found in {in_dir}")

    conj, meta = conjugate_state_dict(sd, num_layers=num_layers, seed=args.seed)
    dt = getattr(torch, args.dtype)
    save_file(
        {k: v.to(dt).contiguous() for k, v in conj.items()},
        str(out_dir / "model.safetensors"),
    )
    (out_dir / "rotations.json").write_text(json.dumps(meta, indent=1))
    for aux in in_dir.iterdir():
        if aux.suffix in {".json", ".txt", ".py"}:
            shutil.copy(aux, out_dir / aux.name)
    print(f"wrote {len(conj)} tensors to {out_dir} (dtype={dt})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
