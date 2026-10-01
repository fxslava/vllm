# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant MLA attention backend (torch reference stub).

Routes MLA decode attention and KV-cache writes through the torch
reference fallbacks in ``vllm/model_executor/layers/rotations.py``
instead of compiled kernels, proving the rotated-latent contracts on
the live engine:

- The cache holds the RMS-normalized ``R_kv``-rotated latent plus the
  canonical RoPE key ``k_pe`` (invariant 2): the model's conjugated
  weights produce both, so this backend performs no rotation itself.
- Absorbed decode queries already live in the rotated latent space
  (invariant 3): scores are plain dot products against the cached
  latents and the returned context is a weighted sum of ROTATED
  latents; the parent scaffolding applies ``kv_b_proj``/``o_proj``.

Cache strides (``--kv-cache-dtype``):

- ``auto`` / ``bfloat16`` / ``float16`` / ``float32``: the standard MLA
  stride ``(kv_lora + rope) * dtype``. Exact math; used by the e2e
  equivalence tests. ``VLLM_TURBOQUANT_MLA_SIMULATE_INT4=1`` additionally
  round-trips the latent through simulated groupwise int4 (robustness
  measurements only).
- ``turboquant_4bit_nc``: the packed 4-bit latent stride, expressed via
  the sanctioned ``AttentionSpec.state_content_bytes`` override (the
  same mechanism the GQA TurboQuant backend uses), so the allocator
  sizes pages from the true byte footprint::

      [ int4-packed latent (ceil(kv_lora/2) B) | per-token fp16 scale
        (2 B) | k_pe (rope * itemsize B) ] rounded up to 16 B.

  The stub then really packs/unpacks nibbles in a uint8 view of the
  cache. This is a placeholder layout for the production TurboQuant
  kernel (groupwise fp8 scales + fp8 k_pe) chosen so the allocation
  path is byte-exact today.

The latent/rope split is not carried by ``MLAAttentionSpec``; the
spec-side packing math therefore reads the rope dim from
``VLLM_TURBOQUANT_MLA_ROPE_DIM`` (default 64, every DeepSeek MLA model
to date). Selection: ``--attention-backend TURBOQUANT_MLA`` or any
``turboquant_*`` ``--kv-cache-dtype`` on platforms that route here.
"""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar

import torch

from vllm import envs
from vllm.config.cache import CacheDType
from vllm.model_executor.layers.rotations import (
    simulate_int4,
    turboquant_paged_attention,
)
from vllm.v1.attention.backends.mla.cpu_mla import (
    CPUMLABackend,
    CPUMLAImpl,
    MLACommonMetadata,
)

_ALIGN = 16


def _packed_state_bytes(kv_lora: int, rope: int, itemsize: int) -> int:
    n = (kv_lora + 1) // 2 + 2 + rope * itemsize
    return (n + _ALIGN - 1) // _ALIGN * _ALIGN


class TurboQuantMLABackend(CPUMLABackend):
    """Backend descriptor routing MLA to the TurboQuant torch stub."""

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "float16",
        "float32",
        "turboquant_4bit_nc",
    ]

    @staticmethod
    def get_name() -> str:
        return "TURBOQUANT_MLA"

    @staticmethod
    def get_impl_cls() -> type["TurboQuantMLAImpl"]:
        return TurboQuantMLAImpl

    @classmethod
    def customize_spec(cls, spec):
        """Override the MLA state footprint with the packed 4-bit stride.

        Only active for turboquant cache dtypes; every other dtype keeps
        the standard MLAAttentionSpec untouched.
        """
        if spec.state_content_bytes is not None:
            return spec
        if not spec.kv_quant_mode.is_turboquant:
            return spec
        rope = envs.VLLM_TURBOQUANT_MLA_ROPE_DIM
        kv_lora = spec.head_size - rope
        if kv_lora <= 0 or kv_lora % 2:
            raise ValueError(
                f"turboquant MLA packing needs a positive even latent dim, "
                f"got head_size={spec.head_size}, rope={rope}"
            )
        from vllm.v1.kv_cache_interface import get_dtype_size

        return replace(
            spec,
            state_content_bytes=_packed_state_bytes(
                kv_lora, rope, get_dtype_size(spec.dtype)
            ),
        )


class TurboQuantMLAImpl(CPUMLAImpl):
    """Torch-stub MLA implementation (reference quality, no kernels)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._tq_packed = self.kv_cache_dtype.startswith("turboquant")
        self._tq_sim_int4 = bool(envs.VLLM_TURBOQUANT_MLA_SIMULATE_INT4)

    # ------------------------------------------------------------------
    # KV cache write
    # ------------------------------------------------------------------
    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        if kv_cache.numel() == 0:
            return
        slots = slot_mapping.flatten().to(torch.long)
        valid = slots >= 0
        if not valid.all():
            slots, kv_c_normed, k_pe = (
                slots[valid],
                kv_c_normed[valid],
                k_pe[valid],
            )
        if slots.numel() == 0:
            return
        if self._tq_packed:
            self._pack_latents(kv_c_normed, k_pe, kv_cache, slots)
            return
        if self._tq_sim_int4:
            kv_c_normed = simulate_int4(kv_c_normed.float()).to(kv_c_normed.dtype)
        flat = kv_cache.view(-1, 1, kv_cache.shape[-1])
        packed = torch.cat(
            [kv_c_normed, k_pe.squeeze(1).to(flat.dtype)], dim=-1
        ).unsqueeze(1)
        flat[slots] = packed.to(flat.dtype)

    def _row_bytes(self, kv_cache: torch.Tensor) -> int:
        """Bytes per cached token in the ALLOCATED tensor.

        NOTE (fail-fast finding): ``customize_spec`` below sets
        ``state_content_bytes`` to the packed footprint, and that override
        is the sanctioned mechanism the GQA TurboQuant backend uses -- but
        the MLA spec/allocator path (MLAAttentionSpec -> strided HND
        tensor) derives the allocation purely from ``head_size * dtype``
        and IGNORES it (verified: identical block counts and tensor
        shapes with ``auto`` vs ``turboquant_4bit_nc``). Until the MLA
        allocator grows byte-level state-stride support (as the GQA path
        did via effective head sizes), packed mode therefore lays the
        packed row into the first bytes of each token's standard-stride
        region: correct, non-overlapping, but not yet memory-saving.
        """
        row = kv_cache.shape[-1] * kv_cache.element_size()
        packed = _packed_state_bytes(
            self.kv_lora_rank, self.qk_rope_head_dim, kv_cache.element_size()
        )
        if packed > row:
            raise ValueError(
                f"turboquant MLA packed row ({packed} B) exceeds the "
                f"allocated per-token stride ({row} B); the allocator "
                "did not honor AttentionSpec.state_content_bytes"
            )
        return row

    def _pack_latents(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slots: torch.Tensor,
    ) -> None:
        """Pack [int4 latent | fp16 scale | k_pe] into each token's row."""
        scale = (
            kv_c_normed.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 7.0
        ).half()  # (T, 1)
        q = torch.round(kv_c_normed.float() / scale.float()).clamp_(-7, 7)
        nib = (q + 8).to(torch.uint8)  # (T, kv_lora) in [1, 15]
        lat_bytes = nib[..., 0::2] | (nib[..., 1::2] << 4)  # (T, kv/2)
        scale_bytes = scale.view(torch.uint8)  # (T, 2)
        pe_bytes = k_pe.squeeze(1).to(kv_cache.dtype).view(torch.uint8)
        packed = _packed_state_bytes(
            self.kv_lora_rank, self.qk_rope_head_dim, kv_cache.element_size()
        )
        row = torch.cat([lat_bytes, scale_bytes, pe_bytes], dim=-1)
        if row.shape[-1] > packed:
            raise ValueError(
                f"turboquant MLA packed row ({row.shape[-1]} B) exceeds the "
                f"packed state size ({packed} B)"
            )
        if row.shape[-1] < packed:
            row = torch.nn.functional.pad(row, (0, packed - row.shape[-1]))
        row_bytes = self._row_bytes(kv_cache)
        # Slice each token's packed row into its (possibly larger)
        # allocated region: view the cache as (N, row_bytes) bytes and
        # write the packed row into the first ``packed`` byte columns.
        u8 = kv_cache.view(torch.uint8)
        total_rows = u8.numel() // row_bytes
        regions = u8.view(total_rows, row_bytes)
        regions[slots, :packed] = row.to(regions.dtype)

    def _unpack_latents(
        self, kv_cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (c_rot, k_pe) dequantized from the packed rows."""
        row_bytes = self._row_bytes(kv_cache)
        u8 = kv_cache.view(torch.uint8).view(-1, row_bytes)
        n_bytes = self.kv_lora_rank // 2
        lat = u8[:, :n_bytes].to(torch.int32)
        lo = (lat & 15).to(torch.float32) - 8.0
        hi = (lat >> 4).to(torch.float32) - 8.0
        q = torch.stack((lo, hi), dim=-1).flatten(-2)  # (N, kv_lora)
        scale = (
            u8[:, n_bytes : n_bytes + 2]
            .contiguous()
            .view(torch.float16)
            .to(torch.float32)
        )  # (N, 1)
        c_rot = q * scale
        pe_off = n_bytes + 2
        pe_end = pe_off + self.qk_rope_head_dim * kv_cache.element_size()
        k_pe = u8[:, pe_off:pe_end].contiguous().view(kv_cache.dtype)
        return c_rot.to(kv_cache.dtype), k_pe

    # ------------------------------------------------------------------
    # Decode (MQA)
    # ------------------------------------------------------------------
    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        layer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert attn_metadata.decode is not None
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)
        q = q.contiguous()
        block_table = attn_metadata.decode.block_table.to(torch.long)
        seq_lens = attn_metadata.decode.seq_lens.to(torch.long)
        if self._tq_packed:
            out = self._decode_packed(q, kv_c_and_k_pe_cache, block_table, seq_lens)
        else:
            out = turboquant_paged_attention(
                q,
                kv_c_and_k_pe_cache,
                self.scale,
                block_table=block_table,
                seq_lens=seq_lens,
                kv_lora_rank=self.kv_lora_rank,
            )
        return out, None

    def _decode_packed(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        block_size = kv_cache.shape[1]
        c_all, kpe_all = self._unpack_latents(kv_cache)
        c_all, kpe_all = c_all.to(q.dtype), kpe_all.to(q.dtype)
        num_tokens, _, _ = q.shape
        out = torch.empty(
            num_tokens, q.shape[1], self.kv_lora_rank, dtype=q.dtype
        )
        for t in range(num_tokens):
            seq_len = int(seq_lens[t])
            n_blocks = (seq_len + block_size - 1) // block_size
            slot_ids = torch.cat(
                [
                    torch.arange(
                        int(b) * block_size,
                        (int(b) + 1) * block_size,
                        device=q.device,
                    )
                    for b in block_table[t, :n_blocks]
                ],
                dim=0,
            )[:seq_len]
            c_rot, k_pe = c_all[slot_ids], kpe_all[slot_ids]
            qt = q[t]
            scores = (qt[:, None, : self.kv_lora_rank] * c_rot[None]).sum(-1) + (
                qt[:, None, self.kv_lora_rank :] * k_pe[None]
            ).sum(-1)
            alpha = torch.softmax((scores * self.scale).float(), dim=-1).to(q.dtype)
            out[t] = alpha @ c_rot
        return out
