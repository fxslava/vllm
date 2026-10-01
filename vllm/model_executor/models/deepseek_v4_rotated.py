# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek (V3-family MLA + MoE) with offline-conjugated randomized
Hadamard rotations: ``DeepSeekV4RotatedForCausalLM``.

NOTE: this is the rotated-pipeline prototype architecture. It reuses the
DeepSeek-V2/V3 execution graph (MLA absorption, paged KV cache, FusedMoE)
and is NOT derived from the DeepseekV4 sparse-MLA architecture in
``vllm/models/deepseek_v4``; the name only brands the rotation prototype.

Numerical design (full derivation in
``vllm/model_executor/layers/rotations.py`` and
``tools/conjugate_deepseek.py``):

1. The residual stream is CANONICAL. Each of the two layer-entry
   ``RMSNorm`` modules is replaced by ``RotatedRMSNorm`` =
   fused RMSNorm + randomized FWHT (norm preserving, invariant 1). Both
   entry norms of a layer share one per-layer ``R_in`` loaded from the
   checkpoint tensor ``model.layers.N.rot_in_signs``.
2. Every weight consuming the rotated norm output is conjugated OFFLINE
   (``tools/conjugate_deepseek.py``): ``q_a_proj``/``kv_a_proj_with_mqa``
   on the input side, the latent rows of ``kv_a_proj_with_mqa`` with
   ``R_kv`` (rope rows canonical, invariant 2), ``kv_b_proj`` input
   columns with ``R_kv`` + the folded ``kv_a_layernorm`` gain,
   ``q_b_proj`` with the folded ``q_a_layernorm`` gain, and the MoE
   router / gate / up projections with ``R_in`` (exact top-k logits,
   invariant 4).
3. Because the ``q_a_layernorm``/``kv_a_layernorm`` gains are folded
   into ``q_b_proj``/``kv_b_proj`` offline, the checkpoint stores ones
   for them; the standard MLA wrapper's ``RMSNorm(x)`` with unit gain
   then degenerates to the required ``x / rms(x)`` (the scalar is
   rotation invariant), and the cache stores the RMS-normalized rotated
   latent.
4. Decode absorption: the MLA backends fold ``kv_b_proj`` (which already
   carries ``R_kv``) into the query, so absorbed queries land in the
   rotated latent space and scores run directly against cached rotated
   latents (invariant 3) -- zero runtime FWHT in any projection, which
   ``layers.rotations.FWHT_SITE_COUNT`` verifies at runtime.

Weight loading: identical to DeepSeek-V3 (the conjugated checkpoint is
HF-layout) plus two extra sign tensors per layer (``rot_in_signs``,
``rot_kv_signs``); the latter is stashed on the decoder layer for the
TurboQuant attention backend and tests. TP-safe: all offline
conjugations act on input dims of column-parallel weights or on
replicated weights, so vLLM's standard output-dim sharding applies
unchanged.
"""

from __future__ import annotations

import re

import torch

from vllm.model_executor.layers.rotations import RotatedRMSNorm
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2DecoderLayer,
    DeepseekV2Model,
    DeepseekV3ForCausalLM,
)
from vllm.model_executor.models.utils import AutoWeightsLoader, PPMissingLayer

_SIGN_RE = re.compile(r"model\.layers\.(\d+)\.(rot_in_signs|rot_kv_signs)")


class DeepSeekV4RotatedModel(DeepseekV2Model):
    """DeepSeek-V2/V3 model whose decoder layers rotate at both entry
    norms. Built by post-init module surgery on the stock layers so that
    every other subsystem (MLA wrapper, MoE, PP, TP, spec decode) sees a
    standard ``DeepseekV2DecoderLayer``."""

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        for layer in self.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            assert isinstance(layer, DeepseekV2DecoderLayer)
            layer.input_layernorm = RotatedRMSNorm.from_rmsnorm(
                layer.input_layernorm
            )
            layer.post_attention_layernorm = RotatedRMSNorm.from_rmsnorm(
                layer.post_attention_layernorm
            )


class DeepSeekV4RotatedForCausalLM(DeepseekV3ForCausalLM):
    """Rotated DeepSeek (V3-family) causal LM. Requires a checkpoint
    produced by ``tools/conjugate_deepseek.py`` (weights conjugated +
    ``rot_*_signs`` tensors embedded)."""

    model_cls = DeepSeekV4RotatedModel

    def load_weights(self, weights) -> set[str]:
        signs: dict[int, dict[str, torch.Tensor]] = {}
        rest: list[tuple[str, torch.Tensor]] = []
        for name, loaded in weights:
            m = _SIGN_RE.fullmatch(name)
            if m:
                signs.setdefault(int(m.group(1)), {})[m.group(2)] = loaded
            else:
                rest.append((name, loaded))

        for idx, layer in enumerate(self.model.layers):
            if isinstance(layer, PPMissingLayer):
                continue
            layer_signs = signs.pop(idx, None)
            if layer_signs is None or "rot_in_signs" not in layer_signs:
                raise ValueError(
                    f"model.layers.{idx}: DeepSeekV4RotatedForCausalLM "
                    "requires 'rot_in_signs' (and expects 'rot_kv_signs') "
                    "from a checkpoint conjugated by "
                    "tools/conjugate_deepseek.py"
                )
            device = layer.input_layernorm.weight.device
            rot_in = layer_signs["rot_in_signs"].to(device)
            layer.input_layernorm.set_rotation(rot_in)
            layer.post_attention_layernorm.set_rotation(rot_in)
            # Kept for TurboQuant attention kernels / tests; the runtime
            # attention path carries R_kv inside the conjugated weights.
            layer.rot_kv_signs = layer_signs.get(
                "rot_kv_signs", torch.empty(0)
            ).to(device)
        if signs:
            raise ValueError(f"unexpected rotation sign tensors: {sorted(signs)}")

        loader = AutoWeightsLoader(self)
        return loader.load_weights(rest)
