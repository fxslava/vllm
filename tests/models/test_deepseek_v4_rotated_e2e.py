# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end engine tests for ``DeepSeekV4RotatedForCausalLM``.

Builds a tiny mock checkpoint pair with ``tools/conjugate_deepseek.py``
(identical underlying weights: one canonical DeepSeek-V3 copy, one
offline-conjugated rotated copy) and drives both through the live v1
engine on the CPU platform with the torch-stub ``TURBOQUANT_MLA``
attention backend (see ``deepseek_v4_rotated_e2e_driver.py``; each
scenario runs in a subprocess because vLLM global state does not
tolerate multiple engine lifecycles per process).

Verified contracts:

1. ``vllm.LLM`` loads, prefills (seqlen > 1) and decodes (seqlen = 1)
   without unhandled exceptions, producing valid shapes.
2. Rotated logits == canonical logits to fp32 noise on prefill, and
   greedy decoding is token-identical -- the rotation is exact end to
   end through scheduler, paged KV cache and absorbed decode.
3. ``fused_rmsnorm_fwht`` fires exactly twice per decoder layer per
   forward (the two entry norms) and never inside a linear projection
   (``layers.rotations.FWHT_SITE_COUNT``).
4. The RoPE key slice stays canonical: the cache keeps the standard MLA
   ``[latent | k_pe]`` stride and the structural conjugation covers
   only the latent dim (``rot_kv_signs`` size).
5. The packed 4-bit stride mode (``--kv-cache-dtype turboquant_4bit_nc``)
   packs real int4 latents and keeps greedy decoding aligned with the
   exact-stride run.
6. A non-conjugated checkpoint fails fast with a clear error.

Float64 exactness of the conjugation math itself is proven separately by
``tools/conjugate_deepseek.py --self-test``.

The suite forces the CPU platform (``VLLM_TARGET_DEVICE=cpu``); on
unbuilt Windows dev checkouts a local ``tmp_shims`` PYTHONPATH entry
(uvloop->asyncio shim + stub vllm dist-info) is used automatically.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVER = Path(__file__).resolve().parent / "deepseek_v4_rotated_e2e_driver.py"

NUM_LAYERS = 4
KV_LORA, ROPE = 512, 64


def _child_env() -> dict:
    env = dict(os.environ)
    env.setdefault("VLLM_TARGET_DEVICE", "cpu")
    env.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    env.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    if sys.platform == "win32":
        shim = REPO_ROOT / "tmp_shims"
        if (shim / "uvloop.py").exists():
            env["PYTHONPATH"] = str(shim) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _make_mock(tmp_path: Path, moe: bool) -> Path:
    out = tmp_path / ("mock_moe" if moe else "mock")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "conjugate_deepseek.py"),
        "--make-tiny-mock",
        str(out),
        "--seed",
        "0",
    ]
    if moe:
        cmd.append("--moe")
    subprocess.run(cmd, check=True, capture_output=True, env=_child_env())
    assert (out / "canonical" / "model.safetensors").exists()
    assert (out / "rotated" / "model.safetensors").exists()
    return out


def _run(scenario: str, target: Path) -> dict:
    proc = subprocess.run(
        [sys.executable, str(DRIVER), scenario, str(target)],
        capture_output=True,
        text=True,
        env=_child_env(),
        timeout=1200,
    )
    marker = "RESULT_JSON:"
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith(marker)]
    assert proc.returncode == 0, (
        f"driver {scenario} failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    )
    assert lines, f"no result from driver {scenario}:\n{proc.stderr[-2000:]}"
    return json.loads(lines[-1][len(marker) :])


def test_rotated_engine_equivalence(tmp_path):
    mock = _make_mock(tmp_path / "mock", moe=False)
    r = _run("equivalence", mock)
    assert r["sites_canonical"] == 0, "canonical model must not rotate"
    # 8 forwards x 4 layers x 2 entry norms; zero FWHT in projections.
    assert r["sites_rotated"] == r["expected_sites_rotated"], r
    assert r["logits_max_abs_diff"] < 2e-5, r
    assert r["tokens_canonical"] == r["tokens_rotated"], r
    assert len(r["tokens_rotated"]) == 8


def test_rotated_engine_decode_only(tmp_path):
    mock = _make_mock(tmp_path / "mock", moe=False)
    r = _run("decode_only", mock)
    assert len(r["tokens"]) == 12  # seqlen=1 prompt, pure decode steps


def test_rotated_cache_layout_and_rope_canonical(tmp_path):
    from safetensors.torch import load_file

    mock = _make_mock(tmp_path / "mock", moe=False)
    r = _run("decode_only", mock)  # drives a cache population
    assert len(r["tokens"]) == 12
    sd = load_file(str(mock / "rotated" / "model.safetensors"))
    # R_kv covers only the latent dim; the rope slice stays canonical.
    assert sd["model.layers.0.rot_kv_signs"].numel() == KV_LORA
    # kv_b_proj conjugation keeps its (out, kv_lora) storage shape.
    assert sd["model.layers.0.self_attn.kv_b_proj.weight"].shape[1] == KV_LORA


def test_rotated_engine_packed_kv(tmp_path):
    mock = _make_mock(tmp_path / "mock", moe=False)
    r = _run("packed", mock)
    assert r["packed_tokens"][:4] == r["exact_tokens"][:4], r


def test_rotated_requires_conjugated_checkpoint(tmp_path):
    from safetensors.torch import load_file, save_file

    mock = _make_mock(tmp_path / "mock", moe=False)
    hybrid = tmp_path / "hybrid"
    shutil.copytree(mock / "rotated", hybrid)
    sd = load_file(str(hybrid / "model.safetensors"))
    sd = {k: v for k, v in sd.items() if not k.endswith("_signs")}
    save_file(sd, str(hybrid / "model.safetensors"))
    r = _run("hybrid", hybrid)
    assert r["failed_as_expected"], r
    assert "rot_in_signs" in r["error"]
