# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Driver for test_deepseek_v4_rotated_e2e.py.

Runs one scenario against a mock checkpoint pair, each in a fresh
process (vLLM global state -- distributed groups, static forward
context -- does not tolerate multiple engine lifecycles per process).
Prints a JSON result blob on stdout.

Usage:
    python deepseek_v4_rotated_e2e_driver.py <scenario> <mock_dir>

Scenarios:
    equivalence   canonical vs rotated: logits diff, greedy tokens,
                  fwht-site counts (invariants 1 & 3)
    decode_only   seqlen=1 prompt through the rotated engine
    packed        turboquant_4bit_nc vs exact-stride greedy tokens
    hybrid        rotated arch on a sign-less checkpoint (must fail)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROMPT = [1, 5, 9, 33, 72, 11, 3]
NUM_LAYERS = 4


def _make_llm(model, **overrides):
    from vllm import LLM

    kwargs = dict(
        model=model,
        skip_tokenizer_init=True,
        enforce_eager=True,
        dtype="float32",
        max_model_len=256,
        max_num_seqs=4,
        gpu_memory_utilization=0.4,
        kv_cache_memory_bytes=64 * 2**20,
        attention_backend="TURBOQUANT_MLA",
    )
    kwargs.update(overrides)
    return LLM(**kwargs)


def _generate(llm, prompt, max_tokens):
    from vllm import SamplingParams

    out = llm.generate(
        prompts=[dict(prompt_token_ids=prompt)],
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0.0),
    )
    return list(out[0].outputs[0].token_ids)


def _stash_logits(llm):
    model = (
        llm.llm_engine.engine_core.engine_core.model_executor.driver_worker
        .model_runner.model
    )
    stash = {}
    orig = model.compute_logits

    def wrapped(hidden_states):
        logits = orig(hidden_states)
        stash.setdefault("logits", logits.detach().clone())
        return logits

    model.compute_logits = wrapped
    return stash


def _rotations():
    from vllm.model_executor.layers import rotations

    return rotations


def scenario_equivalence(mock_dir):
    can = _make_llm(str(mock_dir / "canonical"))
    stash_can = _stash_logits(can)
    _rotations().reset_fwht_counter()
    tokens_can = _generate(can, PROMPT, 8)
    sites_can = _rotations().FWHT_SITE_COUNT

    rot = _make_llm(str(mock_dir / "rotated"))
    stash_rot = _stash_logits(rot)
    _rotations().reset_fwht_counter()
    tokens_rot = _generate(rot, PROMPT, 8)
    sites_rot = _rotations().FWHT_SITE_COUNT

    la = stash_can["logits"][0].double()
    lb = stash_rot["logits"][0].double()
    return {
        "tokens_canonical": tokens_can,
        "tokens_rotated": tokens_rot,
        "logits_max_abs_diff": float((la - lb).abs().max()),
        "sites_canonical": sites_can,
        "sites_rotated": sites_rot,
        "expected_sites_rotated": 8 * NUM_LAYERS * 2,
    }


def scenario_decode_only(mock_dir):
    rot = _make_llm(str(mock_dir / "rotated"))
    tokens = _generate(rot, [7], 12)
    return {"tokens": tokens}


def scenario_packed(mock_dir):
    rot = _make_llm(
        str(mock_dir / "rotated"), kv_cache_dtype="turboquant_4bit_nc"
    )
    packed = _generate(rot, PROMPT, 8)
    del rot
    exact_llm = _make_llm(str(mock_dir / "rotated"))
    exact = _generate(exact_llm, PROMPT, 8)
    return {"packed_tokens": packed, "exact_tokens": exact}


def scenario_hybrid(hybrid_dir):
    try:
        _make_llm(str(hybrid_dir))
    except ValueError as e:
        return {"failed_as_expected": "rot_in_signs" in str(e), "error": str(e)[:200]}
    return {"failed_as_expected": False, "error": ""}


def main() -> int:
    scenario = sys.argv[1]
    target = Path(sys.argv[2])
    if scenario == "equivalence":
        result = scenario_equivalence(target)
    elif scenario == "decode_only":
        result = scenario_decode_only(target)
    elif scenario == "packed":
        result = scenario_packed(target)
    elif scenario == "hybrid":
        result = scenario_hybrid(target)
    else:
        raise SystemExit(f"unknown scenario {scenario}")
    print("RESULT_JSON:" + json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
