# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Runs Llama3.1 on a tiny checkpoint and compares it to previously generated
golden values.
"""

import os
from pathlib import Path

import pytest
from evaluate_llama import (
    PROMPTS,
    NumpyDecoder,
    build_config,
    compare_values,
    find_runtime_path,
    golden_data_fname,
    run_llama3,
)
from llama3.llama3 import Llama3


@pytest.mark.parametrize(
    "model,encoding",
    [
        ("llama3_1", "q4_k"),
    ],
)
def test_llama(model, encoding, testdata_directory):
    golden_data_path = find_runtime_path(
        golden_data_fname(model, encoding), testdata_directory
    )
    expected_results = NumpyDecoder().decode(golden_data_path.read_text())

    # Download weights if not using tiny llama, which is in tree
    weight_path = (
        testdata_directory / "tiny_llama.gguf" if model == "tinyllama" else None
    )
    version = "llama3_1" if model == "tinyllama" else model
    config = build_config(version, weight_path, encoding)
    config.force_naive_kv_cache = True

    actual = run_llama3(Llama3(config), prompts=PROMPTS[:1])

    # These tolerances are VERY high; we have observed relative diffs upwards of 19,000
    # and atol of ~0.65 between M1 and intel CPUs
    compare_values(actual, expected_results, rtol=20000, atol=1.0)
