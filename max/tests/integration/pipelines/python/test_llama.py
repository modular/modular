# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Runs Llama3.1 on a tiny checkpoint and compares it to previously generated
golden values.
"""


import os
from evaluate_llama import find_runtime_path
from pathlib import Path
from evaluate_llama import (
    NumpyDecoder,
    compare_values,
    run_llama3,
    golden_data_fname,
    build_config,
)
from llama3 import Llama3
import pytest


@pytest.mark.parametrize(
    "model,encoding",
    [
        ("tinyllama", "float32"),
        # ("llama3_1", "q4_k"), TODO: MSDK-968 to re-enable
    ],
)
def test_llama(model, encoding, testdata_directory):
    fname = find_runtime_path(golden_data_fname(model, encoding))

    with open(fname) as f:
        expected_results = NumpyDecoder().decode(f.read())

    # Download weights if not using tiny llama, which is in tree
    weight_path = (
        testdata_directory / "tiny_llama.gguf" if model == "tinyllama" else None
    )
    version = "llama3_1" if model == "tinyllama" else model
    config = build_config(version, weight_path, encoding)

    actual = run_llama3(Llama3(config))
    compare_values(actual, expected_results)
