# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Runs Llama3.1 with full weights on GPU and compares it to previously generated
golden values.
"""

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
from llama3 import Llama3


@pytest.mark.parametrize(
    "model,encoding",
    [
        ("llama3_1", "bfloat16"),
    ],
)
def test_llama(model, encoding, testdata_directory):
    golden_data_path = find_runtime_path(
        golden_data_fname(model, encoding), testdata_directory
    )
    expected_results = NumpyDecoder().decode(golden_data_path.read_text())

    # Download weights
    weight_path = None

    version = "llama3_1"
    config = build_config(version, weight_path, encoding)

    actual = run_llama3(Llama3(config), prompts=PROMPTS[:1])
    compare_values(actual, expected_results)
