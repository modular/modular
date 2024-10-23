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
    SupportedTestModels,
    compare_values,
    find_runtime_path,
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
    test_model = SupportedTestModels.get(model, encoding)
    config = test_model.build_config()
    actual = run_llama3(Llama3(config), prompts=PROMPTS[:1])

    golden_data_path = find_runtime_path(
        test_model.golden_data_fname(), testdata_directory
    )
    expected_results = NumpyDecoder().decode(golden_data_path.read_text())
    compare_values(actual, expected_results, rtol=0.01, atol=1e-3)
