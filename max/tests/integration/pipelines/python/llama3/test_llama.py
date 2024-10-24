# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Runs Llama3.1 on a tiny checkpoint and compares it to previously generated
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
from llama3.llama3 import Llama3


@pytest.mark.parametrize(
    "model,encoding",
    [
        ("llama3_1", "q4_k"),
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
    with pytest.raises(AssertionError):
        # TODO(MSDK-968): Q4_K is currently expected not to match golden values.
        # This test will fail once we have fixed the accuracy issue.
        compare_values(actual, expected_results)
