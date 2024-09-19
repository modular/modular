# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Runs Llama3.1 with full weights on GPU and compares it to previously generated
golden values.
"""


from evaluate_llama import find_runtime_path
from llama3 import Llama3
from evaluate_llama import (
    NumpyDecoder,
    compare_values,
    run_llama3,
    build_config,
    golden_data_fname,
)
import pytest


@pytest.mark.parametrize(
    "model,encoding",
    [
        ("llama3_1", "bfloat16"),
    ],
)
def test_llama(model, encoding, testdata_directory):
    fname = find_runtime_path(golden_data_fname(model, encoding))
    with open(fname) as f:
        expected_results = NumpyDecoder().decode(f.read())

    # Download weights
    weight_path = None

    version = "llama3_1"
    config = build_config(version, weight_path, encoding)

    actual = run_llama3(Llama3(config))
    compare_values(actual, expected_results)
