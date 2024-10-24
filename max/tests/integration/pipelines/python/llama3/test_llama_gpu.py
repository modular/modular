# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Runs Llama3.1 with full weights on GPU and compares it to previously generated
golden values.
"""

import numpy as np
import pytest
import torch
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
    eps_bf16 = torch.finfo(torch.bfloat16).eps
    expected_results = NumpyDecoder().decode(golden_data_path.read_text())
    compare_values(
        actual,
        expected_results,
        compare_fn=lambda x, y, desc: np.linalg.norm(x - y)
        / (np.linalg.norm(y) + eps_bf16)
        < 2 * eps_bf16,
    )
