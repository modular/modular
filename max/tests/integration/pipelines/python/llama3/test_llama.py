# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Runs Llama3.1 on a tiny checkpoint and compares it to previously generated
golden values.
"""

import pytest
from max.engine import InferenceSession
from evaluate_llama import SupportedTestModels
from llama3.llama3 import load_llama3_and_kv_manager
from max.pipelines import TextTokenizer
from test_common.evaluate import PROMPTS, compare_values, run_model
from test_common.numpy_encoder import NumpyDecoder
from test_common.path import find_runtime_path


@pytest.mark.parametrize(
    "model,encoding",
    [
        ("llama3_1", "q4_k"),
    ],
)
def test_llama(model, encoding, testdata_directory):
    test_model = SupportedTestModels.get(model, encoding)
    config = test_model.build_config()

    tokenizer = TextTokenizer(config)
    session = InferenceSession(devices=[config.device])
    model, _ = load_llama3_and_kv_manager(config, session)
    actual = run_model(
        model,
        tokenizer,
        prompts=PROMPTS[:1],
    )

    golden_data_path = find_runtime_path(
        test_model.golden_data_fname(), testdata_directory
    )
    expected_results = NumpyDecoder().decode(golden_data_path.read_text())
    with pytest.raises(AssertionError):
        # TODO(MSDK-968): Q4_K is currently expected not to match golden values.
        # This test will fail once we have fixed the accuracy issue.
        compare_values(actual, expected_results)
