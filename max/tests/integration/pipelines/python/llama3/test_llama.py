# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Runs Llama3.1 on a tiny checkpoint and compares it to previously generated
golden values.
"""

import pytest
from evaluate_llama import SupportedTestModels
from max.pipelines import PipelineConfig
from test_common.evaluate import PROMPTS, compare_values, run_model
from test_common.numpy_encoder import NumpyDecoder
from test_common.path import find_runtime_path

pytest_plugins = "test_common.registry"


@pytest.mark.skip("loads llama model, which will download taking a while.")
def test_llama_eos_token_id(pipeline_registry):
    """This test is primarily written to be run in a bespoke fashion, as it is downloads llama-3.1, which can tax CI unnecessarily."""
    config = PipelineConfig(huggingface_repo_id="modularai/llama-3.1")
    _, pipeline = pipeline_registry.retrieve(config)

    # The llama3_1 huggingface config has three eos tokens I want to make sure these are grabbed appropriately.
    assert pipeline._eos_token_id == set([128001, 128008, 128009])
    assert len(pipeline._eos_token_id) == 3


@pytest.mark.parametrize(
    "model,encoding",
    [
        ("llama3_1", "q4_k"),
    ],
)
def test_llama(pipeline_registry, model, encoding, testdata_directory):
    test_model = SupportedTestModels.get(model, encoding)
    config = test_model.build_config()

    tokenizer, pipeline = pipeline_registry.retrieve(config)
    actual = run_model(
        pipeline._pipeline_model,
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
