# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import pytest
from max.interfaces import PipelineTask
from max.pipelines import (
    PIPELINE_REGISTRY,
    PipelineConfig,
    PipelineEngine,
)
from test_common.mocks import mock_pipeline_config_hf_dependencies
from test_common.pipeline_model_dummy import DUMMY_ARCH
from test_common.registry import prepare_registry


@prepare_registry
@mock_pipeline_config_hf_dependencies
def test_registry__test_register() -> None:
    PIPELINE_REGISTRY.register(DUMMY_ARCH)
    assert "LlamaForCausalLM" in PIPELINE_REGISTRY.architectures

    # This should fail when registering the architecture for a second time.
    with pytest.raises(ValueError):
        PIPELINE_REGISTRY.register(DUMMY_ARCH)


@prepare_registry
@mock_pipeline_config_hf_dependencies
def test_registry__test_retrieve_with_unknown_architecture_max_engine() -> None:
    PIPELINE_REGISTRY.register(DUMMY_ARCH)

    with pytest.raises(ValueError):
        config = PipelineConfig(
            model_path="GSAI-ML/LLaDA-8B-Instruct",
            # This forces it to fail if we dont have it.
            engine=PipelineEngine.MAX,
            max_batch_size=1,
            max_length=1,
            trust_remote_code=True,
        )


@prepare_registry
@mock_pipeline_config_hf_dependencies
def test_registry__test_retrieve_with_unknown_architecture_unknown_engine() -> (
    None
):
    PIPELINE_REGISTRY.register(DUMMY_ARCH)

    # Should now raise an error since HuggingFace fallback is removed
    with pytest.raises(
        ValueError, match="MAX-optimized architecture not available"
    ):
        config = PipelineConfig(
            model_path="GSAI-ML/LLaDA-8B-Instruct",
            max_batch_size=1,
            max_length=1,
            trust_remote_code=True,
        )

    @prepare_registry
    @mock_pipeline_config_hf_dependencies
    def test_registry__retrieve_pipeline_task_returns_text_generation() -> None:
        PIPELINE_REGISTRY.register(DUMMY_ARCH)
        config = PipelineConfig(
            model_path="some-model",
            max_batch_size=1,
            max_length=1,
            trust_remote_code=True,
        )
        task = PIPELINE_REGISTRY.retrieve_pipeline_task(config)
        assert task == PipelineTask.TEXT_GENERATION
