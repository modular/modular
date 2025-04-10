# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import pytest
from max.pipelines import (
    PIPELINE_REGISTRY,
    PipelineConfig,
    PipelineEngine,
)
from test_common.pipeline_model import (
    DUMMY_ARCH,
    mock_pipeline_config_hf_dependencies,
    prepare_registry,
)


@prepare_registry
@mock_pipeline_config_hf_dependencies
def test_registry__test_register():
    PIPELINE_REGISTRY.register(DUMMY_ARCH)
    assert "LlamaForCausalLM" in PIPELINE_REGISTRY.architectures

    # This should fail when registering the architecture for a second time.
    with pytest.raises(ValueError):
        PIPELINE_REGISTRY.register(DUMMY_ARCH)


@prepare_registry
@mock_pipeline_config_hf_dependencies
def test_registry__test_retrieve_with_unknown_architecture_max_engine():
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
def test_registry__test_retrieve_with_unknown_architecture_unknown_engine():
    PIPELINE_REGISTRY.register(DUMMY_ARCH)

    config = PipelineConfig(
        model_path="GSAI-ML/LLaDA-8B-Instruct",
        max_batch_size=1,
        max_length=1,
        trust_remote_code=True,
    )
    assert config.engine == PipelineEngine.HUGGINGFACE
