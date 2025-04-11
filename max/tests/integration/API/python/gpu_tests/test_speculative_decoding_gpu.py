# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from pathlib import Path

import pytest
from max.driver import DeviceSpec
from max.pipelines import (
    PIPELINE_REGISTRY,
    PipelineConfig,
    PipelineEngine,
    SupportedEncoding,
)
from test_common.pipeline_model import (
    DUMMY_ARCH,
    mock_estimate_memory_footprint,
    prepare_registry,
)


@prepare_registry
@mock_estimate_memory_footprint
def test_config__validate_speculative_decoding_pipeline():
    PIPELINE_REGISTRY.register(DUMMY_ARCH)

    # Valid device/encoding combinations
    config = PipelineConfig(
        model_path="HuggingFaceTB/SmolLM-135M",
        device_specs=[DeviceSpec.cpu()],
        quantization_encoding=SupportedEncoding.float32,
        draft_model_path="HuggingFaceTB/SmolLM-135M",
    )

    with pytest.raises(ValueError):
        # Invalid device/encoding combinations
        config = PipelineConfig(
            model_path="modularai/llama-3.1",
            device_specs=[DeviceSpec.cpu()],
            quantization_encoding=SupportedEncoding.float32,
            draft_model_path="HuggingFaceTB/SmolLM-135M",
            engine=PipelineEngine.HUGGINGFACE,
        )

    with pytest.raises(ValueError):
        # Test that when the target & draft architectures are different
        # we raise an error.
        config = PipelineConfig(
            model_path="LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
            quantization_encoding=SupportedEncoding.q4_k,
            device_specs=[DeviceSpec.cpu()],
            draft_model_path="HuggingFaceTB/SmolLM-135M",
        )

    with pytest.raises(ValueError):
        # Test that the target & draft architectures are the same,
        # but the tokenizers are different
        config = PipelineConfig(
            model_path="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            quantization_encoding=SupportedEncoding.q4_k,
            weight_path=[
                Path(
                    "lmstudio-community/DeepSeek-R1-Distill-Llama-8B-GGUF/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"
                )
            ],
            draft_model_path="HuggingFaceTB/SmolLM-135M",
        )
