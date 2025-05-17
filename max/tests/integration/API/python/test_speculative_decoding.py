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
from test_common.mocks import mock_estimate_memory_footprint
from test_common.pipeline_model_dummy import DUMMY_ARCH
from test_common.registry import prepare_registry


@prepare_registry
@mock_estimate_memory_footprint
@pytest.mark.skip(
    reason="TODO(AITLIB-339): This test is flaky due to bad huggingface cache hydration"
)
def test_config__validate_device_and_encoding_combinations(
    smollm_135m_local_path,
    llama_3_1_8b_instruct_local_path,
):
    PIPELINE_REGISTRY.register(DUMMY_ARCH)

    # Valid device/encoding combinations
    config = PipelineConfig(
        model_path=smollm_135m_local_path,
        quantization_encoding=SupportedEncoding.float32,
        device_specs=[DeviceSpec.cpu()],
        draft_model_path=smollm_135m_local_path,
    )

    with pytest.raises(ValueError):
        # Invalid device/encoding combinations
        config = PipelineConfig(
            model_path=llama_3_1_8b_instruct_local_path,
            quantization_encoding=SupportedEncoding.float32,
            device_specs=[DeviceSpec.cpu()],
            draft_model_path=smollm_135m_local_path,
            engine=PipelineEngine.HUGGINGFACE,
        )


@pytest.mark.skip(reason="TODO(AITLIB-363): Division by zero error.")
def test_config__validate_target_and_draft_architecture(
    exaone_2_4b_local_path,
    smollm_135m_local_path,
    deepseek_r1_distill_llama_8b_local_path,
):
    with pytest.raises(ValueError):
        # Test that when the target & draft architectures are different
        # we raise an error.
        config = PipelineConfig(
            model_path=exaone_2_4b_local_path,
            quantization_encoding=SupportedEncoding.q4_k,
            device_specs=[DeviceSpec.cpu()],
            draft_model_path=smollm_135m_local_path,
        )

    with pytest.raises(ValueError):
        # Test that the target & draft architectures are the same,
        # but the tokenizers are different
        config = PipelineConfig(
            model_path=deepseek_r1_distill_llama_8b_local_path,
            quantization_encoding=SupportedEncoding.q4_k,
            weight_path=[
                Path(
                    "lmstudio-community/DeepSeek-R1-Distill-Llama-8B-GGUF/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"
                )
            ],
            draft_model_path=smollm_135m_local_path,
        )


def test_config__validate_huggingface_engine(llama_3_1_8b_instruct_local_path):
    """Test that speculative decoding is not supported with HuggingFace engine."""
    with pytest.raises(
        ValueError,
        match="Speculative Decoding not supported with the HuggingFace Engine",
    ):
        PipelineConfig(
            model_path=llama_3_1_8b_instruct_local_path,
            quantization_encoding=SupportedEncoding.bfloat16,
            device_specs=[DeviceSpec.accelerator()],
            draft_model_path=llama_3_1_8b_instruct_local_path,
            engine=PipelineEngine.HUGGINGFACE,
        )
