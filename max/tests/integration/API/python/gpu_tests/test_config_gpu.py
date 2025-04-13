# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from max.driver import DeviceSpec
from max.nn.kv_cache import KVCacheStrategy
from max.pipelines import (
    PIPELINE_REGISTRY,
    PipelineConfig,
    PipelineEngine,
    SupportedEncoding,
)
from test_common.pipeline_model import (
    DUMMY_ARCH,
    DUMMY_GPTQ_ARCH,
    REPLIT_ARCH,
    mock_estimate_memory_footprint,
    mock_huggingface_config,
    mock_huggingface_hub_repo_exists_with_retry,
    prepare_registry,
)


@prepare_registry
@mock_estimate_memory_footprint
def test_config__raises_with_unsupported_GPTQ_format():
    PIPELINE_REGISTRY.register(DUMMY_GPTQ_ARCH)
    # this should work
    config = PipelineConfig(
        model_path="hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4",
        quantization_encoding=SupportedEncoding.gptq,
        device_specs=[DeviceSpec.accelerator()],
    )

    # We expect this to fail.
    with pytest.raises(ValueError):
        unsupported_config = PipelineConfig(
            model_path="jakiAJK/DeepSeek-R1-Distill-Llama-8B_GPTQ-int4",
            quantization_encoding=SupportedEncoding.gptq,
            device_specs=[DeviceSpec.accelerator()],
        )


@prepare_registry
@mock_estimate_memory_footprint
def test_config__test_retrieve_factory_with_known_architecture():
    PIPELINE_REGISTRY.register(DUMMY_ARCH)

    config = PipelineConfig(
        model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
        max_batch_size=1,
        max_length=1,
    )

    _, _ = PIPELINE_REGISTRY.retrieve_factory(pipeline_config=config)


@prepare_registry
@mock_estimate_memory_footprint
def test_config__test_retrieve_factory_with_unsupported_model_path():
    PIPELINE_REGISTRY.register(DUMMY_ARCH)

    config = PipelineConfig(
        model_path="modularai/replit-code-1.5",
        trust_remote_code=True,
        max_batch_size=1,
        max_length=1,
    )
    # Fallback to the generalized pipeline
    assert config.engine == PipelineEngine.HUGGINGFACE


@prepare_registry
@mock_estimate_memory_footprint
def test_config__test_load_factory_with_known_architecture_and_hf_repo_id():
    PIPELINE_REGISTRY.register(DUMMY_ARCH)

    config = PipelineConfig(
        model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
        max_batch_size=1,
        max_length=1,
    )

    _, _ = PIPELINE_REGISTRY.retrieve_factory(pipeline_config=config)


@prepare_registry
@mock_estimate_memory_footprint
def test_config__test_incompatible_quantization_encoding():
    PIPELINE_REGISTRY.register(DUMMY_ARCH)

    with pytest.raises(ValueError):
        # This should raise, as q4_k != bf16.
        config = PipelineConfig(
            model_path="modularai/llama-3.1",
            quantization_encoding=SupportedEncoding.q4_k,
            weight_path=[Path("llama-3.1-8b-instruct-bf16.gguf")],
            max_batch_size=1,
            max_length=1,
        )

    # This should not raise, as bfloat16 == bf16.
    config = PipelineConfig(
        model_path="modularai/llama-3.1",
        quantization_encoding=SupportedEncoding.bfloat16,
        weight_path=[Path("llama-3.1-8b-instruct-bf16.gguf")],
        device_specs=[DeviceSpec.accelerator()],
        max_batch_size=1,
        max_length=1,
    )


@prepare_registry
@mock_estimate_memory_footprint
def test_config__update_cache_strategy():
    PIPELINE_REGISTRY.register(DUMMY_ARCH)

    pipeline_config = PipelineConfig(
        model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
        cache_strategy=KVCacheStrategy.NAIVE,
        max_batch_size=1,
        max_length=1,
    )

    # Naive is not shown as supported in architecture, as
    # such this should change to a support strategy automatically.
    # TODO(AITLIB-293): this assert is triggered without a HF call
    assert (
        pipeline_config.model_config.kv_cache_config.cache_strategy
        == KVCacheStrategy.CONTINUOUS
    )


@prepare_registry
@pytest.mark.skip("TODO: AITLIB-238")
@mock_estimate_memory_footprint
def test_config__update_weight_paths():
    PIPELINE_REGISTRY.register(DUMMY_ARCH)
    PIPELINE_REGISTRY.register(REPLIT_ARCH)

    temp_valid_kernels = [
        ("bf16", 1, 16),
        ("bf16", 24, 128),
        ("bf16", 3, 64),  # SmolLM
        ("bf16", 32, 128),
        ("bf16", 4, 4),
        ("bf16", 8, 128),
        ("bf16", 8, 32),
        ("bf16", 8, 512),
        ("bf16", 8, 64),
        ("bf16", 8, 80),
        ("f32", 1, 16),
        ("f32", 2, 2),
        ("f32", 24, 128),
        ("f32", 3, 64),  # SmolLM
        ("f32", 32, 128),
        ("f32", 4, 4),
        ("f32", 8, 128),
        ("f32", 8, 32),
        ("f32", 8, 512),
        ("f32", 8, 64),
        ("f32", 8, 80),
    ]
    with patch(
        "max.pipelines.kv_cache.cache_params",
        temp_valid_kernels,
    ):
        # This first example, is requesting float32 from a gguf repository.
        config = PipelineConfig(
            model_path="modularai/llama-3.1",
            quantization_encoding=SupportedEncoding.float32,
            max_batch_size=1,
            max_length=512,
        )

        assert len(config.weight_path) == 1
        assert config.weight_path == [Path("llama-3.1-8b-instruct-f32.gguf")]

        # This second example, is requesting float32 from a safetensors repository.
        config = PipelineConfig(
            model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
            quantization_encoding=SupportedEncoding.float32,
            max_batch_size=1,
            max_length=512,
        )

        assert len(config.weight_path) == 1
        assert config.weight_path == [Path("model.safetensors")]

        # This should raise, as this repository, does not have q6_k weights.
        with pytest.raises(
            ValueError, match="compatible weights cannot be found"
        ):
            # This example, should raise, as you are requesting q6_k from a fp32
            # safetensors repo.
            config = PipelineConfig(
                model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
                quantization_encoding=SupportedEncoding.q6_k,
                device_specs=[DeviceSpec.cpu()],
            )

        # This example, should pass, since using fp32 weights for bfloat16 is
        # listed as an alternate encoding for fp32.
        config = PipelineConfig(
            model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
            quantization_encoding=SupportedEncoding.bfloat16,
            device_specs=[DeviceSpec.accelerator()],
            max_batch_size=1,
            max_length=512,
        )

        assert len(config.weight_path) == 1
        assert config.weight_path == [Path("model.safetensors")]

        with pytest.raises(ValueError):
            # This example, should raise as we dont have q4_k listed as supported.
            config = PipelineConfig(
                model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
                quantization_encoding=SupportedEncoding.q4_k,
                engine=PipelineEngine.MAX,
                max_batch_size=1,
                max_length=512,
            )

        # This example, should raise as we dont have q4_k listed as supported.
        # If we don't pass MAX though, we should not fail and fall back to HuggingFace.
        config = PipelineConfig(
            model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
            quantization_encoding=SupportedEncoding.q4_k,
            max_batch_size=1,
            max_length=512,
        )
        assert config.engine == PipelineEngine.HUGGINGFACE

        # This example, should not raise, as we are showing that we have a weight converter for pytorch for Replit.
        config = PipelineConfig(
            model_path="replit/replit-code-v1_5-3b",
            quantization_encoding=SupportedEncoding.bfloat16,
            device_specs=[DeviceSpec.accelerator()],
            trust_remote_code=True,
            max_batch_size=1,
            max_length=1,
        )
        assert config.engine == PipelineEngine.MAX
        assert config.weight_path == [Path("pytorch_model.bin")]

        # Test a partially complete huggingface_repo
        config = PipelineConfig(
            model_path="neubla/tiny-random-LlamaForCausalLM",
            max_batch_size=1,
            max_length=1,
        )
        assert config.quantization_encoding == SupportedEncoding.float32
        assert config.engine == PipelineEngine.MAX
        assert config.weight_path == [Path("model.safetensors")]

        # This example, should not raise as we are passing a valid weights path in a different repository.
        config = PipelineConfig(
            model_path="replit/replit-code-v1_5-3b",
            quantization_encoding=SupportedEncoding.float32,
            trust_remote_code=True,
            weight_path=[
                Path("modularai/replit-code-1.5/replit-code-v1_5-3b-f32.gguf")
            ],
            max_batch_size=1,
            max_length=1,
        )
        assert config.engine == PipelineEngine.MAX
        assert config.weight_path == [Path("replit-code-v1_5-3b-f32.gguf")]
        assert config._weights_repo_id == "modularai/replit-code-1.5"


@prepare_registry
@mock_estimate_memory_footprint
@mock_huggingface_config
@mock_huggingface_hub_repo_exists_with_retry
def test_config__validates_invalid_supported_device():
    with pytest.raises(
        ValueError, match="not compatible with the selected device type 'cpu'"
    ):
        # Invalid device/encoding combinations.
        config = PipelineConfig(
            model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
            device_specs=[DeviceSpec.cpu()],
            quantization_encoding=SupportedEncoding.bfloat16,
            max_length=1,
        )

    with pytest.raises(
        ValueError, match="not compatible with the selected device type 'gpu'"
    ):
        config = PipelineConfig(
            model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
            device_specs=[DeviceSpec.accelerator()],
            quantization_encoding=SupportedEncoding.q6_k,
            max_length=1,
        )
