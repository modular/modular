# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import logging
from functools import wraps
from pathlib import Path
from unittest.mock import PropertyMock, patch

import pytest
from max.driver import DeviceSpec
from max.pipelines import (
    PIPELINE_REGISTRY,
    PipelineConfig,
    PipelineEngine,
    SupportedEncoding,
)
from max.pipelines.kv_cache import (
    KVCacheStrategy,
)
from test_common.pipeline_model import (
    DUMMY_ARCH,
    REPLIT_ARCH,
    DummyLlamaPipelineModel,
)


def prepare_registry(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        PIPELINE_REGISTRY.reset()
        result = func(*args, **kwargs)

        return result

    return wrapper


@prepare_registry
def test_registry__test_register():
    PIPELINE_REGISTRY.register(DUMMY_ARCH)
    assert "LlamaForCausalLM" in PIPELINE_REGISTRY.architectures

    # This should fail when registering the architecture for a second time.
    with pytest.raises(ValueError):
        PIPELINE_REGISTRY.register(DUMMY_ARCH)


@prepare_registry
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
def test_registry__test_retrieve_with_unknown_architecture_unknown_engine():
    PIPELINE_REGISTRY.register(DUMMY_ARCH)

    config = PipelineConfig(
        model_path="GSAI-ML/LLaDA-8B-Instruct",
        max_batch_size=1,
        max_length=1,
        trust_remote_code=True,
    )
    assert config.engine == PipelineEngine.HUGGINGFACE


@prepare_registry
def test_registry__test_retrieve_factory_with_known_architecture():
    PIPELINE_REGISTRY.register(DUMMY_ARCH)

    config = PipelineConfig(
        model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
        max_batch_size=1,
        max_length=1,
    )

    _, _ = PIPELINE_REGISTRY.retrieve_factory(pipeline_config=config)


@prepare_registry
def test_registry__test_retrieve_factory_with_unsupported_model_path():
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
def test_registry__test_load_factory_with_known_architecture_and_hf_repo_id():
    PIPELINE_REGISTRY.register(DUMMY_ARCH)

    config = PipelineConfig(
        model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
        max_batch_size=1,
        max_length=1,
    )

    _, _ = PIPELINE_REGISTRY.retrieve_factory(pipeline_config=config)


@prepare_registry
def test_registry__test_incompatible_quantization_encoding():
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
def test_registry__update_cache_strategy():
    PIPELINE_REGISTRY.register(DUMMY_ARCH)

    pipeline_config = PipelineConfig(
        model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
        cache_strategy=KVCacheStrategy.NAIVE,
        max_batch_size=1,
        max_length=1,
    )

    # Naive is not shown as supported in architecture, as
    # such this should change to a support strategy automatically.
    assert (
        pipeline_config.model_config.kv_cache_config.cache_strategy
        == KVCacheStrategy.CONTINUOUS
    )


@prepare_registry
@pytest.mark.skip("TODO: AITLIB-238")
def test_registry__update_weight_paths():
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
def test_registry__raise_oom_error_weights_size_exceeds_available_memory():
    PIPELINE_REGISTRY.register(DUMMY_ARCH)

    with (
        patch.object(
            DummyLlamaPipelineModel,
            "calculate_max_seq_len",
            return_value=100000,
        ),
        patch(
            "max.driver.Device.stats", new_callable=PropertyMock
        ) as device_mock,
    ):
        device_mock.return_value = {"free_memory": 5 * 1024 * 1024}
        with pytest.raises(
            RuntimeError, match="Model size exceeds available memory"
        ):
            config = PipelineConfig(
                model_path="modularai/llama-3.1",
                max_batch_size=None,
                max_length=None,
            )


@prepare_registry
@pytest.mark.skip("TODO: AITLIB-238")
def test_registry__raise_oom_error_all_defaults_no_valid_solution():
    PIPELINE_REGISTRY.register(DUMMY_ARCH)

    with (
        patch.object(
            DummyLlamaPipelineModel, "calculate_max_seq_len", return_value=10000
        ),
        patch(
            "max.driver.Device.stats", new_callable=PropertyMock
        ) as device_mock,
    ):
        device_mock.return_value = {"free_memory": 30641 * 1024 * 1024}
        with pytest.raises(
            RuntimeError,
        ):
            config = PipelineConfig(
                model_path="modularai/llama-3.1",
                max_batch_size=None,
                max_length=None,
            )


@prepare_registry
def test_registry__raise_oom_error_all_defaults(caplog):
    PIPELINE_REGISTRY.register(DUMMY_ARCH)

    with (
        patch.object(
            DummyLlamaPipelineModel,
            "calculate_max_seq_len",
            return_value=100000,
        ),
        patch(
            "max.driver.Device.stats", new_callable=PropertyMock
        ) as device_mock,
    ):
        device_mock.return_value = {"free_memory": 40000 * 1024 * 1024}
        with caplog.at_level(logging.WARNING):
            config = PipelineConfig(
                model_path="modularai/llama-3.1",
                max_batch_size=None,
                max_length=None,
            )

        assert "Truncated model's default max_length from" in caplog.text


@prepare_registry
def test_registry__raise_oom_error_max_length_set():
    PIPELINE_REGISTRY.register(DUMMY_ARCH)

    with (
        patch.object(
            DummyLlamaPipelineModel,
            "calculate_max_seq_len",
            return_value=9999999999999,
        ),
        patch(
            "max.driver.Device.stats", new_callable=PropertyMock
        ) as device_mock,
    ):
        device_mock.return_value = {"free_memory": 40000 * 1024 * 1024}
        with pytest.raises(
            RuntimeError,
            match=r"Try reducing --max-length to \d+ .*supports batch size of",
        ):
            config = PipelineConfig(
                model_path="modularai/llama-3.1",
                max_batch_size=None,
                max_length=100000,
            )


@prepare_registry
def test_registry__raise_oom_error_max_batch_size_set():
    PIPELINE_REGISTRY.register(DUMMY_ARCH)

    with (
        patch.object(
            DummyLlamaPipelineModel, "calculate_max_seq_len", return_value=4096
        ),
        patch(
            "max.driver.Device.stats", new_callable=PropertyMock
        ) as device_mock,
    ):
        device_mock.return_value = {"free_memory": 40000 * 1024 * 1024}
        with pytest.raises(RuntimeError, match="reducing --max-batch-size to"):
            config = PipelineConfig(
                model_path="modularai/llama-3.1",
                max_batch_size=100000,
                max_length=None,
            )


@prepare_registry
def test_registry__raise_oom_error_max_batch_size_set_and_max_length_set():
    PIPELINE_REGISTRY.register(DUMMY_ARCH)

    with (
        patch.object(
            DummyLlamaPipelineModel,
            "calculate_max_seq_len",
            return_value=9999999999999,
        ),
        patch(
            "max.driver.Device.stats", new_callable=PropertyMock
        ) as device_mock,
    ):
        device_mock.return_value = {"free_memory": 40000 * 1024 * 1024}
        with pytest.raises(RuntimeError, match="reducing --max-batch-size to"):
            config = PipelineConfig(
                model_path="modularai/llama-3.1",
                max_batch_size=100000,
                max_length=4096,
            )


@prepare_registry
def test_registry__validate_speculative_decoding_pipeline():
    PIPELINE_REGISTRY.register(DUMMY_ARCH)

    # Valid device/encoding combinations
    config = PipelineConfig(
        model_path="HuggingFaceTB/SmolLM-135M",
        device_specs=[DeviceSpec.cpu()],
        quantization_encoding=SupportedEncoding.float32,
        draft_model="HuggingFaceTB/SmolLM-135M",
    )

    with pytest.raises(ValueError):
        # Invalid device/encoding combinations
        config = PipelineConfig(
            model_path="modularai/llama-3.1",
            device_specs=[DeviceSpec.cpu()],
            quantization_encoding=SupportedEncoding.float32,
            draft_model="HuggingFaceTB/SmolLM-135M",
            engine=PipelineEngine.HUGGINGFACE,
        )

    with pytest.raises(ValueError):
        # Test that when the target & draft architectures are different
        # we raise an error.
        config = PipelineConfig(
            model_path="LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
            quantization_encoding=SupportedEncoding.q4_k,
            device_specs=[DeviceSpec.cpu()],
            draft_model="HuggingFaceTB/SmolLM-135M",
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
            draft_model="HuggingFaceTB/SmolLM-135M",
        )


@prepare_registry
def test_registry__validates_supported_device():
    PIPELINE_REGISTRY.register(DUMMY_ARCH)

    # Valid device/encoding combinations.
    config = PipelineConfig(
        model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
        device_specs=[DeviceSpec.cpu()],
        quantization_encoding=SupportedEncoding.float32,
        max_length=1,
    )

    config = PipelineConfig(
        model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
        device_specs=[DeviceSpec.accelerator()],
        quantization_encoding=SupportedEncoding.bfloat16,
        max_length=1,
    )

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
