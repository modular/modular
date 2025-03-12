# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import logging
from functools import wraps
from pathlib import Path
from typing import Any, Sequence, cast
from unittest.mock import PropertyMock, patch

import pytest
from max.driver import Device, DeviceSpec, Tensor, load_devices
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import Graph, TensorType
from max.graph.weights import WeightsFormat
from max.pipelines import (
    PIPELINE_REGISTRY,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineEngine,
    PipelineModel,
    PipelineTask,
    SupportedArchitecture,
    SupportedEncoding,
    TextTokenizer,
    upper_bounded_default,
)
from max.pipelines.context import InputContext
from max.pipelines.kv_cache import (
    KVCacheInputs,
    KVCacheManager,
    KVCacheParams,
    KVCacheStrategy,
    estimate_kv_cache_size,
    load_kv_manager,
)
from max.pipelines.pipeline import KVCacheMixin
from transformers import AutoConfig


def prepare_registry(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        PIPELINE_REGISTRY.reset()
        result = func(*args, **kwargs)

        return result

    return wrapper


class DummyModelInputs(ModelInputs):
    input1: Tensor | None = None
    input2: Tensor | None = None
    input3: Tensor | None = None
    input4: Tensor | None = None

    def __init__(
        self,
        input1: Tensor | None = None,
        input2: Tensor | None = None,
        input3: Tensor | None = None,
        input4: Tensor | None = None,
        kv_cache_inputs: KVCacheInputs | None = None,
    ):
        self.input1 = input1
        self.input2 = input2
        self.input3 = input3
        self.input4 = input4
        self.kv_cache_inputs = kv_cache_inputs


class DummyPipelineModel(PipelineModel, KVCacheMixin):
    """A pipeline model with setup, input preparation and execution methods."""

    def execute(
        self,
        model_inputs: ModelInputs,
    ) -> ModelOutputs:
        """Runs the graph."""
        model_inputs = cast(DummyModelInputs, model_inputs)
        return ModelOutputs(next_token_logits=model_inputs.input1)

    @classmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        raise NotImplementedError("calculate_max_seq_len is not implemented")

    def prepare_initial_token_inputs(
        self,
        context_batch: Sequence[InputContext],
        kv_cache_inputs: KVCacheInputs | None = None,
    ) -> DummyModelInputs:
        """Prepares the initial inputs to be passed to `.execute()`.

        The inputs and functionality of this method can vary per model.
        For example, the model inputs could include:
        - Encoded tensors
        - A unique IDs for each tensor if this model uses a KV Cache manager.

        This function would batch the encoded tensors, claim a slot in the kv
        cache if the ID hasn't been seen before, and return the inputs and
        caches as a list of tensors."""
        return DummyModelInputs(
            input1=Tensor.zeros((0, 0), DType.float32),
            input2=Tensor.zeros((0, 0), DType.float32),
            input3=Tensor.zeros((0, 0), DType.float32),
            input4=Tensor.zeros((0, 0), DType.float32),
            kv_cache_inputs=None,
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: ModelInputs,
    ) -> DummyModelInputs:
        """Prepares the secondary inputs to be passed to `.execute()`.

        While `prepare_initial_token_inputs` is responsible for managing the initial inputs.
        This function is responsible for updating the inputs, for each step in a multi-step execution pattern.
        """
        return DummyModelInputs(
            input1=Tensor.zeros((0, 0), DType.float32),
            kv_cache_inputs=prev_model_inputs.kv_cache_inputs,
        )

    @classmethod
    def _get_num_kv_heads(cls, hf_config: Any) -> int:
        if hasattr(hf_config, "num_key_value_heads"):
            return hf_config.num_key_value_heads
        elif hasattr(hf_config, "num_attention_heads"):
            return hf_config.num_attention_heads
        elif hasattr(hf_config, "n_heads"):
            return hf_config.n_heads
        else:
            raise ValueError(
                "num_key_value_heads or num_attention_heads or n_heads not found in huggingface_config"
            )

    @classmethod
    def _get_hidden_size(cls, hf_config: Any) -> int:
        if hasattr(hf_config, "hidden_size"):
            return hf_config.hidden_size
        elif hasattr(hf_config, "d_model"):
            return hf_config.d_model
        else:
            raise ValueError(
                "hidden_size or d_model not found in huggingface_config"
            )

    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        if hasattr(huggingface_config, "num_hidden_layers"):
            return huggingface_config.num_hidden_layers
        elif hasattr(huggingface_config, "num_layers"):
            return huggingface_config.num_layers
        elif hasattr(huggingface_config, "n_layers"):
            return huggingface_config.n_layers
        else:
            raise ValueError(
                "num_hidden_layers or num_layers or n_layers not found in huggingface_config"
            )

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        num_kv_heads = cls._get_num_kv_heads(huggingface_config)
        hidden_size = cls._get_hidden_size(huggingface_config)

        return KVCacheParams(
            dtype=cache_dtype,
            n_kv_heads=num_kv_heads,
            head_dim=hidden_size // num_kv_heads,
            cache_strategy=kv_cache_config.cache_strategy,
            enable_prefix_caching=kv_cache_config.enable_prefix_caching,
            page_size=kv_cache_config.kv_cache_page_size,
            n_devices=n_devices,
        )

    def load_kv_manager(
        self,
        session: InferenceSession,
        available_cache_memory: int | None,
    ) -> KVCacheManager:
        """Provided a PipelineConfig and InferenceSession, load the kv manager."""
        assert available_cache_memory is not None
        num_layers = self.get_num_layers(self.pipeline_config)
        devices = load_devices(self.pipeline_config.model_config.device_specs)

        return load_kv_manager(
            params=self.get_kv_params(
                huggingface_config=self.huggingface_config,
                n_devices=len(self.devices),
                kv_cache_config=self.pipeline_config.kv_cache_config,
                cache_dtype=self.encoding.cache_dtype,
            ),
            max_batch_size=self.pipeline_config.max_batch_size,
            max_seq_len=self.calculate_max_seq_len(
                self.pipeline_config, self.huggingface_config
            ),
            num_layers=num_layers,
            devices=devices,
            available_cache_memory=available_cache_memory,
            session=session,
        )

    @classmethod
    def estimate_kv_cache_size(
        cls,
        pipeline_config: PipelineConfig,
        available_cache_memory: int | None,
        devices: list[Device],
        huggingface_config: AutoConfig,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> int:
        """Estimates the size of the kv cache in bytes."""
        assert available_cache_memory is not None
        assert pipeline_config.max_length is not None
        num_layers = cls.get_num_layers(huggingface_config=huggingface_config)

        return estimate_kv_cache_size(
            params=cls.get_kv_params(
                huggingface_config=huggingface_config,
                n_devices=len(devices),
                kv_cache_config=kv_cache_config,
                cache_dtype=cache_dtype,
            ),
            max_batch_size=pipeline_config.max_batch_size,
            max_seq_len=pipeline_config.max_length,
            num_layers=num_layers,
            available_cache_memory=available_cache_memory,
            devices=devices,
        )

    def load_model(
        self,
        session: InferenceSession,
    ) -> Model:
        """Provided a PipelineConfig and InferenceSession, build and load the model graph."""
        kv_inputs = self.kv_manager.input_symbols()[0]
        with Graph(
            "dummy",
            input_types=[
                TensorType(DType.int64, shape=["batch_size"]),
                *kv_inputs,
            ],
        ) as graph:
            tokens, kv_inputs_value = graph.inputs
            graph.output(tokens)
            return session.load(graph)


class DummyLlamaPipelineModel(DummyPipelineModel):
    @classmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        try:
            return upper_bounded_default(
                upper_bound=huggingface_config.max_position_embeddings,
                default=pipeline_config.max_length,
            )
        except ValueError as e:
            msg = (
                "Unable to infer max_length for DummyModel, the provided "
                f"max_length ({pipeline_config.max_length}) exceeds the "
                f"model's max_position_embeddings "
                f"({huggingface_config.max_position_embeddings})."
            )
            raise ValueError(msg) from e


class DummyReplitPipelineModel(DummyPipelineModel):
    @classmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        try:
            return upper_bounded_default(
                upper_bound=huggingface_config.max_seq_len,
                default=pipeline_config.max_length,
            )
        except ValueError as e:
            msg = (
                "Unable to infer max_length for DummyModel, the provided "
                f"max_length ({pipeline_config.max_length}) exceeds the "
                f"model's max_seq_len "
                f"({huggingface_config.max_seq_len})."
            )
            raise ValueError(msg) from e


DUMMY_ARCH = SupportedArchitecture(
    name="LlamaForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["modularai/llama-3.1"],
    default_encoding=SupportedEncoding.float32,
    supported_encodings={
        SupportedEncoding.float32: [KVCacheStrategy.CONTINUOUS],
        SupportedEncoding.bfloat16: [KVCacheStrategy.CONTINUOUS],
        SupportedEncoding.q6_k: [KVCacheStrategy.CONTINUOUS],
    },
    pipeline_model=DummyLlamaPipelineModel,
    tokenizer=TextTokenizer,
    default_weights_format=WeightsFormat.gguf,
)

REPLIT_ARCH = SupportedArchitecture(
    name="MPTForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["modularai/replit-code-1.5"],
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={
        SupportedEncoding.float32: [KVCacheStrategy.CONTINUOUS],
        SupportedEncoding.bfloat16: [KVCacheStrategy.CONTINUOUS],
    },
    pipeline_model=DummyReplitPipelineModel,
    tokenizer=TextTokenizer,
    default_weights_format=WeightsFormat.gguf,
)


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

    config = PipelineConfig(
        model_path="GSAI-ML/LLaDA-8B-Instruct",
        # This forces it to fail if we dont have it.
        engine=PipelineEngine.MAX,
        max_batch_size=1,
        max_length=1,
        trust_remote_code=True,
    )

    with pytest.raises(ValueError):
        config = PIPELINE_REGISTRY.validate_pipeline_config(config)


@prepare_registry
def test_registry__test_retrieve_with_unknown_architecture_unknown_engine():
    PIPELINE_REGISTRY.register(DUMMY_ARCH)

    config = PipelineConfig(
        model_path="GSAI-ML/LLaDA-8B-Instruct",
        max_batch_size=1,
        max_length=1,
        trust_remote_code=True,
    )

    config = PIPELINE_REGISTRY.validate_pipeline_config(config)
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

    config = PIPELINE_REGISTRY.validate_pipeline_config(
        pipeline_config=config,
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

    # This should raise, as q4_k != bf16.
    config = PipelineConfig(
        model_path="modularai/llama-3.1",
        quantization_encoding=SupportedEncoding.q4_k,
        weight_path=[Path("llama-3.1-8b-instruct-bf16.gguf")],
        max_batch_size=1,
        max_length=1,
    )

    with pytest.raises(ValueError):
        PIPELINE_REGISTRY.validate_pipeline_config(config)

    # This should not raise, as bfloat16 == bf16.
    config = PipelineConfig(
        model_path="modularai/llama-3.1",
        quantization_encoding=SupportedEncoding.bfloat16,
        weight_path=[Path("llama-3.1-8b-instruct-bf16.gguf")],
        device_specs=[DeviceSpec.accelerator()],
        max_batch_size=1,
        max_length=1,
    )

    PIPELINE_REGISTRY.validate_pipeline_config(config)


@prepare_registry
def test_registry__update_cache_strategy():
    PIPELINE_REGISTRY.register(DUMMY_ARCH)

    config = PipelineConfig(
        model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
        cache_strategy=KVCacheStrategy.NAIVE,
        max_batch_size=1,
        max_length=1,
    )

    # Naive is not shown as supported in architecture, as
    # such this should change to a support strategy automatically.
    pipeline_config = PIPELINE_REGISTRY.validate_pipeline_config(config)
    assert (
        pipeline_config.kv_cache_config.cache_strategy
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

        config = PIPELINE_REGISTRY.validate_pipeline_config(config)

        assert len(config.weight_path) == 1
        assert config.weight_path == [Path("llama-3.1-8b-instruct-f32.gguf")]

        # This second example, is requesting float32 from a safetensors repository.
        config = PipelineConfig(
            model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
            quantization_encoding=SupportedEncoding.float32,
            max_batch_size=1,
            max_length=512,
        )

        config = PIPELINE_REGISTRY.validate_pipeline_config(config)

        assert len(config.weight_path) == 1
        assert config.weight_path == [Path("model.safetensors")]

        # This example, should raise, as you are requesting q6_k from a fp32
        # safetensors repo.
        config = PipelineConfig(
            model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
            quantization_encoding=SupportedEncoding.q6_k,
            device_specs=[DeviceSpec.cpu()],
        )

        # This should raise, as this repository, does not have q6_k weights.
        with pytest.raises(
            ValueError, match="compatible weights cannot be found"
        ):
            PIPELINE_REGISTRY.validate_pipeline_config(config)

        # This example, should pass, since using fp32 weights for bfloat16 is
        # listed as an alternate encoding for fp32.
        config = PipelineConfig(
            model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
            quantization_encoding=SupportedEncoding.bfloat16,
            device_specs=[DeviceSpec.accelerator()],
            max_batch_size=1,
            max_length=512,
        )

        # This should pass, since the float16 weights will be used for bfloat16.
        PIPELINE_REGISTRY.validate_pipeline_config(config)
        assert len(config.weight_path) == 1
        assert config.weight_path == [Path("model.safetensors")]

        # This example, should raise as we dont have q4_k listed as supported.
        config = PipelineConfig(
            model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
            quantization_encoding=SupportedEncoding.q4_k,
            engine=PipelineEngine.MAX,
            max_batch_size=1,
            max_length=512,
        )

        with pytest.raises(ValueError):
            config = PIPELINE_REGISTRY.validate_pipeline_config(config)

        # This example, should raise as we dont have q4_k listed as supported.
        # If we don't pass MAX though, we should not fail and fall back to HuggingFace.
        config = PipelineConfig(
            model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
            quantization_encoding=SupportedEncoding.q4_k,
            max_batch_size=1,
            max_length=512,
        )

        config = PIPELINE_REGISTRY.validate_pipeline_config(config)
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

        config = PIPELINE_REGISTRY.validate_pipeline_config(config)
        assert config.engine == PipelineEngine.MAX
        assert config.weight_path == [Path("pytorch_model.bin")]

        # Test a partially complete huggingface_repo
        config = PipelineConfig(
            model_path="neubla/tiny-random-LlamaForCausalLM",
            max_batch_size=1,
            max_length=1,
        )
        config = PIPELINE_REGISTRY.validate_pipeline_config(config)
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

        config = PIPELINE_REGISTRY.validate_pipeline_config(config)
        assert config.engine == PipelineEngine.MAX
        assert config.weight_path == [Path("replit-code-v1_5-3b-f32.gguf")]
        assert config._weights_repo_id == "modularai/replit-code-1.5"


@prepare_registry
def test_registry__raise_oom_error_weights_size_exceeds_available_memory():
    PIPELINE_REGISTRY.register(DUMMY_ARCH)

    config = PipelineConfig(
        model_path="modularai/llama-3.1",
    )

    config.max_batch_size = None
    config.max_length = None
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
            RuntimeError, match="Weights size exceeds available memory"
        ):
            PIPELINE_REGISTRY.validate_pipeline_config(config)


@prepare_registry
@pytest.mark.skip("TODO: AITLIB-238")
def test_registry__raise_oom_error_all_defaults_no_valid_solution():
    PIPELINE_REGISTRY.register(DUMMY_ARCH)

    config = PipelineConfig(
        model_path="modularai/llama-3.1",
    )

    config.max_batch_size = None
    config.max_length = None
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
            PIPELINE_REGISTRY.validate_pipeline_config(config)


@prepare_registry
def test_registry__raise_oom_error_all_defaults(caplog):
    PIPELINE_REGISTRY.register(DUMMY_ARCH)

    config = PipelineConfig(
        model_path="modularai/llama-3.1",
    )

    config.max_batch_size = None
    config.max_length = None
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
            PIPELINE_REGISTRY.validate_pipeline_config(config)

        assert "Truncated model's default max_length from" in caplog.text


@prepare_registry
def test_registry__raise_oom_error_max_length_set():
    PIPELINE_REGISTRY.register(DUMMY_ARCH)

    config = PipelineConfig(
        model_path="modularai/llama-3.1",
    )

    config.max_batch_size = None
    config.max_length = 100000
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
            PIPELINE_REGISTRY.validate_pipeline_config(config)


@prepare_registry
def test_registry__raise_oom_error_max_batch_size_set():
    PIPELINE_REGISTRY.register(DUMMY_ARCH)

    config = PipelineConfig(
        model_path="modularai/llama-3.1",
    )

    config.max_batch_size = 100000
    config.max_length = None
    with (
        patch.object(
            DummyLlamaPipelineModel, "calculate_max_seq_len", return_value=4096
        ),
        patch(
            "max.driver.Device.stats", new_callable=PropertyMock
        ) as device_mock,
    ):
        device_mock.return_value = {"free_memory": 40000 * 1024 * 1024}
        with pytest.raises(
            RuntimeError, match="Try reducing --max-batch-size to"
        ):
            PIPELINE_REGISTRY.validate_pipeline_config(config)


@prepare_registry
def test_registry__raise_oom_error_max_batch_size_set_and_max_length_set():
    PIPELINE_REGISTRY.register(DUMMY_ARCH)

    config = PipelineConfig(
        model_path="modularai/llama-3.1",
    )

    config.max_batch_size = 100000
    config.max_length = 4096
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
            RuntimeError, match="Try reducing --max-batch-size to"
        ):
            PIPELINE_REGISTRY.validate_pipeline_config(config)


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

    PIPELINE_REGISTRY.validate_pipeline_config(config)

    # Invalid device/encoding combinations
    config = PipelineConfig(
        model_path="modularai/llama-3.1",
        device_specs=[DeviceSpec.cpu()],
        quantization_encoding=SupportedEncoding.float32,
        draft_model="HuggingFaceTB/SmolLM-135M",
        engine=PipelineEngine.HUGGINGFACE,
    )

    with pytest.raises(ValueError):
        PIPELINE_REGISTRY.validate_pipeline_config(config)

    # Test that when the target & draft architectures are different
    # we raise an error.
    config = PipelineConfig(
        model_path="LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
        quantization_encoding=SupportedEncoding.q4_k,
        device_specs=[DeviceSpec.cpu()],
        draft_model="HuggingFaceTB/SmolLM-135M",
    )

    with pytest.raises(ValueError):
        PIPELINE_REGISTRY.validate_pipeline_config(config)

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

    with pytest.raises(ValueError):
        PIPELINE_REGISTRY.validate_pipeline_config(config)


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
    PIPELINE_REGISTRY.validate_pipeline_config(config)

    config = PipelineConfig(
        model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
        device_specs=[DeviceSpec.accelerator()],
        quantization_encoding=SupportedEncoding.bfloat16,
        max_length=1,
    )
    PIPELINE_REGISTRY.validate_pipeline_config(config)

    # Invalid device/encoding combinations.
    config = PipelineConfig(
        model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
        device_specs=[DeviceSpec.cpu()],
        quantization_encoding=SupportedEncoding.bfloat16,
        max_length=1,
    )
    with pytest.raises(ValueError, match="not supported on cpu"):
        PIPELINE_REGISTRY.validate_pipeline_config(config)

    config = PipelineConfig(
        model_path="trl-internal-testing/tiny-random-LlamaForCausalLM",
        device_specs=[DeviceSpec.accelerator()],
        quantization_encoding=SupportedEncoding.q6_k,
        max_length=1,
    )
    with pytest.raises(ValueError, match="not supported on gpu"):
        PIPELINE_REGISTRY.validate_pipeline_config(config)
