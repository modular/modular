# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Any, Sequence, cast
from unittest.mock import patch

import pytest
from max.driver import Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import Graph, TensorType
from max.pipelines import (
    PIPELINE_REGISTRY,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineEngine,
    PipelineModel,
    SupportedArchitecture,
    SupportedEncoding,
    TextTokenizer,
    WeightsFormat,
    upper_bounded_default,
)
from max.pipelines.context import InputContext
from max.pipelines.kv_cache import (
    KVCacheManager,
    KVCacheParams,
    KVCacheStrategy,
    estimate_kv_cache_size,
    load_kv_manager,
)
from max.pipelines.kv_cache.cache_params import VALID_KV_KERNELS


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
    ):
        self.input1 = input1
        self.input2 = input2
        self.input3 = input3
        self.input4 = input4


class DummyPipelineModel(PipelineModel):
    """A pipeline model with setup, input preparation and execution methods."""

    def execute(
        self,
        model_inputs: ModelInputs,
        kv_cache_inputs: Sequence[Tensor] | None = None,
    ) -> ModelOutputs:
        """Runs the graph."""
        model_inputs = cast(DummyModelInputs, model_inputs)
        return ModelOutputs(next_token_logits=model_inputs.input1)

    @classmethod
    def calculate_max_seq_len(cls, pipeline_config: PipelineConfig) -> int:
        raise NotImplementedError("calculate_max_seq_len is not implemented")

    def prepare_initial_token_inputs(
        self, context_batch: Sequence[InputContext]
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
        return DummyModelInputs(input1=Tensor.zeros((0, 0), DType.float32))

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
    def get_num_layers(cls, pipeline_config: PipelineConfig) -> int:
        hf_config = pipeline_config.huggingface_config
        if hasattr(hf_config, "num_hidden_layers"):
            return hf_config.num_hidden_layers
        elif hasattr(hf_config, "num_layers"):
            return hf_config.num_layers
        elif hasattr(hf_config, "n_layers"):
            return hf_config.n_layers
        else:
            raise ValueError(
                "num_hidden_layers or num_layers or n_layers not found in huggingface_config"
            )

    @classmethod
    def get_kv_params(cls, pipeline_config: PipelineConfig) -> KVCacheParams:
        cache_dtype = (
            DType.float32
            if pipeline_config.quantization_encoding is not None
            and pipeline_config.quantization_encoding.quantization_encoding
            is not None
            else pipeline_config.dtype
        )
        hf_config = pipeline_config.huggingface_config
        num_kv_heads = cls._get_num_kv_heads(hf_config)
        hidden_size = cls._get_hidden_size(hf_config)

        return KVCacheParams(
            dtype=cache_dtype,
            n_kv_heads=num_kv_heads,
            head_dim=hidden_size // num_kv_heads,
            cache_strategy=pipeline_config.cache_strategy,
            enable_prefix_caching=pipeline_config.enable_prefix_caching,
            page_size=pipeline_config.kv_cache_page_size,
        )

    def load_kv_manager(
        self,
        session: InferenceSession,
        available_cache_memory: int,
    ) -> KVCacheManager:
        """Provided a PipelineConfig and InferenceSession, load the kv manager."""
        num_layers = self.get_num_layers(self.pipeline_config)

        return load_kv_manager(
            params=self.get_kv_params(self.pipeline_config),
            max_cache_batch_size=self.pipeline_config.max_cache_batch_size,
            max_seq_len=self.calculate_max_seq_len(self.pipeline_config),
            num_layers=num_layers,
            devices=self.pipeline_config.devices,
            available_cache_memory=available_cache_memory,
            session=session,
        )

    @classmethod
    def estimate_kv_cache_size(
        cls,
        pipeline_config: PipelineConfig,
        available_cache_memory: int,
        devices: list[Device],
    ) -> int:
        """Estimates the size of the kv cache in bytes."""
        num_layers = cls.get_num_layers(pipeline_config)

        return estimate_kv_cache_size(
            params=cls.get_kv_params(pipeline_config),
            max_cache_batch_size=pipeline_config.max_cache_batch_size,
            max_seq_len=cls.calculate_max_seq_len(pipeline_config),
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
    def calculate_max_seq_len(cls, pipeline_config: PipelineConfig) -> int:
        try:
            return upper_bounded_default(
                upper_bound=pipeline_config.huggingface_config.max_position_embeddings,
                default=pipeline_config.max_length,
            )
        except ValueError as e:
            msg = (
                "Unable to infer max_length for DummyModel, the provided "
                f"max_length ({pipeline_config.max_length}) exceeds the "
                f"model's max_position_embeddings "
                f"({pipeline_config.huggingface_config.max_position_embeddings})."
            )
            raise ValueError(msg) from e


class DummyReplitPipelineModel(DummyPipelineModel):
    @classmethod
    def calculate_max_seq_len(cls, pipeline_config: PipelineConfig) -> int:
        try:
            return upper_bounded_default(
                upper_bound=pipeline_config.huggingface_config.max_seq_len,
                default=pipeline_config.max_length,
            )
        except ValueError as e:
            msg = (
                "Unable to infer max_length for DummyModel, the provided "
                f"max_length ({pipeline_config.max_length}) exceeds the "
                f"model's max_seq_len "
                f"({pipeline_config.huggingface_config.max_seq_len})."
            )
            raise ValueError(msg) from e


DUMMY_ARCH = SupportedArchitecture(
    name="LlamaForCausalLM",
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
    weight_converters={WeightsFormat.safetensors: None},  # type: ignore
)

REPLIT_ARCH = SupportedArchitecture(
    name="MPTForCausalLM",
    example_repo_ids=["modularai/replit-code-1.5"],
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={
        SupportedEncoding.float32: [KVCacheStrategy.CONTINUOUS],
        SupportedEncoding.bfloat16: [KVCacheStrategy.CONTINUOUS],
    },
    pipeline_model=DummyReplitPipelineModel,
    tokenizer=TextTokenizer,
    default_weights_format=WeightsFormat.gguf,
    weight_converters={WeightsFormat.pytorch: None},  # type: ignore
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
        architecture="not_registered",
        huggingface_repo_id="modularai/llama-3.1",
        # This forces it to fail if we dont have it.
        engine=PipelineEngine.MAX,
        max_cache_batch_size=1,
        max_length=512,
    )

    with pytest.raises(ValueError):
        config = PIPELINE_REGISTRY.validate_pipeline_config(config)


@prepare_registry
def test_registry__test_retrieve_with_unknown_architecture_unknown_engine():
    PIPELINE_REGISTRY.register(DUMMY_ARCH)

    config = PipelineConfig(
        architecture="not_registered",
        huggingface_repo_id="modularai/llama-3.1",
        max_cache_batch_size=1,
        max_length=512,
    )

    config = PIPELINE_REGISTRY.validate_pipeline_config(config)
    assert config.engine == PipelineEngine.HUGGINGFACE


@prepare_registry
def test_registry__test_retrieve_factory_with_known_architecture():
    PIPELINE_REGISTRY.register(DUMMY_ARCH)

    config = PipelineConfig(
        architecture="LlamaForCausalLM",
        huggingface_repo_id="modularai/llama-3.1",
        max_cache_batch_size=1,
        max_length=512,
    )

    _, _ = PIPELINE_REGISTRY.retrieve_factory(pipeline_config=config)


@prepare_registry
def test_registry__test_retrieve_factory_with_unsupported_huggingface_repo_id():
    PIPELINE_REGISTRY.register(DUMMY_ARCH)

    config = PipelineConfig(
        huggingface_repo_id="modularai/replit-code-1.5",
        trust_remote_code=True,
        max_cache_batch_size=1,
        max_length=512,
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
        huggingface_repo_id="modularai/llama-3.1",
        max_cache_batch_size=1,
        max_length=512,
    )

    _, _ = PIPELINE_REGISTRY.retrieve_factory(pipeline_config=config)


@prepare_registry
def test_registry__test_incompatible_quantization_encoding():
    PIPELINE_REGISTRY.register(DUMMY_ARCH)

    # This should raise, as q4_k != bf16.
    config = PipelineConfig(
        huggingface_repo_id="modularai/llama-3.1",
        quantization_encoding=SupportedEncoding.q4_k,
        weight_path=[Path("llama-3.1-8b-instruct-bf16.gguf")],
        max_cache_batch_size=1,
        max_length=512,
    )

    with pytest.raises(ValueError):
        PIPELINE_REGISTRY.validate_pipeline_config(config)

    # This should not raise, as bfloat16 == bf16.
    config = PipelineConfig(
        huggingface_repo_id="modularai/llama-3.1",
        quantization_encoding=SupportedEncoding.bfloat16,
        weight_path=[Path("llama-3.1-8b-instruct-bf16.gguf")],
        max_cache_batch_size=1,
        max_length=512,
    )

    PIPELINE_REGISTRY.validate_pipeline_config(config)


@prepare_registry
def test_registry__update_cache_strategy():
    PIPELINE_REGISTRY.register(DUMMY_ARCH)

    config = PipelineConfig(
        huggingface_repo_id="modularai/llama-3.1",
        cache_strategy=KVCacheStrategy.NAIVE,
        max_cache_batch_size=1,
        max_length=512,
    )

    # Naive is not shown as supported in architecture, as
    # such this should change to a support strategy automatically.
    pipeline_config = PIPELINE_REGISTRY.validate_pipeline_config(config)
    assert pipeline_config.cache_strategy == KVCacheStrategy.CONTINUOUS


@prepare_registry
def test_registry__update_weight_paths():
    PIPELINE_REGISTRY.register(DUMMY_ARCH)
    PIPELINE_REGISTRY.register(REPLIT_ARCH)

    temp_valid_kernels = [
        ("f32", 4, 4),
        ("bf16", 4, 4),
        ("bf16", 24, 128),
        ("f32", 24, 128),
    ] + VALID_KV_KERNELS
    with patch(
        "max.pipelines.kv_cache.cache_params.VALID_KV_KERNELS",
        temp_valid_kernels,
    ):
        # This first example, is requesting float32 from a gguf repository.
        config = PipelineConfig(
            huggingface_repo_id="modularai/llama-3.1",
            quantization_encoding=SupportedEncoding.float32,
            max_cache_batch_size=1,
            max_length=512,
        )

        config = PIPELINE_REGISTRY.validate_pipeline_config(config)

        assert len(config.weight_path) == 1
        assert config.weight_path == [Path("llama-3.1-8b-instruct-f32.gguf")]

        # This second example, is requesting float32 from a safetensors repository.
        config = PipelineConfig(
            huggingface_repo_id="trl-internal-testing/tiny-random-LlamaForCausalLM",
            quantization_encoding=SupportedEncoding.float32,
            max_cache_batch_size=1,
            max_length=512,
        )

        config = PIPELINE_REGISTRY.validate_pipeline_config(config)

        assert len(config.weight_path) == 1
        assert config.weight_path == [Path("model.safetensors")]

        # This example, should raise, as you are requesting q6_k from a fp32
        # safetensors repo.
        config = PipelineConfig(
            huggingface_repo_id="trl-internal-testing/tiny-random-LlamaForCausalLM",
            quantization_encoding=SupportedEncoding.q6_k,
        )

        # This should raise, as this repository, does not have q6_k weights.
        with pytest.raises(
            ValueError, match="compatible weights cannot be found"
        ):
            PIPELINE_REGISTRY.validate_pipeline_config(config)

        # This example, should pass, since using fp32 weights for bfloat16 is
        # listed as an alternate encoding for fp32.
        config = PipelineConfig(
            huggingface_repo_id="trl-internal-testing/tiny-random-LlamaForCausalLM",
            quantization_encoding=SupportedEncoding.bfloat16,
            max_cache_batch_size=1,
            max_length=512,
        )

        # This should pass, since the float16 weights will be used for bfloat16.
        PIPELINE_REGISTRY.validate_pipeline_config(config)
        assert len(config.weight_path) == 1
        assert config.weight_path == [Path("model.safetensors")]

        # This example, should raise as we dont have q4_k listed as supported.
        config = PipelineConfig(
            huggingface_repo_id="trl-internal-testing/tiny-random-LlamaForCausalLM",
            quantization_encoding=SupportedEncoding.q4_k,
            engine=PipelineEngine.MAX,
            max_cache_batch_size=1,
            max_length=512,
        )

        with pytest.raises(ValueError):
            config = PIPELINE_REGISTRY.validate_pipeline_config(config)

        # This example, should raise as we dont have q4_k listed as supported.
        # If we don't pass MAX though, we should not fail and fall back to HuggingFace.
        config = PipelineConfig(
            huggingface_repo_id="trl-internal-testing/tiny-random-LlamaForCausalLM",
            quantization_encoding=SupportedEncoding.q4_k,
            max_cache_batch_size=1,
            max_length=512,
        )

        config = PIPELINE_REGISTRY.validate_pipeline_config(config)
        assert config.engine == PipelineEngine.HUGGINGFACE

        # This example, should not raise, as we are showing that we have a weight converter for pytorch for Replit.
        config = PipelineConfig(
            huggingface_repo_id="replit/replit-code-v1_5-3b",
            quantization_encoding=SupportedEncoding.bfloat16,
            trust_remote_code=True,
            max_cache_batch_size=1,
            max_length=512,
        )

        config = PIPELINE_REGISTRY.validate_pipeline_config(config)
        assert config.engine == PipelineEngine.MAX
        assert config.weight_path == [Path("pytorch_model.bin")]

        # Test a partially complete huggingface_repo
        config = PipelineConfig(
            huggingface_repo_id="neubla/tiny-random-LlamaForCausalLM",
            max_cache_batch_size=1,
            max_length=512,
        )
        config = PIPELINE_REGISTRY.validate_pipeline_config(config)
        assert config.quantization_encoding == SupportedEncoding.float32
        assert config.engine == PipelineEngine.MAX
        assert config.weight_path == [Path("model.safetensors")]

        # This example, should not raise as we are passing a valid weights path in a different repository.
        config = PipelineConfig(
            huggingface_repo_id="replit/replit-code-v1_5-3b",
            quantization_encoding=SupportedEncoding.float32,
            trust_remote_code=True,
            weight_path=[
                Path("modularai/replit-code-1.5/replit-code-v1_5-3b-f32.gguf")
            ],
            max_cache_batch_size=1,
            max_length=512,
        )

        config = PIPELINE_REGISTRY.validate_pipeline_config(config)
        assert config.engine == PipelineEngine.MAX
        assert config.weight_path == [Path("replit-code-v1_5-3b-f32.gguf")]
        assert config._weights_repo_id == "modularai/replit-code-1.5"
