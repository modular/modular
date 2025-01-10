# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from functools import wraps
from pathlib import Path
from typing import Sequence

import pytest
from max.driver import Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import Graph, TensorType
from max.pipelines import (
    PIPELINE_REGISTRY,
    ModelOutputs,
    PipelineConfig,
    PipelineEngine,
    PipelineModel,
    SupportedArchitecture,
    SupportedEncoding,
    TextTokenizer,
    WeightsFormat,
)
from max.pipelines.context import InputContext
from max.pipelines.kv_cache import (
    KVCacheManager,
    KVCacheParams,
    KVCacheStrategy,
    estimate_kv_cache_size,
    load_kv_manager,
)


def prepare_registry(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        PIPELINE_REGISTRY.reset()
        result = func(*args, **kwargs)

        return result

    return wrapper


class DummyPipelineModel(PipelineModel):
    """A pipeline model with setup, input preparation and execution methods."""

    def execute(self, *model_inputs: Tensor) -> ModelOutputs:
        """Runs the graph."""
        return ModelOutputs(next_token_logits=model_inputs[0])

    def prepare_initial_token_inputs(
        self, context_batch: Sequence[InputContext]
    ) -> tuple[Tensor, ...]:
        """Prepares the initial inputs to be passed to `.execute()`.

        The inputs and functionality of this method can vary per model.
        For example, the model inputs could include:
        - Encoded tensors
        - A unique IDs for each tensor if this model uses a KV Cache manager.

        This function would batch the encoded tensors, claim a slot in the kv
        cache if the ID hasn't been seen before, and return the inputs and
        caches as a list of tensors."""
        return (Tensor.zeros((0, 0), DType.float32),)

    def prepare_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: tuple[Tensor, ...],
    ) -> tuple[Tensor, ...]:
        """Prepares the secondary inputs to be passed to `.execute()`.

        While `prepare_initial_token_inputs` is responsible for managing the initial inputs.
        This function is responsible for updating the inputs, for each step in a multi-step execution pattern.
        """
        return (Tensor.zeros((0, 0), DType.float32),)

    def _get_kv_params(self) -> KVCacheParams:
        cache_dtype = (
            DType.float32
            if self.pipeline_config.quantization_encoding is not None
            and self.pipeline_config.quantization_encoding.quantization_encoding
            is not None
            else self.pipeline_config.dtype
        )
        return KVCacheParams(
            dtype=cache_dtype,
            n_kv_heads=self.pipeline_config.huggingface_config.num_key_value_heads,
            head_dim=self.pipeline_config.huggingface_config.hidden_size
            // self.pipeline_config.huggingface_config.num_attention_heads,
            cache_strategy=self.pipeline_config.cache_strategy,
            enable_prefix_caching=self.pipeline_config.enable_prefix_caching,
        )

    def load_kv_manager(
        self,
        session: InferenceSession,
        available_cache_memory: int,
    ) -> KVCacheManager:
        """Provided a PipelineConfig and InferenceSession, load the kv manager."""
        return load_kv_manager(
            params=self._get_kv_params(),
            max_cache_batch_size=self.pipeline_config.max_cache_batch_size,
            max_seq_len=self.pipeline_config.huggingface_config.max_seq_len,
            num_layers=self.pipeline_config.huggingface_config.num_hidden_layers,
            devices=self.pipeline_config.devices,
            available_cache_memory=available_cache_memory,
            session=session,
        )

    def estimate_kv_cache_size(self, available_cache_memory: int) -> int:
        return estimate_kv_cache_size(
            params=self._get_kv_params(),
            max_cache_batch_size=self.pipeline_config.max_cache_batch_size,
            max_seq_len=self.pipeline_config.huggingface_config.max_seq_len,
            num_layers=self.pipeline_config.huggingface_config.num_hidden_layers,
            available_cache_memory=available_cache_memory,
            devices=self.pipeline_config.devices,
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


DUMMY_ARCH = SupportedArchitecture(
    name="LlamaForCausalLM",
    example_repo_ids=["modularai/llama-3.1"],
    default_encoding=SupportedEncoding.float32,
    supported_encodings={
        SupportedEncoding.float32: [KVCacheStrategy.CONTINUOUS],
        SupportedEncoding.bfloat16: [KVCacheStrategy.CONTINUOUS],
        SupportedEncoding.q6_k: [KVCacheStrategy.CONTINUOUS],
    },
    pipeline_model=DummyPipelineModel,
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
    pipeline_model=DummyPipelineModel,
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
    )

    with pytest.raises(ValueError):
        config = PIPELINE_REGISTRY.validate_pipeline_config(config)


@prepare_registry
def test_registry__test_retrieve_with_unknown_architecture_unknown_engine():
    PIPELINE_REGISTRY.register(DUMMY_ARCH)

    config = PipelineConfig(
        architecture="not_registered",
        huggingface_repo_id="modularai/llama-3.1",
    )

    config = PIPELINE_REGISTRY.validate_pipeline_config(config)
    assert config.engine == PipelineEngine.HUGGINGFACE


@prepare_registry
def test_registry__test_retrieve_factory_with_known_architecture():
    PIPELINE_REGISTRY.register(DUMMY_ARCH)

    config = PipelineConfig(
        architecture="LlamaForCausalLM",
        huggingface_repo_id="modularai/llama-3.1",
    )

    _, _ = PIPELINE_REGISTRY.retrieve_factory(pipeline_config=config)


@prepare_registry
def test_registry__test_retrieve_factory_with_unsupported_huggingface_repo_id():
    PIPELINE_REGISTRY.register(DUMMY_ARCH)

    config = PipelineConfig(
        huggingface_repo_id="modularai/replit-code-1.5",
        trust_remote_code=True,
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
    )

    with pytest.raises(ValueError):
        PIPELINE_REGISTRY.validate_pipeline_config(config)

    # This should not raise, as bfloat16 == bf16.
    config = PipelineConfig(
        huggingface_repo_id="modularai/llama-3.1",
        quantization_encoding=SupportedEncoding.bfloat16,
        weight_path=[Path("llama-3.1-8b-instruct-bf16.gguf")],
    )

    PIPELINE_REGISTRY.validate_pipeline_config(config)


@prepare_registry
def test_registry__update_cache_strategy():
    PIPELINE_REGISTRY.register(DUMMY_ARCH)

    config = PipelineConfig(
        huggingface_repo_id="modularai/llama-3.1",
        cache_strategy=KVCacheStrategy.NAIVE,
    )

    # Naive is not shown as supported in architecture, as
    # such this should change to a support strategy automatically.
    pipeline_config = PIPELINE_REGISTRY.validate_pipeline_config(config)
    assert pipeline_config.cache_strategy == KVCacheStrategy.CONTINUOUS


@prepare_registry
def test_registry__update_weight_paths():
    PIPELINE_REGISTRY.register(DUMMY_ARCH)
    PIPELINE_REGISTRY.register(REPLIT_ARCH)

    # This first example, is requesting float32 from a gguf repository.
    config = PipelineConfig(
        huggingface_repo_id="modularai/llama-3.1",
        quantization_encoding=SupportedEncoding.float32,
    )

    config = PIPELINE_REGISTRY.validate_pipeline_config(config)

    assert len(config.weight_path) == 1
    assert config.weight_path == [Path("llama-3.1-8b-instruct-f32.gguf")]

    # This second example, is requesting float32 from a safetensors repository.
    config = PipelineConfig(
        huggingface_repo_id="trl-internal-testing/tiny-random-LlamaForCausalLM",
        quantization_encoding=SupportedEncoding.float32,
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
    with pytest.raises(ValueError, match="compatible weights cannot be found"):
        PIPELINE_REGISTRY.validate_pipeline_config(config)

    # This example, should pass, since using fp32 weights for bfloat16 is
    # listed as an alternate encoding for fp32.
    config = PipelineConfig(
        huggingface_repo_id="trl-internal-testing/tiny-random-LlamaForCausalLM",
        quantization_encoding=SupportedEncoding.bfloat16,
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
    )

    with pytest.raises(ValueError):
        config = PIPELINE_REGISTRY.validate_pipeline_config(config)

    # This example, should raise as we dont have q4_k listed as supported.
    # If we don't pass MAX though, we should not fail and fall back to HuggingFace.
    config = PipelineConfig(
        huggingface_repo_id="trl-internal-testing/tiny-random-LlamaForCausalLM",
        quantization_encoding=SupportedEncoding.q4_k,
    )

    config = PIPELINE_REGISTRY.validate_pipeline_config(config)
    assert config.engine == PipelineEngine.HUGGINGFACE

    # This example, should not raise, as we are showing that we have a weight converter for pytorch for Replit.
    config = PipelineConfig(
        huggingface_repo_id="replit/replit-code-v1_5-3b",
        quantization_encoding=SupportedEncoding.bfloat16,
        trust_remote_code=True,
    )

    config = PIPELINE_REGISTRY.validate_pipeline_config(config)
    assert config.engine == PipelineEngine.MAX
    assert config.weight_path == [Path("pytorch_model.bin")]

    # Test a partially complete huggingface_repo
    config = PipelineConfig(
        huggingface_repo_id="neubla/tiny-random-LlamaForCausalLM",
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
    )

    config = PIPELINE_REGISTRY.validate_pipeline_config(config)
    assert config.engine == PipelineEngine.MAX
    assert config.weight_path == [Path("replit-code-v1_5-3b-f32.gguf")]
    assert config._weights_repo_id == "modularai/replit-code-1.5"
