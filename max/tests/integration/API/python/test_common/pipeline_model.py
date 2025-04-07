# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from __future__ import annotations

from collections.abc import Sequence
from functools import wraps
from typing import Any, cast
from unittest.mock import patch

from max.driver import Device, Tensor, load_devices
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
        assert model_inputs.input1 is not None
        return ModelOutputs(
            next_token_logits=model_inputs.input1, logits=model_inputs.input1
        )

    @classmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        raise NotImplementedError("calculate_max_seq_len is not implemented")

    def prepare_initial_token_inputs(
        self,
        context_batch: Sequence[InputContext],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
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
            enable_kvcache_swapping_to_host=kv_cache_config.enable_kvcache_swapping_to_host,
            host_kvcache_swap_space_gb=kv_cache_config.host_kvcache_swap_space_gb,
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
                kv_cache_config=self.pipeline_config.model_config.kv_cache_config,
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
    multi_gpu_supported=True,
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

DUMMY_GPTQ_ARCH = SupportedArchitecture(
    name="LlamaForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4",
        "jakiAJK/DeepSeek-R1-Distill-Llama-8B_GPTQ-int4",
    ],
    default_encoding=SupportedEncoding.float32,
    supported_encodings={
        SupportedEncoding.gptq: [
            KVCacheStrategy.PAGED,
        ],
        SupportedEncoding.float32: [KVCacheStrategy.CONTINUOUS],
        SupportedEncoding.bfloat16: [KVCacheStrategy.CONTINUOUS],
    },
    pipeline_model=DummyLlamaPipelineModel,
    tokenizer=TextTokenizer,
    multi_gpu_supported=True,
    default_weights_format=WeightsFormat.gguf,
)


def prepare_registry(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        PIPELINE_REGISTRY.reset()
        result = func(*args, **kwargs)

        return result

    return wrapper


def mock_estimate_memory_footprint(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with patch.object(
            PIPELINE_REGISTRY, "_estimate_memory_footprint", return_value=0
        ):
            return func(*args, **kwargs)

    return wrapper
