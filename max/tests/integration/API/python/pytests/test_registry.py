# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pytest
from functools import wraps
from typing import Sequence

from max.dtype import DType
from max.driver import Tensor
from max.graph import Graph, TensorType
from max.pipelines import (
    PIPELINE_REGISTRY,
    TextTokenizer,
    SupportedArchitecture,
    SupportedVersion,
    SupportedEncoding,
    PipelineModel,
    PipelineConfig,
    HuggingFaceFile,
    ModelOutputs,
    PipelineEngine,
)
from max.pipelines.context import InputContext
from max.pipelines.kv_cache import (
    KVCacheManager,
    KVCacheStrategy,
    KVCacheParams,
    load_kv_manager,
)
from max.engine import InferenceSession, Model


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
        return (Tensor.zeros(0, 0),)  # type: ignore

    def prepare_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: tuple[Tensor, ...],
    ) -> tuple[Tensor, ...]:
        """Prepares the secondary inputs to be passed to `.execute()`.

        While `prepare_initial_token_inputs` is responsible for managing the initial inputs.
        This function is responsible for updating the inputs, for each step in a multi-step execution pattern.
        """
        return (Tensor.zeros(0, 0),)  # type: ignore

    def _get_kv_params(self) -> KVCacheParams:
        cache_dtype = (
            DType.float32
            if self.pipeline_config.quantization_encoding.quantization_encoding  # type: ignore
            is not None
            else self.pipeline_config.dtype
        )
        return KVCacheParams(
            dtype=cache_dtype,
            n_kv_heads=self.pipeline_config.huggingface_config.num_key_value_heads,
            head_dim=self.pipeline_config.huggingface_config.hidden_size
            // self.pipeline_config.huggingface_config.num_attention_heads,
            cache_strategy=self.pipeline_config.cache_strategy,
        )

    def load_kv_manager(self, session: InferenceSession) -> KVCacheManager:
        """Provided a PipelineConfig and InferenceSession, load the kv manager."""
        return load_kv_manager(
            params=self._get_kv_params(),
            max_cache_batch_size=self.pipeline_config.max_cache_batch_size,
            max_seq_len=self.pipeline_config.huggingface_config.max_seq_len,
            num_layers=self.pipeline_config.huggingface_config.num_hidden_layers,
            devices=[self.pipeline_config.device],
            session=session,
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
            tokens, kv_inputs = graph.inputs  # type: ignore
            graph.output(tokens)
            return session.load(graph)


DUMMY_ARCH = SupportedArchitecture(
    name="LlamaForCausalLM",
    versions=[
        SupportedVersion(
            name="1",
            encodings={
                SupportedEncoding.float32: (
                    [
                        HuggingFaceFile(
                            "modularai/llama-3.1",
                            "llama-3.1-8b-instruct-f32.gguf",
                        )
                    ],
                    [KVCacheStrategy.CONTINUOUS],
                )
            },
            default_encoding=SupportedEncoding.float32,
        )
    ],
    default_version="1",
    pipeline_model=DummyPipelineModel,
    tokenizer=TextTokenizer,
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
    )

    config = PIPELINE_REGISTRY.validate_pipeline_config(config)
    assert config.engine == PipelineEngine.HUGGINGFACE


@prepare_registry
def test_registry__test_retrieve_factory_with_known_architecture():
    PIPELINE_REGISTRY.register(DUMMY_ARCH)

    config = PipelineConfig(
        architecture="LlamaForCausalLM",
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
