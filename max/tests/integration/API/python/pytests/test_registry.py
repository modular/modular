# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from max.dtype import DType
from max.driver import Tensor
from max.graph import Graph, TensorType
from max.pipelines import (
    PIPELINE_REGISTRY,
    IdentityPipelineTokenizer,
    SupportedArchitecture,
    SupportedVersion,
    SupportedEncoding,
    PipelineModel,
)
from max.pipelines.context import InputContext
from max.pipelines.kv_cache import (
    KVCacheManager,
    KVCacheParams,
    load_kv_manager,
)
from max.engine import InferenceSession, Model


class DummyPipelineModel(PipelineModel):
    """A pipeline model with setup, input preparation and execution methods."""

    def execute(self, *model_inputs: Tensor) -> tuple[Tensor, ...]:
        """Runs the graph."""
        return model_inputs[0]  # type: ignore

    def prepare_initial_token_inputs(
        self, context_batch: list[InputContext]
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
            DType.float32 if self.pipeline_config.quantization_encoding.quantization_encoding  # type: ignore
            is not None else self.pipeline_config.dtype
        )
        return KVCacheParams(
            dtype=cache_dtype,
            n_kv_heads=self.pipeline_config.huggingface_config.num_key_value_heads,
            head_dim=self.pipeline_config.huggingface_config.hidden_size
            // self.pipeline_config.huggingface_config.num_attention_heads,
            cache_strategy=self.pipeline_config.cache_strategy,
        )

    def load_kv_manager(self, session: InferenceSession) -> KVCacheManager:
        """Provided a PipelineConfig and InferenceSession, load the kv manager.
        """
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
        """Provided a PipelineConfig and InferenceSession, build and load the model graph.
        """
        kv_inputs = self.kv_manager.input_symbols()
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


def test_registry():
    arch = SupportedArchitecture(
        name="llama",
        versions=[
            SupportedVersion(
                name="1",
                encodings=[
                    SupportedEncoding.float32,
                    SupportedEncoding.bfloat16,
                ],
                default_encoding=SupportedEncoding.float32,
            )
        ],
        default_version="1",
        pipeline_model=DummyPipelineModel,
        tokenizer=IdentityPipelineTokenizer,
    )

    PIPELINE_REGISTRY.register(arch)

    assert "llama" in PIPELINE_REGISTRY.architectures
