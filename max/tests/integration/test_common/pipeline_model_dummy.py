# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType
from max.graph.weights import WeightsFormat
from max.interfaces import (
    PipelineTask,
    TextGenerationContext,
    TextGenerationRequest,
    TokenBuffer,
)
from max.nn.legacy.kv_cache import (
    KVCacheInputs,
    KVCacheParams,
    KVCacheQuantizationConfig,
)
from max.pipelines import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
    SupportedArchitecture,
    SupportedEncoding,
    TextContext,
    TextTokenizer,
    upper_bounded_default,
)
from max.pipelines.lib import KVCacheMixin
from max.pipelines.lib.interfaces import ArchConfigWithAttentionKVCache
from transformers import AutoConfig


@dataclass
class DummyModelInputs(ModelInputs):
    input1: Buffer | None = None
    input2: Buffer | None = None
    input3: Buffer | None = None
    input4: Buffer | None = None


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
        replica_batches: Sequence[Sequence[TextGenerationContext]],
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
            input1=Buffer.zeros((1, 5), DType.float32),
            input2=Buffer.zeros((0, 0), DType.float32),
            input3=Buffer.zeros((0, 0), DType.float32),
            input4=Buffer.zeros((0, 0), DType.float32),
            kv_cache_inputs=None,
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Buffer,
        prev_model_inputs: ModelInputs,
    ) -> DummyModelInputs:
        """Prepares the secondary inputs to be passed to `.execute()`.

        While `prepare_initial_token_inputs` is responsible for managing the initial inputs.
        This function is responsible for updating the inputs, for each step in a multi-step execution pattern.
        """
        return DummyModelInputs(
            input1=Buffer.zeros((0, 0), DType.float32),
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
    def _get_num_layers(cls, huggingface_config: AutoConfig) -> int:
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
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        num_kv_heads = cls._get_num_kv_heads(huggingface_config)
        hidden_size = cls._get_hidden_size(huggingface_config)
        head_dim = hidden_size // num_kv_heads

        kvcache_quant_config = None
        if cache_dtype in (
            DType.float8_e4m3fn,
            DType.float8_e4m3fnuz,
        ):
            # Configure the KVCacheParams quantization parameters.
            kvcache_quant_config = KVCacheQuantizationConfig(
                scale_dtype=DType.float32,
                quantization_granularity=head_dim // 2,
            )

        return KVCacheParams(
            dtype=cache_dtype,
            n_kv_heads=num_kv_heads,
            head_dim=head_dim,
            num_layers=cls._get_num_layers(huggingface_config),
            cache_strategy=kv_cache_config.cache_strategy,
            enable_prefix_caching=kv_cache_config.enable_prefix_caching,
            enable_kvcache_swapping_to_host=kv_cache_config.enable_kvcache_swapping_to_host,
            host_kvcache_swap_space_gb=kv_cache_config.host_kvcache_swap_space_gb,
            page_size=kv_cache_config.kv_cache_page_size,
            devices=devices,
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
            kvcache_quant_config=kvcache_quant_config,
        )

    def load_model(
        self,
        session: InferenceSession,
    ) -> Model:
        """Provided a PipelineConfig and InferenceSession, build and load the model graph."""
        kv_inputs = self.kv_params.get_symbolic_inputs()[0]
        with Graph(
            "dummy",
            input_types=[
                TensorType(
                    DType.int64, shape=["batch_size"], device=DeviceRef.CPU()
                ),
                *kv_inputs,
            ],
        ) as graph:
            tokens, _kv_inputs_value = graph.inputs
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
                default=pipeline_config.model.max_length,
            )
        except ValueError as e:
            raise ValueError(
                "Unable to infer max_length for DummyModel, the provided "
                f"max_length ({pipeline_config.model.max_length}) exceeds the "
                f"model's max_position_embeddings "
                f"({huggingface_config.max_position_embeddings})."
            ) from e


class DummyTextTokenizer(TextTokenizer):
    def __init__(
        self, model_path: str, pipeline_config: PipelineConfig, *args, **kwargs
    ) -> None:
        self.max_length = pipeline_config.model.max_length or 100
        self.delegate = DummyTextTokenizer.Delegate(max_length=self.max_length)

    @property
    def eos(self) -> int:
        """The end of sequence token for this tokenizer."""
        return -1

    @property
    def expects_content_wrapping(self) -> bool:
        return False

    async def new_context(self, request: TextGenerationRequest) -> TextContext:
        prompt: str | Sequence[int]
        if request.prompt is None:
            # Definitely not the correct way to do this, but it's for testing
            # purposes.
            prompt = str(request.messages)
        else:
            prompt = request.prompt
        token_ids = await self.encode(prompt)

        max_length = self.max_length
        if request.sampling_params.max_new_tokens is not None:
            max_length = min(
                self.max_length,
                token_ids.shape[0] + request.sampling_params.max_new_tokens,
            )

        return TextContext(
            request_id=request.request_id,
            max_length=max_length,
            tokens=TokenBuffer(token_ids.astype(np.int64, copy=False)),
            log_probabilities=request.logprobs,
            log_probabilities_echo=request.echo,
            sampling_params=request.sampling_params,
        )

    async def encode(
        self, prompt: str | Sequence[int], add_special_tokens: bool = True
    ) -> npt.NDArray[np.integer[Any]]:
        return self.delegate.encode(prompt, add_special_tokens)

    async def decode(
        self, encoded: npt.NDArray[np.integer[Any]], **kwargs
    ) -> str:
        return self.delegate.decode(encoded, **kwargs)

    class Delegate:
        def __init__(self, max_length: int) -> None:
            self.max_length = max_length

        def encode(
            self, prompt: str | Sequence[int], add_special_tokens: bool = True
        ) -> npt.NDArray[np.integer[Any]]:
            if isinstance(prompt, str):
                return np.array([ord(c) for c in prompt[: self.max_length]])
            else:
                return np.array(prompt[: self.max_length])

        def decode(
            self, encoded: npt.NDArray[np.integer[Any]], **kwargs
        ) -> str:
            return "".join([chr(i) for i in encoded])


@dataclass(kw_only=True)
class DummyLlamaArchConfig(ArchConfigWithAttentionKVCache):
    @property
    def num_key_value_heads(self) -> int:
        """Number of key-value heads to use for the KV cache."""
        return DummyLlamaPipelineModel._get_num_kv_heads(
            self.huggingface_config
        )

    @property
    def head_dim(self) -> int:
        """Dimensionality of each attention head."""
        hidden_size = DummyLlamaPipelineModel._get_hidden_size(
            self.huggingface_config
        )
        num_kv_heads = DummyLlamaPipelineModel._get_num_kv_heads(
            self.huggingface_config
        )
        return hidden_size // num_kv_heads

    @property
    def num_layers(self) -> int:
        """Number of hidden layers in the model."""
        assert self.huggingface_config is not None
        return DummyLlamaPipelineModel._get_num_layers(self.huggingface_config)

    @property
    def model_max_seq_len(self) -> int:
        """The maximum sequence length that can be processed by the model."""
        assert self.huggingface_config is not None
        return self.huggingface_config.max_position_embeddings


DUMMY_LLAMA_ARCH = SupportedArchitecture(
    name="LlamaForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "HuggingFaceTB/SmolLM-135M",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "deepseek-ai/deepseek-coder-6.7b-instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-Guard-3-8B",
        "modularai/Llama-3.1-8B-Instruct-GGUF",
        "trl-internal-testing/tiny-random-LlamaForCausalLM",
    ],
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={
        SupportedEncoding.gptq: ["paged"],
        # q4_k intentionally left out to test a valid SupportedEncoding but not
        # supported by the model (supported_encoding).
        SupportedEncoding.q4_0: ["paged"],
        SupportedEncoding.q6_k: ["paged"],
        SupportedEncoding.float32: ["paged"],
        SupportedEncoding.bfloat16: ["paged"],
        SupportedEncoding.float8_e4m3fn: ["paged"],
    },
    pipeline_model=DummyLlamaPipelineModel,
    tokenizer=DummyTextTokenizer,
    context_type=TextContext,
    multi_gpu_supported=True,
    default_weights_format=WeightsFormat.gguf,
    config=DummyLlamaArchConfig,
)

DUMMY_LLAMA_GPTQ_ARCH = SupportedArchitecture(
    name="LlamaForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4",
        "jakiAJK/DeepSeek-R1-Distill-Llama-8B_GPTQ-int4",
        "modularai/Llama-3.1-8B-Instruct-GGUF",
    ],
    default_encoding=SupportedEncoding.float32,
    supported_encodings={
        SupportedEncoding.gptq: ["paged"],
        SupportedEncoding.float32: ["paged"],
        SupportedEncoding.bfloat16: ["paged"],
    },
    pipeline_model=DummyLlamaPipelineModel,
    tokenizer=DummyTextTokenizer,
    context_type=TextContext,
    multi_gpu_supported=True,
    default_weights_format=WeightsFormat.gguf,
    config=DummyLlamaArchConfig,
)

DUMMY_GEMMA_ARCH = SupportedArchitecture(
    name="Gemma3ForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        # it = Instruction tuned (recommended).
        # pt = Pre-trained.
        "google/gemma-3-1b-it",
        "google/gemma-3-1b-pt",
        # TODO(MODELS-487): >=4B models have a slightly different architecture
        # and config and use a different rotary embedding. These will likely
        # need a separate SupportedArchitecture registration.
        # "google/gemma-3-4b-it",
        # "google/gemma-3-4b-pt",
        # "google/gemma-3-12b-it",
        # "google/gemma-3-12b-pt",
        # "google/gemma-3-27b-it",
        # "google/gemma-3-27b-pt",
    ],
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={
        SupportedEncoding.bfloat16: ["paged"],
    },
    pipeline_model=DummyPipelineModel,
    tokenizer=DummyTextTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    rope_type="normal",
    multi_gpu_supported=False,
    config=DummyLlamaArchConfig,
)

ARCHITECTURES = [DUMMY_LLAMA_ARCH, DUMMY_GEMMA_ARCH]
