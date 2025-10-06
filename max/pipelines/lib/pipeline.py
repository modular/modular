# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
# mypy: disable-error-code="import-not-found"
"""MAX pipeline for model inference and generation."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Iterable, Sequence
from dataclasses import dataclass
from os import environ
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Protocol,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)

import llguidance.hf
import llguidance.numpy
import numpy as np
import numpy.typing as npt
from llguidance import LLMatcher
from max.driver import Device, Tensor, load_devices
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.graph.weights import (
    Weights,
    WeightsAdapter,
    WeightsFormat,
    load_weights,
    weights_format,
)
from max.interfaces import (
    BatchLogitsProcessor,
    GenerationStatus,
    InputContext,
    LogProbabilities,
    Pipeline,
    PipelineOutputsDict,
    PipelineTokenizer,
    RequestID,
    RequestType,
    TextGenerationInputs,
    TextGenerationOutput,
    TextGenerationRequest,
)
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheInputsSequence,
    KVCacheParams,
    MultiPagedKVCacheManager,
    PagedKVCacheManager,
    infer_optimal_batch_size,
)
from max.nn.transformer import ReturnLogits
from max.profiler import Tracer, traced
from transformers import AutoConfig, PreTrainedTokenizerFast

if TYPE_CHECKING:
    from .config import PipelineConfig

from .config_enums import RepoType, SupportedEncoding
from .hf_utils import download_weight_files
from .kv_cache_config import KVCacheConfig
from .lora import LoRAManager
from .sampling import (
    FusedSamplingProcessor,
    apply_logits_processors,
    token_sampler,
)

logger = logging.getLogger("max.pipelines")


def upper_bounded_default(upper_bound: int, default: int | None) -> int:
    """
    Given an upper bound and an optional default value, returns a final value
    that cannot exceed the upper bound.

    Args:
        default: The default value to use, or None to use the upper bound.
        upper_bound: The upper bound to use.

    Raises:
        ValueError: If the provided default value exceeds the upper bound.

    Returns:
        The final value.
    """
    if default is None:
        return upper_bound
    elif default > upper_bound:
        raise ValueError(
            f"default value provided ({default}) exceeds the upper bound ({upper_bound})"
        )
    return default


class ModelInputs:
    """
    Base class for model inputs.
    Use this class to encapsulate inputs for your model.
    You may store any number of dataclass fields

    The following example demonstrates how to create a custom inputs class for a model:

    .. code-block:: python

        class ReplitInputs(ModelInputs):
            tokens: Tensor
            input_row_offsets: Tensor

            def __init__(self, tokens: Tensor, input_row_offsets: Tensor):
                self.tokens = tokens
                self.input_row_offsets = input_row_offsets

        # Create tensors
        tokens = Tensor.zeros((1, 2, 3), DType.int64)
        input_row_offsets = Tensor.zeros((1, 1, 1), DType.int64)

        # Initialize inputs
        inputs = ReplitInputs(tokens=tokens, input_row_offsets=input_row_offsets)

        # Access tensors
        list(inputs) == [tokens, input_row_offsets]  # Output: True
    """

    kv_cache_inputs: KVCacheInputs | None = None

    lora_ids: Tensor | None = None
    """Tensor containing the LoRA ids."""

    lora_ranks: Tensor | None = None
    """Tensor containing the LoRA ranks"""

    def update(self, **kwargs) -> None:
        key: str
        value: Any
        for key, value in kwargs.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)


@dataclass(frozen=True)
class ModelOutputs:
    logits: Tensor
    """Logits for a variable number of tokens per sequence."""

    next_token_logits: Tensor | None = None
    """Logits for just the next token."""

    logit_offsets: Tensor | None = None
    """Offsets to access variable length logits for each sequence."""


T = TypeVar("T", bound=InputContext)


class PipelineModel(ABC, Generic[T]):
    """A pipeline model with setup, input preparation and execution methods."""

    _MAX_DEFAULT_BATCH_SIZE = 4096
    _MIN_DEFAULT_BATCH_SIZE = 1

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        # TODO: This is no longer necessary inside PipelineModel since it can be
        # inferred directly from model_config, remove it and from
        # other PipelineModel methods that depend on it.
        huggingface_config: AutoConfig,
        encoding: SupportedEncoding,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None,
        return_logits: ReturnLogits,
    ) -> None:
        self.pipeline_config = pipeline_config
        self.huggingface_config = huggingface_config
        self.encoding = encoding
        self.devices = devices
        self.kv_cache_config = kv_cache_config
        self.weights = weights
        self.adapter = adapter
        self.return_logits = return_logits

        # Initialize `max_seq_len` here to avoid repeated HF config access.
        self.max_seq_len = self.calculate_max_seq_len(
            pipeline_config, huggingface_config
        )

        if isinstance(self, KVCacheMixin):
            self.kv_manager = self.load_kv_manager(
                session, self.kv_cache_config._available_cache_memory
            )

        self._lora_manager: LoRAManager | None = (
            LoRAManager(
                pipeline_config.lora_config,
                pipeline_config.model_config.model_name,
                self.dtype,
                pipeline_config.zmq_endpoint_base,
            )
            if pipeline_config.lora_config
            else None
        )

    @property
    def lora_manager(self) -> LoRAManager | None:
        return self._lora_manager

    @property
    def dtype(self) -> DType:
        # AudioGeneratorPipeline passes Nones for all args except pipeline config
        return (
            self.encoding.dtype
            if self.encoding is not None
            else self.pipeline_config.model_config.quantization_encoding.dtype
        )

    @classmethod
    @abstractmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        """Calculate the optimal max sequence length for the model.
        Models are expected to implement this method.

        The following example shows how to implement this method for a Mistral model:

        .. code-block:: python

            class MistralModel(PipelineModel):
                @classmethod
                def calculate_max_seq_len(cls, pipeline_config, huggingface_config) -> int:
                    try:
                        return upper_bounded_default(
                            upper_bound=huggingface_config.max_seq_len,
                            default=pipeline_config.max_length,
                        )
                    except ValueError as e:
                        raise ValueError(
                            "Unable to infer max_length for Mistral, the provided "
                            f"max_length ({pipeline_config.max_length}) exceeds the "
                            f"model's max_seq_len ({huggingface_config.max_seq_len})."
                        ) from e

        Args:
            pipeline_config: Configuration for the pipeline.
            huggingface_config: Hugging Face model configuration.

        Returns:
            int: The maximum sequence length to use.
        """
        raise NotImplementedError(
            "PipelineModel must implement calculate_max_seq_len"
        )

    @classmethod
    def infer_optimal_batch_size(
        cls,
        pipeline_config: PipelineConfig,
        available_cache_memory: int,
        huggingface_config: AutoConfig,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> int:
        """Returns the estimated optimal batch size to run the model
        given current memory constraints."""
        if not issubclass(cls, KVCacheMixin):
            # we rely on the KVCache setup to know optimal batch size.
            # If we don't have that, default to BS=1.
            return 1
        elif len(devices) == 1 and devices[0].is_host:
            # batching on CPU is generally not useful, so we hard-code a batch size of 1.
            return 1

        # TODO we should map HF configs to a unified MAX Config object
        # this would help avoid these excessive calls to class methods.
        n_layers = cls.get_num_layers(huggingface_config=huggingface_config)

        kv_params = cls.get_kv_params(
            huggingface_config=huggingface_config,
            n_devices=len(devices),
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )
        inferred_batch_size = infer_optimal_batch_size(
            params=kv_params,
            max_seq_len=cls.calculate_max_seq_len(
                pipeline_config, huggingface_config=huggingface_config
            ),
            num_layers=n_layers,
            available_cache_memory=available_cache_memory,
            devices=devices,
        )

        # clamp the floor of the inferred batch size to 1 and the ceiling to 4096
        inferred_batch_size = max(
            cls._MIN_DEFAULT_BATCH_SIZE,
            min(inferred_batch_size, cls._MAX_DEFAULT_BATCH_SIZE),
        )
        return inferred_batch_size

    @classmethod
    def estimate_weights_size(cls, pipeline_config: PipelineConfig) -> int:
        """Calculates the estimated memory consumption of our model."""

        # TODO move this logic to the PipelineModel instead of PipelineConfig class.
        # Better yet, make this more accurate by loading and measuring memory consumption
        # after we load the model
        return pipeline_config.model_config.weights_size()

    @classmethod
    def estimate_activation_memory(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        """Estimates the activation memory required for model execution.

        This accounts for temporary memory buffers used during model execution,
        such as intermediate activations and working buffers.

        The default implementation returns 0 for backward compatibility.
        Models with significant activation memory requirements should override
        this method to provide accurate estimates.

        Args:
            pipeline_config: Pipeline configuration
            huggingface_config: HuggingFace model configuration

        Returns:
            Estimated activation memory in bytes
        """
        return 0

    @abstractmethod
    def execute(
        self,
        model_inputs: ModelInputs,
    ) -> ModelOutputs:
        """Executes the graph with the given inputs.

        Args:
            model_inputs: The model inputs to execute, containing tensors and any other
                required data for model execution.

        Returns:
            ModelOutputs containing the pipeline's output tensors.

        This is an abstract method that must be implemented by concrete PipelineModels
        to define their specific execution logic.
        """

    @abstractmethod
    def prepare_initial_token_inputs(
        self,
        context_batch: Sequence[T],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> ModelInputs:
        """Prepares the initial inputs to be passed to `.execute()`.

        The inputs and functionality of this method can vary per model.
        For example, the model inputs could include:
        - Encoded tensors
        - A unique IDs for each tensor if this model uses a KV Cache manager.
        - kv_cache_inputs: The kv cache inputs required for the model. This
        should be None if the model does not use KV Cache.
        This function would batch the encoded tensors, claim a slot in the kv
        cache if the ID hasn't been seen before, and return the inputs and
        caches as a list of tensors."""
        ...

    @abstractmethod
    def prepare_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: ModelInputs,
    ) -> ModelInputs:
        """Prepares the secondary inputs to be passed to `.execute()`.

        While `prepare_initial_token_inputs` is responsible for managing the initial inputs.
        This function is responsible for updating the inputs, for each step in a multi-step execution pattern.
        """
        ...

    def compute_log_probabilities(
        self,
        session: InferenceSession,
        model_inputs: ModelInputs,
        model_outputs: ModelOutputs,
        next_tokens: Tensor,
        batch_top_n: list[int],
        batch_echo: list[bool],
    ) -> list[LogProbabilities | None] | None:
        """Optional method that can be overridden to compute log probabilities.

        Args:
            session: Inference session to compute log probabilities within.
            model_inputs: Inputs to the model returned by
                `prepare_*_token_inputs()`.
            model_outputs: Outputs returned by `execute()`.
            next_tokens: Sampled tokens. Should have shape=[batch size]
            batch_top_n: Number of top log probabilities to return per input in
                the batch. For any element where `top_n == 0`, the
                LogProbabilities is skipped.
            batch_echo: Whether to include input tokens in the returned log
                probabilities.

        Returns:
            List of log probabilities.
        """
        raise NotImplementedError(
            f"Log probabilities not implemented for {type(self)}."
        )


@runtime_checkable
class KVCacheMixin(Protocol):
    def load_kv_manager(
        self, session: InferenceSession, available_cache_memory: int | None
    ) -> PagedKVCacheManager:
        """Provided a PipelineConfig and InferenceSession, loads the KV manager.

        Args:
            session: Inference session to compile and init the KV cache.
            available_cache_memory: Amount of memory available to the KV cache,
                in bytes.

        Returns:
            Either a single KV cache manager or a tuple of KV cache managers:
            one per input modality.
        """
        ...

    # TODO(AITLIB-265): Remove this altogether from all PipelineModels.
    @classmethod
    @abstractmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        """Returns the KV cache params for the pipeline model."""
        ...

    # TODO(AITLIB-265): Remove this altogether from all PipelineModels.
    @classmethod
    @abstractmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        """Returns the number of layers for the pipeline model."""
        ...

    @classmethod
    @abstractmethod
    def estimate_kv_cache_size(
        cls,
        pipeline_config: PipelineConfig,
        available_cache_memory: int,
        devices: list[Device],
        huggingface_config: AutoConfig,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> int:
        """Estimates the size of the kv cache in bytes."""
        ...


def get_paged_manager(
    pipeline: Pipeline[Any, Any],
) -> PagedKVCacheManager | None:
    if (
        hasattr(pipeline, "_pipeline_model")
        and hasattr(pipeline._pipeline_model, "kv_manager")
        and isinstance(pipeline._pipeline_model.kv_manager, PagedKVCacheManager)
    ):
        return pipeline._pipeline_model.kv_manager

    return None


@dataclasses.dataclass
class BatchInfo:
    """Information about a batch of requests passed to the pipeline"""

    past_seq_lens: list[int]
    """Coordinated list of past sequence lengths (i.e. context lengths)"""

    seq_lens: list[int]
    """Coordinated list of sequence lengths, i.e. prompt_len or 1"""

    num_steps: int
    """Number of steps to do in the pipeline"""


@runtime_checkable
class _TextGenerationProtocol(Protocol, Generic[T, RequestType]):
    @property
    def kv_managers(self) -> list[PagedKVCacheManager]: ...

    @property
    def pipeline_config(self) -> PipelineConfig: ...

    @property
    def tokenizer(
        self,
    ) -> PipelineTokenizer[T, npt.NDArray[np.integer[Any]], RequestType]: ...

    def execute(
        self,
        inputs: TextGenerationInputs[T],
    ) -> PipelineOutputsDict[TextGenerationOutput]: ...

    def release(self, request_id: RequestID) -> None: ...


class GenerateMixin(
    _TextGenerationProtocol[T, RequestType], Generic[T, RequestType]
):
    def generate(
        self, prompts: RequestType | list[RequestType]
    ) -> list[TextGenerationOutput]:
        """Generates outputs for the given prompts."""
        if not isinstance(prompts, list):
            prompts = [prompts]

        async def _generate() -> list[TextGenerationOutput]:
            res = [
                TextGenerationOutput(
                    request_id=prompt.request_id,
                    tokens=[],
                    final_status=GenerationStatus.ACTIVE,
                )
                for prompt in prompts
            ]
            async for outputs in self.generate_async(prompts):
                for i, output in enumerate(outputs):
                    res[i].tokens.extend(output.tokens)
                    if output.is_done:
                        res[i].final_status = output.final_status
            return res

        return asyncio.run(_generate())

    async def generate_async(
        self, prompts: RequestType | list[RequestType]
    ) -> AsyncGenerator[list[TextGenerationOutput], None]:
        if not isinstance(prompts, list):
            prompts = [prompts]

        context_batch: list[T] = []
        for prompt in prompts:
            context = await self.tokenizer.new_context(prompt)
            context_batch.append(context)

        kv_managers = self.kv_managers
        data_parallel_degree = (
            self.pipeline_config.model_config.data_parallel_degree
        )

        # Create inputs to the model. If data parallelism is enabled, group them
        # by replica.
        batches: list[dict[RequestID, T]] = []
        batch_to_replica_idx: dict[RequestID, int] = {}
        if data_parallel_degree > 1 and len(kv_managers) > 1:
            # We don't support speculative decoding when data parallelism is
            # enabled, because the KV cache managers might place the same
            # context on different devices/replicas.
            raise ValueError(
                "Having multiple KV managers (e.g. when using"
                " speculative decoding) is not supported when data "
                "parallelism is enabled."
            )
        if data_parallel_degree > 1 and not isinstance(
            kv_managers[0], MultiPagedKVCacheManager
        ):
            raise ValueError(
                "MultiPagedKVCacheManager is required when data parallelism is enabled."
            )
        batches = [{} for _ in range(data_parallel_degree)]
        for context in context_batch:
            req_id = context.request_id
            # Use whatever replica the main models KVCache recommends.
            replica_idx = kv_managers[0].get_or_recommend_replica(context)
            # Claim the slot for all kv_managers (eg: main + draft model)
            for kv_manager in self.kv_managers:
                kv_manager.external_claim(req_id, replica_idx=replica_idx)
            batches[replica_idx][req_id] = context
            batch_to_replica_idx[req_id] = replica_idx
        inputs = TextGenerationInputs(
            batches=batches,
            num_steps=self.pipeline_config.max_num_steps,
        )

        # Generate outputs until all requests are done.

        done = 0

        try:
            while done < len(context_batch):
                step_outputs = self.execute(inputs)
                outputs = []
                for request_id, output in step_outputs.items():
                    outputs.append(output)
                    if output.is_done:
                        done += 1
                        # Remove the request from the batch passed to the next
                        # call to execute.
                        replica_idx = batch_to_replica_idx[request_id]
                        batches[replica_idx].pop(request_id)

                        self.release(request_id)
                yield outputs

                # Yield to the event loop.  If at no other point (e.g.
                # tokenizer.decode which we await earlier does not yield to the
                # event loop), it will be at this point that we'll receive a
                # CancelledError if our future was canceled (e.g., we received a
                # SIGINT).
                await asyncio.sleep(0)
            assert all(len(batch) == 0 for batch in batches)
        finally:
            # Release remaining requests if the generation was interrupted.
            for batch in batches:
                for request_id in batch:
                    self.release(request_id)


TextGenerationPipelineType: TypeAlias = Pipeline[
    TextGenerationInputs[T], TextGenerationOutput
]


class TextGenerationPipeline(
    TextGenerationPipelineType[T],
    GenerateMixin[T, TextGenerationRequest],
    Generic[T],
):
    """Generalized token generator pipeline."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        pipeline_model: type[PipelineModel[T]],
        # TODO: This should be removed.
        eos_token_id: int,
        weight_adapters: dict[WeightsFormat, WeightsAdapter],
        tokenizer: PipelineTokenizer[
            T,
            npt.NDArray[np.integer[Any]],
            TextGenerationRequest,
        ],
    ) -> None:
        self._pipeline_config = pipeline_config
        self._devices = load_devices(pipeline_config.model_config.device_specs)
        self._weight_adapters = weight_adapters
        self._tokenizer = tokenizer

        self.batch_info_output_fname = environ.get(
            "MAX_BATCH_INFO_FILENAME", None
        )
        self.batch_infos: list[BatchInfo] = []

        # Expand eos tokens if more are provided in pipeline_config
        if (
            "eos_token_id"
            in self._pipeline_config.model_config.huggingface_config
        ):
            eos_tokens = self._pipeline_config.model_config.huggingface_config.eos_token_id
            if isinstance(eos_tokens, int):
                if eos_tokens != eos_token_id:
                    msg = f"eos_token_id provided in huggingface config ({eos_tokens}), does not match provided eos_token_id ({eos_token_id}), using provided eos_token_id"
                    logger.warning(msg)

                self._eos_token_id = set([eos_tokens])
            elif isinstance(eos_tokens, list):
                if eos_token_id in eos_tokens:
                    self._eos_token_id = set(eos_tokens)
                else:
                    self._eos_token_id = set([eos_token_id])
            else:
                msg = f"eos_token_id in huggingface_config, is neither int or list: {eos_tokens}"
                logger.warning(msg)
                self._eos_token_id = set([eos_token_id])

        else:
            self._eos_token_id = set([eos_token_id])

        # Create a grammar compiler if constrained decoding is enabled
        self.vocab_size = None

        if pipeline_config.sampling_config.enable_structured_output:
            assert hasattr(self.tokenizer, "delegate")
            hf_tokenizer = self.tokenizer.delegate
            assert isinstance(hf_tokenizer, PreTrainedTokenizerFast)
            self.vocab_size = len(hf_tokenizer)
            self._tokenizer_info = llguidance.hf.from_tokenizer(
                hf_tokenizer, n_vocab=self.vocab_size
            )

        # Initialize Session.
        session = InferenceSession(devices=self._devices)
        self.session = session

        # Enable profiling if enabled.
        session.gpu_profiling(
            self._pipeline_config.profiling_config.gpu_profiling
        )

        # Use experimental kernels if enabled by env var `USE_EXPERIMENTAL_KERNELS`.
        session._use_experimental_kernels(
            self._pipeline_config.use_experimental_kernels
        )

        # Set PDL level if enabled by env var `PDL_LEVEL`.
        session._pdl_level(self._pipeline_config.pdl_level)

        # Load model.
        if not self._pipeline_config.model_config.quantization_encoding:
            raise ValueError("quantization_encoding must not be None")

        # Retrieve the weight id, if different than the model_path

        # TODO: These should ideally not call _weights_repo_id directly. I believe
        # huggingface_weight_repo_id property can be used here?
        weight_model_id = (
            self._pipeline_config.model_config._weights_repo_id
            if self._pipeline_config.model_config._weights_repo_id
            else self._pipeline_config.model_config.model_path
        )

        weight_paths: list[Path] = []
        if (
            self._pipeline_config.model_config.huggingface_weight_repo.repo_type
            == RepoType.online
        ):
            # Download weight files if not existent.
            weight_paths = download_weight_files(
                huggingface_model_id=weight_model_id,
                filenames=[
                    str(x)
                    for x in self._pipeline_config.model_config.weight_path
                ],
                revision=self._pipeline_config.model_config.huggingface_weight_revision,
                force_download=self._pipeline_config.model_config.force_download,
            )
        else:
            # Make sure the weight paths are absolute paths
            weight_paths = [
                self._pipeline_config.model_config.model_path / x
                for x in self._pipeline_config.model_config.weight_path
            ]

        self._pipeline_model = pipeline_model(
            pipeline_config=self._pipeline_config,
            session=session,
            huggingface_config=self._pipeline_config.model_config.huggingface_config,
            encoding=self._pipeline_config.model_config.quantization_encoding,
            devices=self._devices,
            kv_cache_config=self._pipeline_config.model_config.kv_cache_config,
            weights=load_weights(weight_paths),
            adapter=self._weight_adapters.get(
                weights_format(weight_paths), None
            ),
            return_logits=ReturnLogits.ALL
            if self._pipeline_config.enable_echo
            else ReturnLogits.LAST_TOKEN,
        )

        # Load sampler.
        self._sampler = session.load(
            token_sampler(
                self._pipeline_config.sampling_config,
                device=DeviceRef.from_device(self._devices[0]),
            )
        )

    @property
    def pipeline_config(self) -> PipelineConfig:
        return self._pipeline_config

    @property
    def tokenizer(
        self,
    ) -> PipelineTokenizer[
        T,
        npt.NDArray[np.integer[Any]],
        TextGenerationRequest,
    ]:
        return self._tokenizer

    @property
    def kv_managers(
        self,
    ) -> list[PagedKVCacheManager]:
        return [self._pipeline_model.kv_manager]

    def calculate_num_steps(
        self,
        num_steps: int,
        context: T,
    ) -> int:
        max_seq_len = self._pipeline_model.max_seq_len
        num_available_steps = context.compute_num_available_steps(max_seq_len)

        if num_available_steps <= 0:
            raise ValueError(
                f"Request {context.request_id} length ({context.current_length}) is larger than or equal to the configured max_length ({max_seq_len})"
            )

        return min(num_available_steps, num_steps)

    @traced
    def prepare_batch(
        self,
        batches: list[list[T]],
        num_steps: int,
    ) -> tuple[ModelInputs, int, npt.NDArray[np.int32] | None]:
        tracer: Tracer = Tracer("prepare_batch")
        flat_batch = [context for batch in batches for context in batch]

        if self._pipeline_config.sampling_config.enable_structured_output:
            assert self.vocab_size is not None
            bitmask = llguidance.numpy.allocate_token_bitmask(
                len(flat_batch), self.vocab_size
            )
        else:
            bitmask = None

        tracer.next("claim_cache_rows")
        for replica_idx, batch in enumerate(batches):
            for i, context in enumerate(batch):
                # Initialize a matcher if needed
                if context.json_schema and context.matcher is None:
                    if not self._pipeline_config.sampling_config.enable_structured_output:
                        raise ValueError(
                            "json_schema provided but constrained decoding is not enabled."
                        )

                    try:
                        serialized_grammar = LLMatcher.grammar_from_json_schema(
                            context.json_schema,
                            defaults={
                                "whitespace_flexible": False,
                            },
                        )
                        matcher = LLMatcher(
                            self._tokenizer_info, serialized_grammar
                        )
                        context.set_matcher(matcher)
                    except Exception as e:
                        msg = f"Json schema provided in request cannot be compiled to valid grammar. \
                        Please update your json schema to produce valid structured output. From llguidance: {e}"
                        logger.warning(msg)
                        # I am removing the json_schema, so it doesn't try to load the grammar repeatedly.
                        context.json_schema = None  # type: ignore

                if context.matcher:
                    jump_forward_tokens = context.matcher.compute_ff_tokens()
                    for token in jump_forward_tokens:
                        context.jump_ahead(token)

                # Claim cache rows for context.
                if not self._pipeline_model.kv_manager.contains(
                    context.request_id
                ):
                    self._pipeline_model.kv_manager.external_claim(
                        context.request_id, replica_idx=replica_idx
                    )

                # Update num_steps.
                num_steps = self.calculate_num_steps(num_steps, context)

                # Update bitmask
                if (
                    self._pipeline_config.sampling_config.enable_structured_output
                    and context.matcher
                    and bitmask is not None
                ):
                    llguidance.numpy.fill_next_token_bitmask(
                        context.matcher, bitmask, index=i
                    )

        # `fetch` may shorten the input context by bumping the start_idx.
        tracer.next("fetch_kv_cache")
        kv_cache_inputs = self._pipeline_model.kv_manager.fetch(
            flat_batch, num_steps
        )

        return (
            self._pipeline_model.prepare_initial_token_inputs(
                context_batch=flat_batch,
                kv_cache_inputs=KVCacheInputsSequence(
                    kv_cache_inputs=kv_cache_inputs
                ),
            ),
            num_steps,
            bitmask,
        )

    @traced
    def _maybe_sort_loras(self, batch: dict[RequestID, T]):  # noqa: ANN202
        """
        Maybe sorts the batch by LoRA Ids. Requests that use the same LoRA need
        to be adjacent to each other.
        """
        if self._pipeline_model._lora_manager is None:
            return batch

        return self._pipeline_model._lora_manager.sort_lora_batch(batch)

    def _record_batch_info(self, contexts: Iterable[T], num_steps: int) -> None:
        """
        Records batch information for the current inference step.

        Args:
            contexts (Iterable[TextGenerationContextType]): An iterable of context objects, each containing
                'start_idx' (past sequence length) and 'active_length' (current sequence length).
            num_steps (int): The number of steps processed in this batch.

        Side Effects:
            Appends a BatchInfo instance to self.batch_infos, capturing the past sequence lengths,
            current sequence lengths, and number of steps for the batch.
        """
        self.batch_infos.append(
            BatchInfo(
                past_seq_lens=[x.start_idx for x in contexts],
                seq_lens=[x.active_length for x in contexts],
                num_steps=num_steps,
            )
        )

    def __del__(self) -> None:
        if self.batch_info_output_fname is not None:
            output = {
                "batch_data": [dataclasses.asdict(x) for x in self.batch_infos]
            }
            with open(self.batch_info_output_fname, "w") as f:
                json.dump(output, f, indent=2)
                f.flush()  # Refer to MAXSERV-893

    @traced
    def execute(
        self,
        inputs: TextGenerationInputs[T],
    ) -> PipelineOutputsDict[TextGenerationOutput]:
        """Provided a batch, process batch inputs, execute the graph for num_steps in a multi-step scenario,
        then decode the tokens holistically and return the list of decoded tokens.
        """
        batches = []
        for batch in inputs.batches:
            batch = self._maybe_sort_loras(batch)
            batches.append(list(batch.values()))
            if self.batch_info_output_fname is not None:
                self._record_batch_info(batch.values(), inputs.num_steps)
        del batch  # Unbind the batch variable.

        tracer: Tracer = Tracer("compute_parameters")

        # Flatten our batch for consistent indexing.
        context_batch = [context for batch in batches for context in batch]

        # Get extra compute parameters for each input.
        batch_top_n = [context.log_probabilities for context in context_batch]
        compute_log_probabilities = any(batch_top_n)
        batch_echo: list[bool] = [
            context.log_probabilities_echo for context in context_batch
        ]

        # Prepare the batch.
        model_inputs, num_steps, bitmask = self.prepare_batch(
            batches, inputs.num_steps
        )

        # Multistep execution loop.
        tracer.next("prepare_sampling_processor")
        sampling_processor = FusedSamplingProcessor(
            sampler=self._sampler,
            pipeline_config=self._pipeline_config,
            context_batch=context_batch,
            num_steps=num_steps,
            device=self._devices[0],
            bitmask=bitmask,
            vocab_size=self.vocab_size,
        )
        batch_processors: list[BatchLogitsProcessor] = [sampling_processor]

        curr_step_inputs = model_inputs
        batch_log_probabilities = []
        tracer.next(f"multistep_execution_loop_{num_steps}_steps")
        for i in range(num_steps):
            tracer.push(f"step_{i}")

            # Execute the model and get next tokens.
            try:
                model_outputs = self._pipeline_model.execute(
                    model_inputs=curr_step_inputs
                )
            except Exception:
                batch_size = len(context_batch)
                cache_tokens = sum(ctx.start_idx for ctx in context_batch)
                input_tokens = sum(ctx.active_length for ctx in context_batch)
                logger.error(
                    "Encountered an exception while executing batch: "
                    f"{batch_size=:}, {cache_tokens=:}, {input_tokens=:}, {num_steps=:}"
                )
                raise  # re-raise the original exception

            # Sample next token.
            tracer.next("sample_next_token")
            apply_logits_processors(
                context_batch=context_batch,
                batch_logits=model_outputs.logits,
                batch_logit_offsets=model_outputs.logit_offsets,
                batch_processors=batch_processors,
            )
            new_tokens = sampling_processor.new_tokens
            assert new_tokens is not None

            if compute_log_probabilities:
                try:
                    tracer.next("compute_log_probabilities")
                    batch_log_probabilities.append(
                        self._pipeline_model.compute_log_probabilities(
                            self.session,
                            curr_step_inputs,
                            model_outputs,
                            new_tokens,
                            batch_top_n,
                            batch_echo,
                        )
                    )
                except NotImplementedError:
                    logger.warning(
                        "Unable to compute log probabilities for"
                        f" {self._pipeline_config.model_config.model_path}"
                    )
                    batch_log_probabilities.append(None)
            # Check if we're on our last iteration. If so, skip preparing the next batch
            if i == num_steps - 1:
                tracer.pop()  # pops f"step_{i}"
                break
            # Prepare inputs for the next token in multistep execution
            tracer.next("increment_cache_lengths")  # pops sample_next_token

            assert isinstance(
                curr_step_inputs.kv_cache_inputs, KVCacheInputsSequence
            ), (
                "prepare_batch instantiates and passes this as a KVCacheInputsSequence"
            )
            assert isinstance(
                curr_step_inputs.kv_cache_inputs.kv_cache_inputs, list
            ), "increment_cache_lengths instantiates and passes this as a list"
            curr_step_inputs.kv_cache_inputs.kv_cache_inputs = (
                self._pipeline_model.kv_manager.increment_cache_lengths(
                    curr_step_inputs.kv_cache_inputs.kv_cache_inputs,
                    curr_step_inputs,
                )
            )

            tracer.next("prepare_next_token_inputs")  # pops inc_cache_lengths
            curr_step_inputs = self._pipeline_model.prepare_next_token_inputs(
                new_tokens, curr_step_inputs
            )
            tracer.pop()  # pops step_{i}

        # Do the copy to host for each token generated.
        tracer.next(
            "generated_tokens.to(CPU())"
        )  # pops multistep_execution_loop_steps
        generated_tokens_host = sampling_processor.generated_tokens.to_numpy()

        # Update the context object.
        tracer.push("update_context")
        res: dict[RequestID, TextGenerationOutput] = {}
        for batch_index, context in enumerate(context_batch):
            for step in range(num_steps):
                # Convert to a Python scalar to improve serialization performance.
                next_token = int(generated_tokens_host[batch_index, step])

                # Get log probs if needed.
                log_probs: LogProbabilities | None = None
                if compute_log_probabilities and (
                    log_probs_for_step := batch_log_probabilities[step]
                ):
                    log_probs = log_probs_for_step[batch_index]

                context.update(
                    new_token=next_token, log_probabilities=log_probs
                )
                if context.is_done:
                    break

            res[context.request_id] = context.to_generation_output()

        # Update the cache lengths in our kv_cache manager.
        # This should be done after the contexts are updated.
        tracer.next("kv_manager.step")  # pops prepare_response
        self._pipeline_model.kv_manager.step(context_batch)
        tracer.pop()  # pops kv_manager.step

        return res

    def release(self, request_id: RequestID) -> None:
        """Mark the context as complete, releasing the cache slot from the KV manager."""
        self._pipeline_model.kv_manager.release(request_id)
