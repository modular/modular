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
"""MAX pipeline model base classes for model execution."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Generic

from max.driver import (
    Buffer,
    Device,
    is_virtual_device_mode,
)
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Value
from max.graph.weights import Weights, WeightsAdapter
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheInputsInterface,
    KVCacheParamInterface,
    PagedCacheValues,
)
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.context import BaseContextType, LogProbabilities
from max.pipelines.kv_cache.config import KVCacheConfig
from max.pipelines.lora import LoRAInputs, LoRAManager
from max.pipelines.modeling.config_enums import supported_encoding_dtype
from transformers import AutoConfig

if TYPE_CHECKING:
    from max.pipelines.lib.config import PipelineConfig

    from .batch_processor import BatchProcessor

logger = logging.getLogger("max.pipelines")


class AlwaysSignalBuffersMixin:
    """Mixin for models that always require signal buffers.

    Use this for models that use VocabParallelEmbedding or other distributed
    components that always perform allreduce, even on single-device setups.

    Models using this mixin build graphs that always include signal buffer
    inputs, regardless of device count. This is typically because they use
    distributed embedding layers or other components that call allreduce
    operations unconditionally.
    """

    devices: list[Device]
    """Device list that must be provided by the model class."""

    @cached_property
    def signal_buffers(self) -> list[Buffer]:
        """Override to always create signal buffers.

        Models using this mixin have distributed components that always
        perform allreduce, even for single-device setups. Therefore,
        signal buffers are always required to match the graph inputs.

        In compile-only mode (virtual device mode), returns an empty list
        to avoid GPU memory allocation which is not supported.

        Returns:
            List of signal buffer tensors, one per device, or empty list
            in compile-only mode.
        """
        # In compile-only mode (virtual device mode), skip signal buffer
        # allocation since VirtualDevice does not support memory allocation.
        # Signal buffers are only needed during model execution, not compilation.
        if is_virtual_device_mode():
            return []

        # Import here to avoid circular dependency
        from max.nn.comm import Signals

        # Signals.allocate initializes the signal buffers and enables p2p access
        return Signals.allocate(self.devices)


@dataclass
class ModelOutputs:
    """Pipeline model outputs.

    Shape conventions below are for text-generation pipelines:

    - ``B``: batch size
    - ``V``: vocabulary size
    - ``H``: hidden-state width
    - ``T``: number of returned logit rows (depends on return mode)

    The shape depends on the value of the :class:`ReturnLogits` and :class:`ReturnHiddenStates`
    enums. Unless we are running with spec decoding, we use ``ReturnLogits.LAST_TOKEN``
    and ``ReturnHiddenStates.NONE``.
    """

    logits: Buffer
    """Primary logits buffer.

    For text generation this has shape ``[T, V]`` where:
    - last-token mode: ``T = B`` (default)
    - all-token mode: ``T = total_input_tokens``
    - variable mode: ``T = logit_offsets[-1]`` (typically ``B * return_n_logits``)
    """

    next_token_logits: Buffer | None = None
    """Next-token logits for text generation, shape ``[B, V]`` when present."""

    logit_offsets: Buffer | None = None
    """Cumulative row offsets into ``logits`` for text generation.

    Shape is ``[B + 1]``. Per-sequence logits are:
    ``logits[logit_offsets[i]:logit_offsets[i + 1], :]``.
    """

    hidden_states: Buffer | None = None
    """Optional hidden states for text generation.

    Single-device shape is ``[T_h, H]`` where:
    - none mode: NONE (default)
    - last-token mode: ``T_h = B``
    - all-token mode: ``T_h = total_input_tokens``

    For data parallel models, the hs will be on the first gpu since it is replicated.
    """


@dataclass(kw_only=True)
class ModelInputs:
    """Base class for model inputs.

    Use this class to encapsulate inputs for your model; you may store any
    number of dataclass fields.

    The following example demonstrates how to create a custom inputs class:

    .. code-block:: python

        @dataclass
        class ReplitInputs(ModelInputs):
            tokens: Buffer
            input_row_offsets: Buffer

        # Create tensors
        tokens = Buffer.zeros((1, 2, 3), DType.int64)
        input_row_offsets = Buffer.zeros((1, 1, 1), DType.int64)

        # Initialize inputs
        inputs = ReplitInputs(tokens=tokens, input_row_offsets=input_row_offsets)

        # Access tensors
        list(inputs) == [tokens, input_row_offsets]  # Output: True
    """

    kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None
    """KV cache graph inputs holding every (DP replica x TP shard) device's
    inputs: a ``KVCacheInputs`` leaf, or a ``MultiKVCacheInputs`` tree for
    multi-cache models. ``flatten()`` yields the full positional input list."""

    lora: LoRAInputs | None = None
    """Per-batch LoRA adapter buffers, or ``None`` when LoRA is disabled."""

    hidden_states: Buffer | list[Buffer] | None = None
    """Hidden states for a variable number of tokens per sequence.

    For data parallel models, this can be a list of Buffers where each Buffer
    contains hidden states for the sequences assigned to that device.
    """

    def update(self, **kwargs) -> None:
        """Updates attributes from keyword arguments (only existing, non-None)."""
        key: str
        value: Any
        for key, value in kwargs.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        """Returns positional Buffer inputs for model ABI calls."""
        raise NotImplementedError(
            f"{type(self).__name__} does not define model ABI buffers."
        )


@dataclass(kw_only=True)
class UnifiedEagleOutputs(ModelOutputs):
    """Outputs from a unified EAGLE graph execution."""

    num_accepted_draft_tokens: Buffer
    next_tokens: Buffer
    next_draft_tokens: Buffer

    # HACK: These are required to inherit from ModelOutputs but are unused
    # for UnifiedEagleOutputs!
    logits: Buffer | None = None  # type: ignore[assignment]
    next_token_logits: None = None
    logit_offsets: None = None
    hidden_states: None = None


@dataclass(kw_only=True)
class UnifiedSpecDecodeInputs(ModelInputs):
    """Shared spec-decode fields + buffer-tail packing for unified ``*Inputs``.

    Each arch composes the tail via :meth:`_spec_decode_tail_buffers`, which
    mirrors ``build_spec_decode_input_types`` and must stay in lockstep with it.
    """

    draft_tokens: Buffer | None = None
    seed: Buffer | None = None
    temperature: Buffer | None = None
    top_k: Buffer | None = None
    max_k: Buffer | None = None
    top_p: Buffer | None = None
    min_top_p: Buffer | None = None
    in_thinking_phase: Buffer | None = None
    pinned_bitmask: Buffer | None = None
    wait_payload: Buffer | None = None
    device_bitmask_scratch: Buffer | None = None

    structured_output: bool = False
    """Whether this graph was compiled with constrained-decoding bitmask
    inputs. Mirrors ``pipeline_config.needs_bitmask_constraints`` -- the same
    value that gates the bitmask triple in ``build_spec_decode_input_types`` --
    so the buffer tail and the graph signature derive the decision from one
    place. Set by each capable module's ``prepare_initial_token_inputs``."""

    def _spec_decode_tail_buffers(
        self,
        *,
        include_in_thinking_phase: bool,
        supports_structured_output: bool = True,
    ) -> tuple[Buffer, ...]:
        # draft_tokens, seed, and the five sampling params are unconditional in
        # build_spec_decode_input_types; assert them so a missing one is a loud
        # error, not a silently shortened ABI tuple. (Draft KV lives in the
        # {"target", "draft"} tree, packed by super().buffers.)
        assert self.draft_tokens is not None
        tail: tuple[Buffer, ...] = (self.draft_tokens,)
        assert self.seed is not None
        tail += (self.seed,)
        assert self.temperature is not None
        assert self.top_k is not None
        assert self.max_k is not None
        assert self.top_p is not None
        assert self.min_top_p is not None
        tail += (
            self.temperature,
            self.top_k,
            self.max_k,
            self.top_p,
            self.min_top_p,
        )
        if include_in_thinking_phase:
            assert self.in_thinking_phase is not None
            tail += (self.in_thinking_phase,)
        # Gate the bitmask triple on two compile-time flags, not a runtime
        # pinned_bitmask is not None check: supports_structured_output
        # is False for dflash (sets pinned_bitmask but declares no bitmask graph
        # inputs); structured_output mirrors needs_bitmask_constraints.
        if supports_structured_output and self.structured_output:
            assert self.pinned_bitmask is not None
            assert self.wait_payload is not None
            assert self.device_bitmask_scratch is not None
            tail += (
                self.pinned_bitmask,
                self.wait_payload,
                self.device_bitmask_scratch,
            )
        return tail


class PipelineModel(ABC, Generic[BaseContextType]):
    """A pipeline model with setup, input preparation and execution methods."""

    #: Optional batch processor class for input/output handling.
    batch_processor_cls: ClassVar[type[BatchProcessor[Any, Any]] | None] = None
    #: Config class used to delegate ``calculate_max_seq_len`` and KV params.
    model_config_cls: ClassVar[type[Any] | None] = None

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None,
        return_logits: ReturnLogits,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
        max_batch_size: int = 1,
    ) -> None:
        self.pipeline_config = pipeline_config
        self.max_batch_size = max_batch_size
        self.devices = devices
        self.device_refs = [DeviceRef.from_device(d) for d in devices]
        self.kv_cache_config = kv_cache_config
        self.weights = weights
        self.adapter = adapter
        self.return_logits = return_logits
        self.return_hidden_states = return_hidden_states

        # Initialize `max_seq_len` here to avoid repeated HF config access.
        self.max_seq_len = self.calculate_max_seq_len(
            pipeline_config, self.huggingface_config
        )

        self._lora_manager: LoRAManager | None = (
            LoRAManager(
                pipeline_config.lora,
                pipeline_config.model.model_name,
                self.dtype,
                self.huggingface_config.num_attention_heads,
                self.huggingface_config.num_key_value_heads,
                self.huggingface_config.head_dim,
                self.max_seq_len * max_batch_size,
            )
            if pipeline_config.lora
            else None
        )

        self._batch_processor: BatchProcessor[Any, Any] | None = None
        batch_processor_cls = type(self).batch_processor_cls
        if batch_processor_cls is not None:
            from .batch_processor import BatchProcessorRuntime

            model_config_cls = getattr(type(self), "model_config_cls", None)
            if model_config_cls is None:
                raise ValueError(
                    f"{type(self).__qualname__} sets batch_processor_cls but "
                    "does not define model_config_cls."
                )
            arch_config = model_config_cls.initialize(pipeline_config)
            pad_token_id = getattr(self.huggingface_config, "pad_token_id", 0)
            self._batch_processor = batch_processor_cls(
                arch_config,
                BatchProcessorRuntime(
                    pipeline_config=pipeline_config,
                    devices=devices,
                    return_logits=return_logits,
                    return_hidden_states=return_hidden_states,
                    signal_buffers=self.signal_buffers,
                    lora_manager=self._lora_manager,
                    pad_token_id=pad_token_id or 0,
                    max_batch_size=self.max_batch_size,
                ),
            )

    @property
    def batch_processor(self) -> BatchProcessor[Any, Any] | None:
        """Returns the batch processor when configured."""
        return self._batch_processor

    @property
    def huggingface_config(self) -> AutoConfig:
        """Returns the HuggingFace config from pipeline config.

        For multimodal models (e.g., Pixtral, Gemma3 multimodal), this
        returns the top-level config which contains both text_config and
        vision_config. Models should explicitly access .text_config or
        .vision_config as needed.

        Returns:
            The HuggingFace AutoConfig for this model.

        Raises:
            ValueError: If HuggingFace config could not be loaded.
        """
        config = self.pipeline_config.model.huggingface_config
        if config is None:
            raise ValueError(
                f"HuggingFace config is required but could not be loaded for "
                f"model '{self.pipeline_config.model.model_path}'. "
                "Ensure the model repository contains a valid config.json."
            )
        return config

    @property
    def lora_manager(self) -> LoRAManager | None:
        """Returns the LoRA manager if LoRA is enabled, otherwise None."""
        return self._lora_manager

    @cached_property
    def signal_buffers(self) -> list[Buffer]:
        """Lazily initialize signal buffers for multi-GPU communication collectives.

        Signal buffers are only needed during model execution, not during compilation.
        By deferring their allocation, we avoid memory allocation in compile-only mode.

        Returns:
            List of signal buffer tensors, one per device for multi-device setups,
            or an empty list for single-device setups or compile-only mode.
        """
        # In compile-only mode (virtual device mode), skip signal buffer
        # allocation since VirtualDevice does not support memory allocation.
        if is_virtual_device_mode():
            return []

        if len(self.devices) <= 1:
            return []

        # Import here to avoid circular dependency
        from max.nn.comm import Signals

        # Signals.allocate initializes the signal buffers and enables p2p access
        return Signals.allocate(self.devices)

    @property
    def dtype(self) -> DType:
        """Returns the model data type from pipeline config."""
        quantization_encoding = self.pipeline_config.model.quantization_encoding
        if quantization_encoding is None:
            raise ValueError("quantization_encoding must not be None")
        return supported_encoding_dtype(quantization_encoding)

    @property
    def sampler_custom_extensions(self) -> Sequence[Path]:
        """Custom-op extension paths to compile the sampler graph with."""
        return ()

    @classmethod
    def _calculate_max_seq_len_from_config(
        cls,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
    ) -> int:
        """Delegates to ``model_config_cls.calculate_max_seq_len`` or ``initialize().get_max_seq_len()``."""
        model_config_cls = cls.model_config_cls
        if model_config_cls is None:
            raise NotImplementedError(
                f"{cls.__qualname__} must set `model_config_cls` "
                "or override `calculate_max_seq_len()`."
            )
        calculate = getattr(model_config_cls, "calculate_max_seq_len", None)
        if calculate is not None:
            return calculate(pipeline_config, huggingface_config)
        return model_config_cls.initialize(pipeline_config).get_max_seq_len()

    @classmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        """Calculates the optimal max sequence length for the model.

        Default implementation delegates to ``model_config_cls``. Override when
        pipeline-model semantics differ from the config (for example, bounding
        ``max_length`` where the config is permissive).

        Args:
            pipeline_config: Configuration for the pipeline.
            huggingface_config: Hugging Face model configuration.

        Returns:
            int: The maximum sequence length to use.
        """
        return cls._calculate_max_seq_len_from_config(
            pipeline_config, huggingface_config
        )

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

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[BaseContextType]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> ModelInputs:
        """Prepares the initial inputs to be passed to ``execute()``.

        The inputs and functionality can vary per model. For example, model
        inputs could include encoded tensors, unique IDs per tensor when using
        a KV cache manager, and ``kv_cache_inputs`` (or None if the model does
        not use KV cache). This method typically batches encoded tensors,
        claims a KV cache slot if needed, and returns the inputs and caches.

        When :attr:`batch_processor_cls` is set, delegates to the batch processor.
        """
        if self._batch_processor is not None:
            return self._batch_processor.prepare_initial_token_inputs(
                replica_batches,
                kv_cache_inputs=kv_cache_inputs,
                return_n_logits=return_n_logits,
            )
        return self._prepare_initial_token_inputs(
            replica_batches, kv_cache_inputs, return_n_logits
        )

    def _prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[BaseContextType]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> ModelInputs:
        raise NotImplementedError(
            f"{type(self).__qualname__} must implement prepare_initial_token_inputs "
            "or set batch_processor_cls."
        )

    def compute_log_probabilities(
        self,
        session: InferenceSession,
        model_inputs: ModelInputs,
        model_outputs: ModelOutputs,
        next_tokens: Buffer,
        batch_top_n: list[int],
        batch_echo: list[bool],
    ) -> list[LogProbabilities | None]:
        """Optional method that can be overridden to compute log probabilities.

        Args:
            session: Inference session to compute log probabilities within.
            model_inputs: Inputs to the model returned by
                ``prepare_initial_token_inputs()``.
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


class PipelineModelWithKVCache(PipelineModel[BaseContextType]):
    """A pipeline model that supports KV cache."""

    kv_params: KVCacheParamInterface

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None,
        return_logits: ReturnLogits,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
        max_batch_size: int = 1,
    ) -> None:
        super().__init__(
            pipeline_config=pipeline_config,
            session=session,
            devices=devices,
            kv_cache_config=kv_cache_config,
            weights=weights,
            adapter=adapter,
            return_logits=return_logits,
            return_hidden_states=return_hidden_states,
            max_batch_size=max_batch_size,
        )
        self.kv_params = type(self).get_kv_params(
            huggingface_config=self.huggingface_config,
            pipeline_config=self.pipeline_config,
            devices=self.device_refs,
            kv_cache_config=self.kv_cache_config,
            cache_dtype=self.pipeline_config.model.kv_cache.cache_dtype,
        )

    def _unflatten_kv_inputs(
        self, kv_inputs_flat: Sequence[Value[Any]]
    ) -> list[PagedCacheValues]:
        # This helper supports single-cache (leaf) models; multi-cache trees
        # are unflattened by the architecture itself.
        kv_inputs = self.kv_params.unflatten_kv_inputs(iter(kv_inputs_flat))
        assert isinstance(kv_inputs, KVCacheInputs)
        return list(kv_inputs.inputs)

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParamInterface:
        """Returns the KV cache params for the pipeline model.

        Delegates to ``model_config_cls.construct_kv_params(...)``.
        Subclasses with custom KV behavior should override this method.
        """
        model_config_cls = cls.model_config_cls
        if model_config_cls is None:
            raise NotImplementedError(
                f"{cls.__qualname__} must set `model_config_cls` "
                "or override `get_kv_params()`."
            )
        return model_config_cls.construct_kv_params(
            huggingface_config,
            pipeline_config,
            devices,
            kv_cache_config,
            cache_dtype,
        )
