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
from typing import TYPE_CHECKING, Any, Generic

from max.driver import Buffer, Device, is_virtual_device_mode
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.graph.weights import Weights, WeightsAdapter
from max.interfaces import BaseContextType, LogProbabilities
from max.nn.legacy.kv_cache import KVCacheInputs
from max.nn.legacy.transformer import ReturnHiddenStates, ReturnLogits
from transformers import AutoConfig

from ..config_enums import SupportedEncoding
from ..kv_cache_config import KVCacheConfig
from ..lora import LoRAManager
from .kv_cache import KVCacheMixin

if TYPE_CHECKING:
    from ..config import PipelineConfig

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

        from max.nn.legacy.comm import Signals

        return [
            Buffer.zeros(
                shape=(Signals.NUM_BYTES,),
                dtype=DType.uint8,
                device=dev,
            )
            for dev in self.devices
        ]


@dataclass
class ModelOutputs:
    """Pipeline model outputs.

    Shape conventions below are for text-generation pipelines:

    - ``B``: batch size
    - ``V``: vocabulary size
    - ``H``: hidden-state width
    - ``T``: number of returned logit rows (depends on return mode)

    The shape depends on the value of the `ReturnLogits` and `ReturnHiddenStates`
    enums. Unless we are running with spec decoding, we use `ReturnLogits.LAST_TOKEN`
    and `ReturnHiddenStates.NONE`.
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

    hidden_states: Buffer | list[Buffer] | None = None
    """Optional hidden states for text generation.

    Single-device shape is ``[T_h, H]`` where:
    - none mode: NONE (default)
    - last-token mode: ``T_h = B``
    - all-token mode: ``T_h = total_input_tokens``

    For data parallel models, this can be a list of Buffers where each Buffer
    has shape ``[T_h_device, H]`` for the sequences assigned to that device.
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

    kv_cache_inputs: KVCacheInputs | None = None

    lora_ids: Buffer | None = None
    """Buffer containing the LoRA ids."""

    lora_ranks: Buffer | None = None
    """Buffer containing the LoRA ranks"""

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


class PipelineModel(ABC, Generic[BaseContextType]):
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
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
    ) -> None:
        self.pipeline_config = pipeline_config
        self.huggingface_config = huggingface_config
        self.encoding = encoding
        self.devices = devices
        self.device_refs = [DeviceRef.from_device(d) for d in devices]
        self.kv_cache_config = kv_cache_config
        self.weights = weights
        self.adapter = adapter
        self.return_logits = return_logits
        self.return_hidden_states = return_hidden_states

        # Initialize `max_seq_len` here to avoid repeated HF config access.
        self.max_seq_len = self.calculate_max_seq_len(
            pipeline_config, huggingface_config
        )

        if isinstance(self, KVCacheMixin):
            self.kv_params = self.get_kv_params(
                huggingface_config=huggingface_config,
                pipeline_config=pipeline_config,
                devices=self.device_refs,
                kv_cache_config=kv_cache_config,
                cache_dtype=pipeline_config.model.kv_cache.cache_dtype,
            )
            assert self.kv_cache_config._available_cache_memory is not None, (
                "Available cache memory should have been set during memory estimation"
            )
            assert pipeline_config.max_batch_size is not None, (
                "max_batch_size should have been set during memory estimation"
            )
            self.kv_managers = self.load_kv_managers(
                kv_params=self.kv_params,
                max_batch_size=pipeline_config.max_batch_size,
                max_seq_len=self.max_seq_len,
                session=session,
                available_cache_memory=self.kv_cache_config._available_cache_memory,
            )
            self.kv_manager = self.kv_managers[0]

        self._lora_manager: LoRAManager | None = (
            LoRAManager(
                pipeline_config.lora,
                pipeline_config.model.model_name,
                self.dtype,
                huggingface_config.num_attention_heads,
                huggingface_config.num_key_value_heads,
                huggingface_config.head_dim,
                pipeline_config.zmq_endpoint_base,
            )
            if pipeline_config.lora
            else None
        )

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

        # Import here to avoid circular dependency
        from max.nn.legacy.comm import Signals

        # Initialize state needed for communication collectives.
        # Contents of signal buffer should be filled with zeros.
        return (
            [
                Buffer.zeros(
                    shape=(Signals.NUM_BYTES,),
                    dtype=DType.uint8,
                    device=dev,
                )
                for dev in self.devices
            ]
            if len(self.devices) > 1
            # Skip creating buffers for single-device, where communication
            # collectives shouldn't be called.
            else []
        )

    @property
    def dtype(self) -> DType:
        """Returns the model data type (from encoding or pipeline config)."""
        # AudioGeneratorPipeline passes Nones for all args except pipeline config
        return (
            self.encoding.dtype
            if self.encoding is not None
            else self.pipeline_config.model.quantization_encoding.dtype
        )

    @classmethod
    @abstractmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        """Calculates the optimal max sequence length for the model.

        Models are expected to implement this method. The following example
        shows how to implement it for a Mistral model:

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
    def estimate_weights_size(cls, pipeline_config: PipelineConfig) -> int:
        """Calculates the estimated memory consumption of our model."""
        # TODO move this logic to the PipelineModel instead of PipelineConfig class.
        # Better yet, make this more accurate by loading and measuring memory consumption
        # after we load the model
        return pipeline_config.model.weights_size()

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
            huggingface_config: Hugging Face model configuration

        Returns:
            Estimated activation memory in bytes
        """
        del pipeline_config, huggingface_config  # Unused.
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
        replica_batches: Sequence[Sequence[BaseContextType]],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> ModelInputs:
        """Prepares the initial inputs to be passed to ``.execute()``.

        The inputs and functionality can vary per model. For example, model
        inputs could include encoded tensors, unique IDs per tensor when using
        a KV cache manager, and ``kv_cache_inputs`` (or None if the model does
        not use KV cache). This method typically batches encoded tensors,
        claims a KV cache slot if needed, and returns the inputs and caches.
        """
        ...

    @abstractmethod
    def prepare_next_token_inputs(
        self,
        next_tokens: Buffer,
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
        next_tokens: Buffer,
        batch_top_n: list[int],
        batch_echo: list[bool],
    ) -> list[LogProbabilities | None]:
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
