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
"""MAX pipeline model base classes for model execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Generic

from max.driver import Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph.weights import (
    Weights,
    WeightsAdapter,
)
from max.interfaces import (
    BaseContextType,
    LogProbabilities,
)
from max.kv_cache import infer_optimal_batch_size
from max.nn.kv_cache import KVCacheInputs
from max.nn.transformer import ReturnLogits
from transformers import AutoConfig

if TYPE_CHECKING:
    from ..config import PipelineConfig

from ..config_enums import SupportedEncoding
from ..kv_cache_config import KVCacheConfig
from ..lora import LoRAManager
from .kv_cache import KVCacheMixin


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
    def signal_buffers(self) -> list[Tensor]:
        """Override to always create signal buffers.

        Models using this mixin have distributed components that always
        perform allreduce, even for single-device setups. Therefore,
        signal buffers are always required to match the graph inputs.

        Returns:
            List of signal buffer tensors, one per device.
        """
        from max.nn import Signals

        return [
            Tensor.zeros(
                shape=(Signals.NUM_BYTES,),
                dtype=DType.uint8,
                device=dev,
            )
            for dev in self.devices
        ]


@dataclass(frozen=True)
class ModelOutputs:
    logits: Tensor
    """Logits for a variable number of tokens per sequence."""

    next_token_logits: Tensor | None = None
    """Logits for just the next token."""

    logit_offsets: Tensor | None = None
    """Offsets to access variable length logits for each sequence."""


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

    @cached_property
    def signal_buffers(self) -> list[Tensor]:
        """Lazily initialize signal buffers for multi-GPU communication collectives.

        Signal buffers are only needed during model execution, not during compilation.
        By deferring their allocation, we avoid memory allocation in compile-only mode.

        Returns:
            List of signal buffer tensors, one per device for multi-device setups,
            or an empty list for single-device setups.
        """
        # Import here to avoid circular dependency
        from max.nn import Signals

        # Initialize state needed for communication collectives.
        # Contents of signal buffer should be filled with zeros.
        return (
            [
                Tensor.zeros(
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
        context_batch: Sequence[BaseContextType],
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
