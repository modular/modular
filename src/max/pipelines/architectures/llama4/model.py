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

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import cast

import numpy as np
from max.driver import Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.graph.weights import Weights, WeightsAdapter
from max.nn import ReturnLogits, Signals
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheInputsSequence,
    KVCacheManager,
    KVCacheParams,
    RaggedKVCacheInputs,
    estimate_kv_cache_size,
    load_kv_manager,
)
from max.pipelines import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
    SupportedEncoding,
)
from max.pipelines.core import TextContext
from max.pipelines.pipeline import KVCacheMixin
from transformers import AutoConfig

from .llama4 import Llama4
from .model_config import Llama4Config

logger = logging.getLogger("max.pipelines")


class Llama4Inputs(ModelInputs):
    """A class representing inputs for the Llama4 model.

    This class encapsulates the input tensors required for the Llama4 model
    execution.
    """

    tokens: np.ndarray | Tensor
    """Tensor containing the input token IDs."""

    input_row_offsets: np.ndarray | Tensor
    """Tensor containing the offsets for each row in the ragged input sequence,
    or the attention mask for the padded input sequence."""

    cache_positions: np.ndarray | Tensor
    """Positions in the cache of each input token."""

    signal_buffers: list[Tensor]
    """Device buffers used for synchronization in communication collectives."""

    def __init__(
        self,
        tokens: np.ndarray | Tensor,
        input_row_offsets: np.ndarray | Tensor,
        cache_positions: np.ndarray | Tensor,
        signal_buffers: list[Tensor],
        kv_cache_inputs: KVCacheInputs | None = None,
    ) -> None:
        """
        Args:
            tokens: Input token IDs.
            input_row_offsets: Input row offsets (ragged tensors).
            cache_positions: Positions in the cache of each input token.
            signal_buffers: Device buffers used for synchronization in
                communication collectives.
        """
        self.tokens = tokens
        self.input_row_offsets = input_row_offsets
        self.cache_positions = cache_positions
        self.signal_buffers = signal_buffers
        self.kv_cache_inputs = kv_cache_inputs


class Llama4Model(PipelineModel[TextContext], KVCacheMixin):
    """A Llama 4 pipeline model for text generation.

    This class integrates the Llama 4 architecture with the MAX Engine pipeline
    infrastructure, handling model loading, KV cache management, and input preparation
    for inference.
    """

    model: Model
    """The compiled and initialized MAX Engine model ready for inference."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        huggingface_config: AutoConfig,
        encoding: SupportedEncoding,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
    ) -> None:
        """
        Args:
            pipeline_config: The configuration settings for the entire pipeline.
            session: The MAX Engine inference session managing the runtime.
            huggingface_config: The configuration loaded from HuggingFace
                (:obj:`transformers.AutoConfig`).
            encoding: The quantization and data type encoding used for the model
                (:obj:`max.pipelines.config_enums.SupportedEncoding`).
            devices: A list of MAX Engine devices (:obj:`max.driver.Device`) to
                run the model on.
            kv_cache_config: Configuration settings for the Key-Value cache
                (:obj:`max.pipelines.max_config.KVCacheConfig`).
            weights: The model weights (:obj:`max.graph.weights.Weights`).
            adapter: An optional adapter to modify weights before loading
                (:obj:`max.graph.weights.WeightsAdapter`).
            return_logits: The number of top logits to return from the model
                execution.
        """
        super().__init__(
            pipeline_config,
            session,
            huggingface_config,
            encoding,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits,
        )

        self.model = self.load_model(session)

        # Contents of signal buffer should be filled with zeros.
        self.signal_buffers = [
            Tensor.zeros(
                shape=(Signals.NUM_BYTES,),
                dtype=DType.uint8,
                device=dev,
            )
            for dev in self.devices
        ]

    @staticmethod
    def calculate_max_seq_len(
        pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        """Calculates the maximum sequence length for the Llama 4 model.

        Uses the `max_length` from the :obj:`max.pipelines.config.PipelineConfig`
        if provided, otherwise falls back to the `max_position_embeddings` from
        the HuggingFace configuration's text config.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.
            huggingface_config: The HuggingFace model configuration object
                (:obj:`transformers.AutoConfig`).

        Returns:
            The calculated maximum sequence length.
        """
        max_seq_len = pipeline_config.max_length
        if max_seq_len:
            return max_seq_len

        # Access max_position_embeddings from the text_config
        return huggingface_config.text_config.max_position_embeddings

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        """Gets the parameters required to configure the KV cache for Llama 4.

        Delegates to the :obj:`Llama4Config.get_kv_params` static method.

        Args:
            huggingface_config: The HuggingFace model configuration object
                (:obj:`transformers.AutoConfig`).
            n_devices: The number of devices the model will run on.
            kv_cache_config: The MAX Engine KV cache configuration settings
                (:obj:`max.pipelines.max_config.KVCacheConfig`).
            cache_dtype: The desired data type for the KV cache
                (:obj:`max.dtype.DType`).

        Returns:
            The configured :obj:`max.pipelines.kv_cache.KVCacheParams` object.
        """
        return Llama4Config.get_kv_params(
            huggingface_config,
            n_devices,
            kv_cache_config,
            cache_dtype,
        )

    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        """Gets the number of hidden layers from the HuggingFace configuration.

        Delegates to the :obj:`Llama4Config.get_num_layers` static method.

        Args:
            huggingface_config: The HuggingFace model configuration object
                (:obj:`transformers.AutoConfig`).

        Returns:
            The number of hidden layers.
        """
        return Llama4Config.get_num_layers(huggingface_config)

    @classmethod
    def estimate_kv_cache_size(
        cls,
        pipeline_config: PipelineConfig,
        available_cache_memory: int,
        devices: list[Device],
        huggingface_config: AutoConfig,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> int:
        """Estimates the size of the KV cache required for the Llama 4 model in bytes.

        Args:
            pipeline_config: The configuration for the pipeline.
            available_cache_memory: The total memory available for the KV cache
                in bytes.
            huggingface_config: The HuggingFace model configuration object
                (:obj:`transformers.AutoConfig`).
            devices: A list of MAX Engine devices (:obj:`max.driver.Device`) the
                model will run on.
            kv_cache_config: Configuration settings for the KV cache
                (:obj:`max.pipelines.max_config.KVCacheConfig`).
            cache_dtype: The data type for the KV cache (:obj:`max.dtype.DType`).

        Returns:
            The estimated size of the KV cache in bytes.
        """
        return estimate_kv_cache_size(
            params=Llama4Config.get_kv_params(
                huggingface_config=huggingface_config,
                n_devices=len(devices),
                kv_cache_config=kv_cache_config,
                cache_dtype=cache_dtype,
            ),
            max_batch_size=pipeline_config.max_batch_size,
            max_seq_len=cls.calculate_max_seq_len(
                pipeline_config,
                huggingface_config=huggingface_config,
            ),
            num_layers=Llama4Config.get_num_layers(
                huggingface_config=huggingface_config,
            ),
            available_cache_memory=available_cache_memory,
            devices=devices,
        )

    def load_model(self, session: InferenceSession) -> Model:
        """Loads the compiled Llama 4 model into the MAX Engine session.

        Note:
            This method currently returns a dummy Model object and needs to be
            implemented with the actual graph building and model loading logic
            for Llama 4.

        Args:
            session: The MAX Engine inference session.

        Returns:
            The loaded MAX Engine model object.
        """
        assert self.pipeline_config.max_batch_size, (
            "Expected max_batch_size to be set"
        )
        self._input_row_offsets_prealloc = Tensor.from_numpy(
            np.arange(self.pipeline_config.max_batch_size + 1, dtype=np.uint32)
        ).to(self.devices[0])

        logger.info("Building and compiling model...")
        before = time.perf_counter()
        graph = self._build_graph()
        model = session.load(graph, weights_registry=self.state_dict)
        after = time.perf_counter()
        logger.info(
            f"Building and compiling model took {after - before:.6f} seconds"
        )
        return model

    def _build_graph(self):
        device0 = self.devices[0]
        device_ref = DeviceRef(device0.label, device0.id)
        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        # NOTE: input_row_offsets_len should be batch_size + 1.
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        )
        cache_positions_type = TensorType(
            DType.uint32, shape=["total_seq_len"], device=device_ref
        )

        huggingface_config = self.huggingface_config
        if self.adapter:
            state_dict = self.adapter(
                dict(self.weights.items()),
                huggingface_config=huggingface_config,
                pipeline_config=self.pipeline_config,
            )
        else:
            state_dict = {
                key: value.data() for key, value in self.weights.items()
            }
        model_config = Llama4Config.generate(
            pipeline_config=self.pipeline_config,
            huggingface_config=huggingface_config,
            state_dict=state_dict,
            dtype=self.dtype,
            n_devices=len(self.devices),
            logits_postprocessor=None,
            attention_bias=huggingface_config.text_config.attention_bias,
            cache_dtype=self.encoding.cache_dtype,
            kv_cache_config=self.kv_cache_config,
            return_logits=self.return_logits,
        )
        nn_model = Llama4(model_config)
        nn_model.load_state_dict(state_dict, weight_alignment=1)
        self.state_dict = nn_model.state_dict()

        signals = Signals(
            devices=(DeviceRef(d.label, d.id) for d in self.devices)
        )
        kv_cache_args = self.kv_manager.input_symbols()
        flattened_kv_types = [
            kv_type for sublist in kv_cache_args for kv_type in sublist
        ]
        with Graph(
            getattr(self.huggingface_config, "model_type", "Llama4"),
            input_types=[
                tokens_type,
                input_row_offsets_type,
                cache_positions_type,
                *signals.input_types(),
                *flattened_kv_types,
            ],
        ) as graph:
            tokens, input_row_offsets, cache_positions, *variadic_args = (
                graph.inputs
            )

            # Multi-GPU passes a signal buffer per device: unmarshal those.
            signal_buffers = [
                v.buffer for v in variadic_args[: len(self.devices)]
            ]

            # Unmarshal the remaining arguments, which are for KV cache.
            kv_cache = [v.tensor for v in variadic_args[len(self.devices) :]]

            kv_caches_per_dev = self._unflatten_kv_inputs(kv_cache)

            outputs = nn_model(
                tokens.tensor,
                cache_positions.tensor,
                signal_buffers,
                kv_caches_per_dev,
                input_row_offsets=input_row_offsets,
            )
            graph.output(*outputs)
        return graph

    def _unflatten_kv_inputs(
        self, kv_inputs_flat: Sequence[TensorValue]
    ) -> list[tuple[TensorValue, ...]]:
        kv_params = Llama4Config.get_kv_params(
            huggingface_config=self.huggingface_config,
            n_devices=len(self.devices),
            kv_cache_config=self.kv_cache_config,
            cache_dtype=self.encoding.cache_dtype,
        )
        n_devices = kv_params.n_devices
        fetch_types = self.kv_manager.input_symbols()[0]
        len_of_kv_tuple_per_dev = len(list(fetch_types))
        kv_caches_per_dev = [
            tuple(
                kv_inputs_flat[
                    i * len_of_kv_tuple_per_dev : (i + 1)
                    * len_of_kv_tuple_per_dev
                ]
            )
            for i in range(n_devices)
        ]
        return kv_caches_per_dev

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Executes the Llama 4 model with the prepared inputs.

        Note:
            This method currently returns dummy output and needs to be implemented
            with the actual model execution logic for Llama 4.

        Args:
            model_inputs: The prepared inputs for the model execution, typically including
                token IDs, attention masks/offsets, and KV cache inputs.

        Returns:
            An object containing the output logits from the model execution.
        """
        model_inputs = cast(Llama4Inputs, model_inputs)
        curr_kv_cache_inputs = model_inputs.kv_cache_inputs or ()
        model_outputs = self.model.execute(
            model_inputs.tokens,
            model_inputs.input_row_offsets,
            model_inputs.cache_positions,
            *model_inputs.signal_buffers,
            *curr_kv_cache_inputs,
        )
        if len(model_outputs) == 3:
            return ModelOutputs(
                logits=cast(Tensor, model_outputs[1]),
                next_token_logits=cast(Tensor, model_outputs[0]),
                logit_offsets=cast(Tensor, model_outputs[2]),
            )
        else:
            return ModelOutputs(
                logits=cast(Tensor, model_outputs[0]),
                next_token_logits=cast(Tensor, model_outputs[0]),
            )

    def prepare_initial_token_inputs(
        self,
        context_batch: Sequence[TextContext],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> ModelInputs:
        """Prepares the initial inputs for the first execution pass of the Llama 4 model.

        Note:
            This method currently returns dummy inputs and needs to be implemented
            with the actual input preparation logic for Llama 4, considering
            batching, ragged tensors, and KV cache integration.

        Args:
            context_batch: A sequence of :obj:`TextContext` objects representing
                the input prompts.
            kv_cache_inputs: Optional inputs required by the KV cache manager.

        Returns:
            The prepared :obj:`ModelInputs` object for the initial execution step.
        """
        assert kv_cache_inputs is not None
        kv_cache_inputs = cast(KVCacheInputsSequence, kv_cache_inputs)

        # This needs to be replaced with actual input preparation
        # Get input_row_offsets: start and end position of each batch in the
        # combined total_seq_len dimension.
        input_row_offsets = np.cumsum(
            [0] + [ctx.active_length for ctx in context_batch],
            dtype=np.uint32,
        )

        # Create a ragged token vector of length: sum(len(t) for t in tokens).
        tokens = np.concatenate([ctx.next_tokens for ctx in context_batch])

        # Create cache positions for each token.
        cache_positions = []
        ragged_kv_cache_inputs = cast(
            RaggedKVCacheInputs, kv_cache_inputs.kv_cache_inputs[0]
        )
        for n, ctx in enumerate(context_batch):
            cache_length = ragged_kv_cache_inputs.cache_lengths.to_numpy()[n]
            cache_positions.append(
                np.arange(
                    cache_length,
                    cache_length + len(ctx.next_tokens),
                    dtype=np.uint32,
                )
            )

        return Llama4Inputs(
            tokens=Tensor.from_numpy(tokens).to(self.devices[0]),
            input_row_offsets=Tensor.from_numpy(input_row_offsets).to(
                self.devices[0]
            ),
            cache_positions=Tensor.from_numpy(
                np.concatenate(cache_positions)
            ).to(self.devices[0]),
            signal_buffers=self.signal_buffers,
            kv_cache_inputs=kv_cache_inputs,
        )

    def prepare_next_token_inputs(
        self, next_tokens: Tensor, prev_model_inputs: ModelInputs
    ) -> ModelInputs:
        """Prepares the inputs for subsequent execution steps in a multi-step generation.

        Note:
            This method currently returns dummy inputs and needs to be implemented
            with the actual input preparation logic for subsequent steps in Llama 4,
            efficiently handling the next token and KV cache updates.

        Args:
            next_tokens: The tensor containing the token IDs generated in the previous step.
            prev_model_inputs: The :obj:`ModelInputs` used in the previous execution step.

        Returns:
            The prepared :obj:`ModelInputs` object for the next execution step.
        """
        prev_model_inputs = cast(Llama4Inputs, prev_model_inputs)
        row_offsets_size = prev_model_inputs.input_row_offsets.shape[0]
        next_row_offsets = self._input_row_offsets_prealloc[:row_offsets_size]

        # Create cache positions for each token.
        kv_cache_inputs = cast(
            KVCacheInputsSequence, prev_model_inputs.kv_cache_inputs
        )
        ragged_kv_cache_inputs = cast(
            RaggedKVCacheInputs, kv_cache_inputs.kv_cache_inputs[0]
        )
        return Llama4Inputs(
            tokens=next_tokens,
            input_row_offsets=next_row_offsets,
            cache_positions=ragged_kv_cache_inputs.cache_lengths,
            signal_buffers=self.signal_buffers,
            kv_cache_inputs=prev_model_inputs.kv_cache_inputs,
        )

    def load_kv_manager(
        self, session: InferenceSession, available_cache_memory: int | None
    ) -> KVCacheManager:
        """Loads and initializes the KVCacheManager for the Llama 4 model.

        Configures the KV cache manager based on model parameters, pipeline settings,
        and available memory.

        Args:
            session: The MAX Engine inference session.
            available_cache_memory: The amount of memory available for the KV cache in bytes.

        Returns:
            An initialized :obj:`KVCacheManager` instance.
        """
        return load_kv_manager(
            params=Llama4Config.get_kv_params(
                huggingface_config=self.huggingface_config,
                n_devices=len(self.devices),
                kv_cache_config=self.kv_cache_config,
                cache_dtype=self.encoding.cache_dtype,
            ),
            max_batch_size=self.pipeline_config.max_batch_size,
            max_seq_len=self.calculate_max_seq_len(
                self.pipeline_config, huggingface_config=self.huggingface_config
            ),
            num_layers=Llama4Config.get_num_layers(
                huggingface_config=self.huggingface_config
            ),
            devices=self.devices,
            available_cache_memory=available_cache_memory,
            page_size=self.kv_cache_config.kv_cache_page_size,
            session=session,
        )
