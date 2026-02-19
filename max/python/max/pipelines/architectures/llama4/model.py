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

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, Value
from max.graph.weights import Weights, WeightsAdapter
from max.nn.legacy.comm import Signals
from max.nn.legacy.kv_cache import (
    KVCacheInputs,
    KVCacheInputsSequence,
    KVCacheParams,
    PagedCacheValues,
    RaggedKVCacheInputs,
)
from max.nn.legacy.transformer import ReturnLogits
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    AlwaysSignalBuffersMixin,
    CompilationTimer,
    KVCacheConfig,
    KVCacheMixin,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
)
from transformers import AutoConfig

from .llama4 import Llama4
from .model_config import Llama4Config

logger = logging.getLogger("max.pipelines")


@dataclass
class Llama4Inputs(ModelInputs):
    """A class representing inputs for the Llama4 model.

    This class encapsulates the input tensors required for the Llama4 model
    execution.
    """

    tokens: npt.NDArray[np.integer[Any]] | Buffer
    """Tensor containing the input token IDs."""

    input_row_offsets: npt.NDArray[np.integer[Any]] | Buffer
    """Tensor containing the offsets for each row in the ragged input sequence,
    or the attention mask for the padded input sequence."""

    cache_positions: npt.NDArray[np.integer[Any]] | Buffer
    """Positions in the cache of each input token."""

    signal_buffers: list[Buffer]
    """Device buffers used for synchronization in communication collectives."""


class Llama4Model(
    AlwaysSignalBuffersMixin, PipelineModel[TextContext], KVCacheMixin
):
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
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits,
        )

        self.model = self.load_model(session)

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
        max_seq_len = pipeline_config.model.max_length
        if max_seq_len:
            return max_seq_len

        # Access max_position_embeddings from the text_config
        return huggingface_config.text_config.max_position_embeddings

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        """Gets the parameters required to configure the KV cache for Llama 4.

        Delegates to the :obj:`Llama4Config.construct_kv_params` static method.

        Args:
            huggingface_config: The HuggingFace model configuration object
                (:obj:`transformers.AutoConfig`).
            pipeline_config: The MAX Engine pipeline configuration.
            devices: The list of devices the model will run on.
            kv_cache_config: The MAX Engine KV cache configuration settings
                (:obj:`max.pipelines.max_config.KVCacheConfig`).
            cache_dtype: The desired data type for the KV cache
                (:obj:`max.dtype.DType`).

        Returns:
            The configured :obj:`max.pipelines.kv_cache.KVCacheParams` object.
        """
        return Llama4Config.construct_kv_params(
            huggingface_config,
            pipeline_config,
            devices,
            kv_cache_config,
            cache_dtype,
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
        self._input_row_offsets_prealloc = Buffer.from_numpy(
            np.arange(self.pipeline_config.max_batch_size + 1, dtype=np.uint32)
        ).to(self.devices[0])

        timer = CompilationTimer("model")
        graph = self._build_graph()
        timer.mark_build_complete()
        model = session.load(graph, weights_registry=self.state_dict)
        timer.done()

        return model

    def _build_graph(self):  # noqa: ANN202
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
        model_config = Llama4Config.initialize(self.pipeline_config)
        model_config.finalize(
            huggingface_config=huggingface_config,
            state_dict=state_dict,
            return_logits=self.return_logits,
        )
        nn_model = Llama4(model_config)
        nn_model.load_state_dict(
            state_dict,
            weight_alignment=1,
            strict=False,  # We do not use vision weights
        )
        self.state_dict = nn_model.state_dict()

        signals = Signals(
            devices=(DeviceRef(d.label, d.id) for d in self.devices)
        )
        kv_cache_args = self.kv_params.get_symbolic_inputs()
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
            kv_collection = self._unflatten_kv_inputs(
                variadic_args[len(self.devices) :]
            )

            outputs = nn_model(
                tokens.tensor,
                cache_positions.tensor,
                signal_buffers,
                kv_collection,
                input_row_offsets=input_row_offsets,
            )
            graph.output(*outputs)
        return graph

    def _unflatten_kv_inputs(
        self, kv_inputs_flat: Sequence[Value[Any]]
    ) -> list[PagedCacheValues]:
        n_devices = self.kv_params.n_devices
        fetch_types = self.kv_params.get_symbolic_inputs()[0]
        len_of_kv_tuple_per_dev = len(list(fetch_types))
        kv_caches_per_dev: list[PagedCacheValues] = []
        for i in range(n_devices):
            start_idx = i * len_of_kv_tuple_per_dev
            kv_caches_per_dev.append(
                PagedCacheValues(
                    kv_blocks=kv_inputs_flat[start_idx].buffer,
                    cache_lengths=kv_inputs_flat[start_idx + 1].tensor,
                    lookup_table=kv_inputs_flat[start_idx + 2].tensor,
                    max_lengths=kv_inputs_flat[start_idx + 3].tensor,
                )
            )
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
            assert isinstance(model_outputs[0], Buffer)
            assert isinstance(model_outputs[1], Buffer)
            assert isinstance(model_outputs[2], Buffer)
            return ModelOutputs(
                logits=model_outputs[1],
                next_token_logits=model_outputs[0],
                logit_offsets=model_outputs[2],
            )
        else:
            assert isinstance(model_outputs[0], Buffer)
            return ModelOutputs(
                logits=model_outputs[0],
                next_token_logits=model_outputs[0],
            )

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> ModelInputs:
        """Prepares the initial inputs for the first execution pass of the Llama 4 model.

        Note:
            This method currently returns dummy inputs and needs to be implemented
            with the actual input preparation logic for Llama 4, considering
            batching, ragged tensors, and KV cache integration.

        Args:
            replica_batches: A sequence of sequences of :obj:`TextContext` objects representing
                the input prompts for each replica.
            kv_cache_inputs: Optional inputs required by the KV cache manager.

        Returns:
            The prepared :obj:`ModelInputs` object for the initial execution step.
        """
        if len(replica_batches) > 1:
            raise ValueError("Model does not support DP>1")

        context_batch = replica_batches[0]

        assert kv_cache_inputs is not None
        kv_cache_inputs = cast(KVCacheInputsSequence, kv_cache_inputs)

        # This needs to be replaced with actual input preparation
        # Get input_row_offsets: start and end position of each batch in the
        # combined total_seq_len dimension.
        input_row_offsets = np.cumsum(
            [0] + [ctx.tokens.active_length for ctx in context_batch],
            dtype=np.uint32,
        )

        # Create a ragged token vector of length: sum(len(t) for t in tokens).
        tokens = np.concatenate([ctx.tokens.active for ctx in context_batch])

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
                    cache_length + len(ctx.tokens.active),
                    dtype=np.uint32,
                )
            )

        return Llama4Inputs(
            tokens=Buffer.from_numpy(tokens).to(self.devices[0]),
            input_row_offsets=Buffer.from_numpy(input_row_offsets).to(
                self.devices[0]
            ),
            cache_positions=Buffer.from_numpy(
                np.concatenate(cache_positions)
            ).to(self.devices[0]),
            signal_buffers=self.signal_buffers,
            kv_cache_inputs=kv_cache_inputs,
        )

    def prepare_next_token_inputs(
        self, next_tokens: Buffer, prev_model_inputs: ModelInputs
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
