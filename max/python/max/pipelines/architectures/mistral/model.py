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
from typing import Any

import numpy as np
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, DeviceRef, Graph, TensorType, Value
from max.graph.weights import (
    SafetensorWeights,
    WeightData,
    Weights,
    WeightsAdapter,
)
from max.nn.legacy.comm import Signals
from max.nn.legacy.kv_cache import (
    KVCacheInputs,
    KVCacheParams,
    PagedCacheValues,
    uses_opaque,
)
from max.nn.legacy.layer import Module
from max.nn.legacy.transformer import ReturnLogits
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    CompilationTimer,
    KVCacheConfig,
    KVCacheMixin,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
    upper_bounded_default,
)
from max.profiler import traced
from transformers import AutoConfig

from .distributed_mistral import DistributedMistral
from .mistral import Mistral
from .model_config import MistralConfig

logger = logging.getLogger("max.pipelines")


@dataclass
class MistralInputs(ModelInputs):
    """A class representing inputs for the Mistral model.

    This class encapsulates the input tensors required for the Mistral model execution:
    - input_tokens: A tensor containing the input token IDs
    - input_row_offsets: A tensor containing the offsets for each row in the ragged input sequence
    - return_n_logits: A tensor containing the number of expected token logits.
    """

    input_tokens: Buffer
    input_row_offsets: Buffer
    signal_buffers: list[Buffer]
    """Device buffers used for synchronization in communication collectives."""
    return_n_logits: Buffer


class MistralModel(PipelineModel[TextContext], KVCacheMixin):
    model: Model
    """Compiled and initialized model ready for inference."""

    signal_buffers: list[Buffer]
    """Device buffers used for synchronization in communication collectives."""

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

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Runs the graph."""
        assert isinstance(model_inputs, MistralInputs)

        curr_kv_cache_inputs = model_inputs.kv_cache_inputs or ()

        model_outputs = self.model.execute(
            model_inputs.input_tokens,
            model_inputs.input_row_offsets,
            model_inputs.return_n_logits,
            *model_inputs.signal_buffers,
            *curr_kv_cache_inputs,
        )
        if len(model_outputs) == 3:
            assert isinstance(model_outputs[0], Buffer)
            assert isinstance(model_outputs[1], Buffer)
            assert isinstance(model_outputs[2], Buffer)
            return ModelOutputs(
                next_token_logits=model_outputs[0],
                logits=model_outputs[1],
                logit_offsets=model_outputs[2],
            )
        else:
            assert isinstance(model_outputs[0], Buffer)
            return ModelOutputs(
                next_token_logits=model_outputs[0],
                logits=model_outputs[0],
            )

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> MistralInputs:
        if len(replica_batches) > 1:
            raise ValueError("Model does not support DP>1")

        context_batch = replica_batches[0]

        if not uses_opaque(self.kv_cache_config.cache_strategy):
            # TODO(MODELS-407): Consider deleting the padded path entirely.
            raise ValueError("Mistral unsupported for padded token batches")

        # Get input_row_offsets: start and end position of each batch in the
        # combined total_seq_len dimension.
        input_row_offsets = Buffer.from_numpy(
            np.cumsum(
                [0] + [ctx.tokens.active_length for ctx in context_batch],
                dtype=np.uint32,
            )
        ).to(self.devices[0])

        # Create a ragged token vector of length: sum(len(t) for t in tokens).
        next_tokens_batch = Buffer.from_numpy(
            np.concatenate([ctx.tokens.active for ctx in context_batch])
        ).to(self.devices[0])

        return MistralInputs(
            input_tokens=next_tokens_batch,
            input_row_offsets=input_row_offsets,
            signal_buffers=self.signal_buffers,
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            kv_cache_inputs=kv_cache_inputs,
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Buffer,
        prev_model_inputs: ModelInputs,
    ) -> MistralInputs:
        assert isinstance(prev_model_inputs, MistralInputs)

        if not uses_opaque(self.kv_cache_config.cache_strategy):
            # TODO(MODELS-407): Consider deleting the padded path entirely.
            raise ValueError("multistep unsupported for padded token batches")

        row_offsets_size = prev_model_inputs.input_row_offsets.shape[0]
        next_row_offsets = self._input_row_offsets_prealloc[:row_offsets_size]

        return MistralInputs(
            input_tokens=next_tokens,
            input_row_offsets=next_row_offsets,
            signal_buffers=self.signal_buffers,
            return_n_logits=prev_model_inputs.return_n_logits,
            kv_cache_inputs=prev_model_inputs.kv_cache_inputs,
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
        return MistralConfig.construct_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

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
                "Unable to infer max_length for Mistral, the provided "
                f"max_length ({pipeline_config.model.max_length}) exceeds the "
                f"model's max_position_embeddings "
                f"({huggingface_config.max_position_embeddings})."
            ) from e

    def _get_state_dict(
        self,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
    ) -> dict[str, WeightData]:
        text_config = getattr(
            self.huggingface_config, "text_config", self.huggingface_config
        )
        if self.adapter:
            state_dict = self.adapter(
                dict(self.weights.items()),
                huggingface_config=text_config,
                pipeline_config=self.pipeline_config,
            )
        else:
            state_dict = {
                key: value.data() for key, value in self.weights.items()
            }
        return state_dict

    def graph_inputs(self) -> tuple[TensorType | BufferType, ...]:
        # Generate DeviceRef
        device_ref = DeviceRef.from_device(self.devices[0])

        # Construct general input types
        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )

        kv_inputs = self.kv_params.get_symbolic_inputs().flatten()

        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        )

        if len(self.devices) > 1:
            # Flatten kv types for each device
            signals = Signals(
                devices=(DeviceRef(d.label, d.id) for d in self.devices)
            )
            return (
                tokens_type,
                input_row_offsets_type,
                return_n_logits_type,
                *signals.input_types(),
                *kv_inputs,
            )
        else:
            return (
                tokens_type,
                input_row_offsets_type,
                return_n_logits_type,
                *kv_inputs,
            )

    def _unflatten_kv_inputs(
        self, kv_inputs_flat: Sequence[Value[Any]]
    ) -> list[PagedCacheValues]:
        fetch_types = self.kv_params.get_symbolic_inputs()[0]
        len_of_kv_tuple_per_dev = len(list(fetch_types))
        kv_caches_per_dev: list[PagedCacheValues] = []
        for i in range(self.kv_params.n_devices):
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

    @traced
    def _build_graph(
        self, weights: Weights, adapter: WeightsAdapter | None = None
    ) -> Graph:
        # Retrieve config
        state_dict = self._get_state_dict(weights, adapter)

        model_config = MistralConfig.initialize_from_config(
            self.pipeline_config,
            getattr(
                self.huggingface_config, "text_config", self.huggingface_config
            ),
        )
        model_config.return_logits = self.return_logits

        # Get Graph Inputs
        graph_inputs = self.graph_inputs()

        # Build Graph
        nn_model: Module
        if len(self.devices) > 1:
            nn_model = DistributedMistral(model_config)
            nn_model.load_state_dict(
                state_dict,
                weight_alignment=1,
                strict=False,  # TODO(MODELS-551) vision tower weights not used
            )
            self.state_dict = nn_model.state_dict()

            with Graph("mistral", input_types=[*graph_inputs]) as graph:
                tokens, input_row_offsets, return_n_logits, *variadic_args = (
                    graph.inputs
                )

                # Multi-GPU passes a signal buffer per device: unmarshal these.
                signal_buffers = [
                    v.buffer for v in variadic_args[: len(self.devices)]
                ]

                # Unmarshal the remaining arguments, which are for KV cache.
                kv_caches_per_dev = self._unflatten_kv_inputs(
                    variadic_args[len(self.devices) :]
                )

                outputs = nn_model(
                    tokens.tensor,
                    signal_buffers,
                    kv_caches_per_dev,
                    return_n_logits.tensor,
                    input_row_offsets.tensor,
                )

                graph.output(*outputs)
                return graph

        else:
            nn_model = Mistral(model_config)
            nn_model.load_state_dict(
                state_dict,
                weight_alignment=1,
                strict=False,  # TODO(MODELS-551) vision tower weights not used
            )
            self.state_dict = nn_model.state_dict()

            with Graph("mistral", input_types=graph_inputs) as graph:
                tokens, input_row_offsets, return_n_logits, *kv_cache_inputs = (
                    graph.inputs
                )
                kv_collection = PagedCacheValues(
                    kv_blocks=kv_cache_inputs[0].buffer,
                    cache_lengths=kv_cache_inputs[1].tensor,
                    lookup_table=kv_cache_inputs[2].tensor,
                    max_lengths=kv_cache_inputs[3].tensor,
                )
                outputs = nn_model(
                    tokens.tensor,
                    kv_collection,
                    return_n_logits.tensor,
                    input_row_offsets.tensor,
                )
                graph.output(*outputs)
                return graph

    @traced
    def load_model(
        self,
        session: InferenceSession,
    ) -> Model:
        if self.pipeline_config.enable_echo:
            raise ValueError(
                "Mistral model does not currently implement enable echo."
            )

        # Pre-allocate a buffer for input_row_offsets in multistep execution.
        # We do this to avoid materializing and copying a buffer with each multistep step
        assert self.pipeline_config.max_batch_size, (
            "Expected max_batch_size to be set"
        )
        self._input_row_offsets_prealloc = Buffer.from_numpy(
            np.arange(self.pipeline_config.max_batch_size + 1, dtype=np.uint32)
        ).to(self.devices[0])

        if not isinstance(self.weights, SafetensorWeights):
            raise ValueError(
                "only safetensors weights are currently supported in Mistral models."
            )

        timer = CompilationTimer("model")
        graph = self._build_graph(self.weights, self.adapter)
        timer.mark_build_complete()
        model = session.load(graph, weights_registry=self.state_dict)
        timer.done()

        return model
