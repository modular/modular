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
"""Implements the DeepseekV2 nn.model."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from max.driver import Buffer, Device, DeviceSpec
from max.dtype import DType
from max.engine.api import InferenceSession, Model
from max.graph import BufferType, DeviceRef, Graph, TensorType, Value
from max.graph.weights import SafetensorWeights, Weights, WeightsAdapter
from max.interfaces import LogProbabilities
from max.nn.comm import Signals
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheParams,
    PagedCacheValues,
)
from max.nn.layer import Module
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    CompilationTimer,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModelWithKVCache,
    upper_bounded_default,
)
from max.pipelines.lib.log_probabilities import (
    compute_log_probabilities_ragged,
    log_probabilities_ragged_graph,
)
from transformers import AutoConfig

from .deepseekV2 import DeepseekV2
from .distributed_deepseekV2 import DistributedDeepseekV2
from .model_config import DeepseekV2Config

logger = logging.getLogger("max.pipelines")


@dataclass
class DeepseekV2Inputs(ModelInputs):
    """A class representing inputs for the DeepseekV2 model.

    This class encapsulates the input tensors required for the DeepseekV2 model execution:
    - tokens: A tensor containing the input token IDs
    - input_row_offsets: A tensor containing the offsets for each row in the ragged input sequence
    - return_n_logits: A tensor containing the number of logits to return
    """

    tokens: Buffer
    input_row_offsets: Buffer
    signal_buffers: list[Buffer]
    """Device buffers used for synchronization in communication collectives."""

    return_n_logits: Buffer = field(kw_only=True)


class DeepseekV2Model(PipelineModelWithKVCache[TextContext]):
    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.ALL,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
    ) -> None:
        if pipeline_config.model.device_specs[0] == DeviceSpec.cpu():
            raise ValueError("DeepseekV2 currently only supported on gpu.")

        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits,
            return_hidden_states,
        )

        self.model = self.load_model(session)
        self.logprobs_device = devices[0]
        self.logprobs_model = self.load_logprobs_model(session)

    def execute(
        self,
        model_inputs: ModelInputs,
    ) -> ModelOutputs:
        assert isinstance(model_inputs, DeepseekV2Inputs)

        curr_kv_cache_inputs = model_inputs.kv_cache_inputs or ()
        model_outputs = self.model.execute(
            model_inputs.tokens,
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
    ) -> DeepseekV2Inputs:
        if len(replica_batches) > 1:
            raise ValueError("Model does not support DP>1")

        context_batch = replica_batches[0]
        # Get input_row_offsets: start and end position of each batch in the
        # combined total_seq_len dimension.
        input_row_offsets = np.cumsum(
            [0] + [ctx.tokens.active_length for ctx in context_batch],
            dtype=np.uint32,
        )

        # Create a ragged token vector of length: sum(len(t) for t in tokens).
        tokens = np.concatenate([ctx.tokens.active for ctx in context_batch])

        return DeepseekV2Inputs(
            tokens=Buffer.from_numpy(tokens).to(self.devices[0]),
            input_row_offsets=Buffer.from_numpy(input_row_offsets).to(
                self.devices[0]
            ),
            signal_buffers=self.signal_buffers,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ).to(self.devices[0]),
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Buffer,
        prev_model_inputs: ModelInputs,
    ) -> DeepseekV2Inputs:
        assert isinstance(prev_model_inputs, DeepseekV2Inputs)
        row_offsets_size = prev_model_inputs.input_row_offsets.shape[0]
        next_row_offsets = self._input_row_offsets_prealloc[:row_offsets_size]
        return DeepseekV2Inputs(
            tokens=next_tokens,
            input_row_offsets=next_row_offsets,
            signal_buffers=self.signal_buffers,
            kv_cache_inputs=prev_model_inputs.kv_cache_inputs,
            return_n_logits=prev_model_inputs.return_n_logits,
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
        return DeepseekV2Config.construct_kv_params(
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
                "Unable to infer max_length for DeepseekV2, the provided "
                f"max_length ({pipeline_config.model.max_length}) exceeds the "
                f"model's max_seq_len "
                f"({huggingface_config.max_position_embeddings})."
            ) from e

    def graph_inputs(self) -> tuple[TensorType | BufferType, ...]:
        # Generate DeviceRef
        device_ref = DeviceRef.from_device(self.devices[0])

        # Construct general input types
        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=device_ref
        )

        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        )

        if len(self.devices) > 1:
            signals = Signals(
                devices=(DeviceRef(d.label, d.id) for d in self.devices)
            )
            return (
                tokens_type,
                input_row_offsets_type,
                return_n_logits_type,
                *signals.input_types(),
                *self.kv_params.get_symbolic_inputs().flatten(),
            )
        else:
            return (
                tokens_type,
                input_row_offsets_type,
                return_n_logits_type,
                *self.kv_params.get_symbolic_inputs().flatten(),
            )

    def _unflatten_kv_inputs(
        self, kv_inputs_flat: Sequence[Value[Any]]
    ) -> list[PagedCacheValues]:
        kv_params = self.get_kv_params(
            huggingface_config=self.huggingface_config,
            pipeline_config=self.pipeline_config,
            devices=[DeviceRef.from_device(d) for d in self.devices],
            kv_cache_config=self.kv_cache_config,
            cache_dtype=self.pipeline_config.model.kv_cache.cache_dtype,
        )
        n_devices = kv_params.n_devices
        fetch_types = kv_params.get_symbolic_inputs()[0]
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

    def _build_graph(self) -> Graph:
        # Pre-allocate a buffer for input_row_offsets in multistep execution.
        # We do this to avoid materializing and copying a buffer with each multistep step
        max_batch_size = self.pipeline_config.max_batch_size
        assert max_batch_size, "Expected max_batch_size to be set"

        self._input_row_offsets_prealloc = Buffer.from_numpy(
            np.arange(max_batch_size + 1, dtype=np.uint32)
        ).to(self.devices[0])

        # Read in weights.
        if not isinstance(self.weights, SafetensorWeights):
            raise ValueError(
                "only safetensors weights supported in DeepseekV2."
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

        model_config = DeepseekV2Config.initialize(self.pipeline_config)
        model_config.max_batch_context_length = (
            self.pipeline_config.max_batch_total_tokens
            or model_config.max_batch_context_length
        )

        # Get Graph Inputs
        graph_inputs = self.graph_inputs()

        # Build Graph
        nn_model: Module
        if len(self.devices) > 1:
            nn_model = DistributedDeepseekV2(model_config)
            nn_model.load_state_dict(
                state_dict, weight_alignment=1, strict=False
            )
            self.state_dict = nn_model.state_dict()

            with Graph("deepseekV2", input_types=[*graph_inputs]) as graph:
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
            nn_model = DeepseekV2(model_config)
            nn_model.load_state_dict(state_dict, weight_alignment=1)
            self.state_dict = nn_model.state_dict()

            with Graph("deepseekV2", input_types=[*graph_inputs]) as graph:
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

    def load_model(
        self,
        session: InferenceSession,
    ) -> Model:
        timer = CompilationTimer("model")
        graph = self._build_graph()
        timer.mark_build_complete()
        model = session.load(graph, weights_registry=self.state_dict)
        timer.done()

        return model

    def load_logprobs_model(self, session: InferenceSession) -> Model:
        # TODO: Perhaps 'levels' ought to be configurable.
        graph = log_probabilities_ragged_graph(
            DeviceRef.from_device(self.logprobs_device), levels=3
        )
        return session.load(graph)

    def compute_log_probabilities(
        self,
        session: InferenceSession,
        model_inputs: ModelInputs,
        model_outputs: ModelOutputs,
        next_tokens: Buffer,
        batch_top_n: list[int],
        batch_echo: list[bool],
    ) -> list[LogProbabilities | None]:
        assert model_outputs.next_token_logits is not None
        next_token_logits = model_outputs.next_token_logits
        sampled_tokens = next_tokens.to_numpy()

        model_inputs = cast(DeepseekV2Inputs, model_inputs)
        tokens = model_inputs.tokens.to_numpy()
        input_row_offsets = model_inputs.input_row_offsets.to_numpy()

        # Determine if we have full logits for all tokens or only last-token logits.
        # Full logits are only available when return_logits is ALL or VARIABLE.
        has_full_logits = self.return_logits in (
            ReturnLogits.ALL,
            ReturnLogits.VARIABLE,
        )

        # If echo is requested but we don't have full logits, raise an error.
        if any(batch_echo) and not has_full_logits:
            raise ValueError(
                "Log probabilities with echo=true requires enable_echo=true "
                "in the pipeline configuration to return logits for all tokens."
            )

        # Pass logits=None when we only have last-token logits.
        # compute_log_probabilities_ragged will use next_token_logits instead.
        logits = model_outputs.logits if has_full_logits else None

        return compute_log_probabilities_ragged(
            self.logprobs_device,
            self.logprobs_model,
            input_row_offsets=input_row_offsets,
            logits=logits,
            next_token_logits=next_token_logits,
            tokens=tokens,
            sampled_tokens=sampled_tokens,
            batch_top_n=batch_top_n,
            batch_echo=batch_echo,
        )
