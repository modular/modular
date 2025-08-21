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
import math
import time
from collections.abc import Sequence
from typing import Optional

import numpy as np
from max.driver import Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.graph.weights import (
    SafetensorWeights,
    WeightData,
    Weights,
    WeightsAdapter,
)
from max.nn import Module, ReturnLogits, Signals
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheManager,
    KVCacheParams,
    estimate_kv_cache_size,
    load_kv_manager,
)
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
    SupportedEncoding,
    upper_bounded_default,
)
from max.profiler import traced
from transformers import AutoConfig

from .distributed_mistral import DistributedMistral
from .mistral import Mistral
from .model_config import MistralConfig

logger = logging.getLogger("max.pipelines")


class MistralInputs(ModelInputs):
    """A class representing inputs for the Mistral model.

    This class encapsulates the input tensors required for the Mistral model execution:
    - input_tokens: A tensor containing the input token IDs
    - input_row_offsets: A tensor containing the offsets for each row in the ragged input sequence
    - return_n_logits: A tensor containing the number of expected token logits.
    """

    input_tokens: Tensor
    input_row_offsets: Tensor
    signal_buffers: list[Tensor]
    """Device buffers used for synchronization in communication collectives."""
    return_n_logits: Tensor

    def __init__(
        self,
        input_tokens: Tensor,
        input_row_offsets: Tensor,
        signal_buffers: list[Tensor],
        return_n_logits: Tensor,
        kv_cache_inputs: KVCacheInputs | None = None,
    ) -> None:
        self.input_tokens = input_tokens
        self.input_row_offsets = input_row_offsets
        self.signal_buffers = signal_buffers
        self.return_n_logits = return_n_logits
        self.kv_cache_inputs = kv_cache_inputs


class MistralModel(PipelineModel[TextContext]):
    model: Model
    """Compiled and initialized model ready for inference."""

    signal_buffers: list[Tensor]
    """Device buffers used for synchronization in communication collectives."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        huggingface_config: AutoConfig,
        encoding: SupportedEncoding,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: Optional[WeightsAdapter] = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
    ) -> None:
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

        # Initialize state needed for communication collectives.
        # Contents of signal buffer should be filled with zeros.
        self.signal_buffers = (
            [
                Tensor.zeros(
                    shape=(Signals.NUM_BYTES,), dtype=DType.uint8, device=dev
                )
                for dev in self.devices
            ]
            if len(self.devices) > 1
            # Skip creating buffers for single-device, where communication
            # collectives shouldn't be called.
            else []
        )

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
            assert isinstance(model_outputs[0], Tensor)
            assert isinstance(model_outputs[1], Tensor)
            assert isinstance(model_outputs[2], Tensor)
            return ModelOutputs(
                next_token_logits=model_outputs[0],
                logits=model_outputs[1],
                logit_offsets=model_outputs[2],
            )
        else:
            assert isinstance(model_outputs[0], Tensor)
            return ModelOutputs(
                next_token_logits=model_outputs[0],
                logits=model_outputs[0],
            )

    def prepare_initial_token_inputs(
        self,
        context_batch: Sequence[TextContext],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> MistralInputs:
        if not self.kv_cache_config.cache_strategy.uses_opaque():
            # TODO(MODELS-407): Consider deleting the padded path entirely.
            msg = "Mistral unsupported for padded token batches"
            raise ValueError(msg)

        # Get input_row_offsets: start and end position of each batch in the
        # combined total_seq_len dimension.
        input_row_offsets = Tensor.from_numpy(
            np.cumsum(
                [0] + [ctx.active_length for ctx in context_batch],
                dtype=np.uint32,
            )
        ).to(self.devices[0])

        # Create a ragged token vector of length: sum(len(t) for t in tokens).
        next_tokens_batch = np.concatenate(
            [ctx.next_tokens for ctx in context_batch]
        )
        next_tokens_batch = Tensor.from_numpy(next_tokens_batch).to(
            self.devices[0]
        )

        return MistralInputs(
            input_tokens=next_tokens_batch,
            input_row_offsets=input_row_offsets,
            signal_buffers=self.signal_buffers,
            return_n_logits=Tensor.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            kv_cache_inputs=kv_cache_inputs,
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: ModelInputs,
    ) -> MistralInputs:
        assert isinstance(prev_model_inputs, MistralInputs)

        if not self.kv_cache_config.cache_strategy.uses_opaque():
            # TODO(MODELS-407): Consider deleting the padded path entirely.
            msg = "multistep unsupported for padded token batches"
            raise ValueError(msg)

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
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return MistralConfig.get_kv_params(
            huggingface_config=huggingface_config,
            n_devices=n_devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        return MistralConfig.get_num_layers(huggingface_config)

    @classmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        try:
            return upper_bounded_default(
                upper_bound=huggingface_config.max_position_embeddings,
                default=pipeline_config.max_length,
            )
        except ValueError as e:
            msg = (
                "Unable to infer max_length for Mistral, the provided "
                f"max_length ({pipeline_config.max_length}) exceeds the "
                f"model's max_position_embeddings "
                f"({huggingface_config.max_position_embeddings})."
            )
            raise ValueError(msg) from e

    def load_kv_manager(
        self,
        session: InferenceSession,
        available_cache_memory: int,
    ) -> KVCacheManager:
        assert self.devices, "devices must be provided to load kv manager."
        return load_kv_manager(
            params=MistralConfig.get_kv_params(
                huggingface_config=self.huggingface_config,
                n_devices=len(self.devices),
                kv_cache_config=self.kv_cache_config,
                cache_dtype=self.encoding.cache_dtype,
            ),
            max_batch_size=self.pipeline_config.max_batch_size,
            max_seq_len=self.calculate_max_seq_len(
                self.pipeline_config, huggingface_config=self.huggingface_config
            ),
            num_layers=MistralConfig.get_num_layers(
                huggingface_config=self.huggingface_config
            ),
            devices=self.devices,
            available_cache_memory=available_cache_memory,
            page_size=self.kv_cache_config.kv_cache_page_size,
            session=session,
        )

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
        """Estimates the size of the kv cache in bytes."""
        assert devices, "devices must be provided to estimate kv cache size."
        return estimate_kv_cache_size(
            params=MistralConfig.get_kv_params(
                huggingface_config=huggingface_config,
                n_devices=len(devices),
                kv_cache_config=kv_cache_config,
                cache_dtype=cache_dtype,
            ),
            max_batch_size=pipeline_config.max_batch_size,
            max_seq_len=cls.calculate_max_seq_len(
                pipeline_config, huggingface_config=huggingface_config
            ),
            num_layers=MistralConfig.get_num_layers(huggingface_config),
            available_cache_memory=available_cache_memory,
            devices=devices,
        )

    def _get_state_dict(
        self,
        weights: Weights,
        adapter: Optional[WeightsAdapter] = None,
    ) -> dict[str, WeightData]:
        pipeline_config = self.pipeline_config
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
        return state_dict

    def graph_inputs(self) -> tuple[TensorType]:
        # Generate DeviceRef
        device_ref = DeviceRef.from_device(self.devices[0])

        # Construct general input types
        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )

        kv_inputs = self.kv_manager.input_symbols()

        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        )

        if len(self.devices) > 1:
            # Flatten kv types for each device
            flattened_kv_types = [
                kv_type for sublist in kv_inputs for kv_type in sublist
            ]
            signals = Signals(
                devices=(DeviceRef(d.label, d.id) for d in self.devices)
            )
            return (
                tokens_type,
                input_row_offsets_type,
                return_n_logits_type,
                *signals.input_types(),
                *flattened_kv_types,
            )
        else:
            return (
                tokens_type,
                input_row_offsets_type,
                return_n_logits_type,
                *kv_inputs[0],
            )

    def _unflatten_kv_inputs(
        self, kv_inputs_flat: Sequence[TensorValue]
    ) -> list[tuple[TensorValue, ...]]:
        kv_params = MistralConfig.get_kv_params(
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

    @traced
    def _build_graph(
        self, weights: Weights, adapter: Optional[WeightsAdapter] = None
    ) -> Graph:
        # Retrieve config
        state_dict = self._get_state_dict(weights, adapter)

        kv_params = MistralConfig.get_kv_params(
            huggingface_config=self.huggingface_config,
            n_devices=len(self.devices),
            kv_cache_config=self.kv_cache_config,
            cache_dtype=self.encoding.cache_dtype,
        )
        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in self.pipeline_config.model_config.device_specs
        ]
        model_config = MistralConfig(
            hidden_size=self.huggingface_config.hidden_size,
            num_attention_heads=self.huggingface_config.num_attention_heads,
            num_key_value_heads=kv_params.n_kv_heads,
            num_hidden_layers=self.huggingface_config.num_hidden_layers,
            vocab_size=self.huggingface_config.vocab_size,
            dtype=self.dtype,
            kv_params=kv_params,
            return_logits=self.return_logits,
            attention_multiplier=math.sqrt(1 / kv_params.head_dim),
            head_dim=self.huggingface_config.head_dim,
            rope_theta=self.huggingface_config.rope_theta,
            max_seq_len=self.calculate_max_seq_len(
                self.pipeline_config, huggingface_config=self.huggingface_config
            ),
            rms_norm_eps=self.huggingface_config.rms_norm_eps,
            feed_forward_length=self.huggingface_config.intermediate_size,
            devices=device_refs,
        )

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
                kv_cache = [
                    v.tensor for v in variadic_args[len(self.devices) :]
                ]

                kv_caches_per_dev = self._unflatten_kv_inputs(kv_cache)

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
                outputs = nn_model(
                    tokens.tensor,
                    [inp.tensor for inp in kv_cache_inputs],
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
            msg = "Mistral model does not currently implement enable echo."
            raise ValueError(msg)

        # Pre-allocate a buffer for input_row_offsets in multistep execution.
        # We do this to avoid materializing and copying a buffer with each multistep step
        assert self.pipeline_config.max_batch_size, (
            "Expected max_batch_size to be set"
        )
        self._input_row_offsets_prealloc = Tensor.from_numpy(
            np.arange(self.pipeline_config.max_batch_size + 1, dtype=np.uint32)
        ).to(self.devices[0])

        if not isinstance(self.weights, SafetensorWeights):
            msg = "only safetensors weights are currently supported in Mistral models."
            raise ValueError(msg)

        logger.info("Building and compiling model...")
        before = time.perf_counter()
        graph = self._build_graph(self.weights, self.adapter)
        after_build = time.perf_counter()

        logger.info(f"Building graph took {after_build - before:.6f} seconds")

        before_compile = time.perf_counter()
        model = session.load(graph, weights_registry=self.state_dict)
        after = time.perf_counter()

        logger.info(
            f"Compiling model took {after - before_compile:.6f} seconds"
        )

        logger.info(
            f"Building and compiling model took {after - before:.6f} seconds"
        )

        return model
