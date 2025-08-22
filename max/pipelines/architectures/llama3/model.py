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
from math import ceil
from typing import Any, Callable, Literal, Optional, Union

import numpy as np
from max.driver import Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, DeviceRef, Graph, TensorType, TensorValue
from max.graph.weights import WeightData, Weights, WeightsAdapter
from max.interfaces import LogProbabilities
from max.nn import ReturnLogits, Signals
from max.nn.kv_cache import (
    FetchPagedKVCacheCollection,
    KVCacheInputs,
    KVCacheManager,
    KVCacheParams,
    KVCacheStrategy,
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
)
from max.pipelines.lib.log_probabilities import (
    compute_log_probabilities_ragged,
    log_probabilities_ragged_graph,
)
from max.profiler import traced
from transformers import AutoConfig

from .distributed_llama import DistributedLlama3
from .llama3 import Llama3
from .model_config import Llama3Config
from .pipeline_parallel_llama3 import PipelineParallelLlama3

logger = logging.getLogger("max.pipelines")


class Llama3Inputs(ModelInputs):
    """A class representing inputs for the Llama3 model.

    This class encapsulates the input tensors required for the Llama3 model
    execution.
    """

    tokens: Tensor
    """Tensor containing the input token IDs."""

    input_row_offsets: Tensor
    """Tensor containing the offsets for each row in the ragged input
    sequence."""

    signal_buffers: list[Tensor]
    """Device buffers used for synchronization in communication collectives."""

    return_n_logits: Tensor

    def __init__(
        self,
        tokens: Tensor,
        input_row_offsets: Tensor,
        signal_buffers: list[Tensor],
        return_n_logits: Tensor,
        kv_cache_inputs: KVCacheInputs | None = None,
        lora_ids: Tensor | None = None,
        lora_ranks: Tensor | None = None,
    ) -> None:
        """
        Args:
            tokens: Input token IDs.
            input_row_offsets: Input row offsets (ragged tensors).
            signal_buffers: Device buffers used for synchronization in
                communication collectives.
        """
        self.tokens = tokens
        self.input_row_offsets = input_row_offsets
        self.signal_buffers = signal_buffers
        self.kv_cache_inputs = kv_cache_inputs
        self.return_n_logits = return_n_logits
        self.lora_ids = lora_ids
        self.lora_ranks = lora_ranks


class LlamaModelBase(PipelineModel[TextContext]):
    """Base Llama pipeline model implementation."""

    model: Model
    """Compiled and initialized model ready for inference."""

    signal_buffers: list[Tensor]
    """Device buffers used for synchronization in communication collectives."""

    norm_method: Literal["rms_norm"] | Literal["layer_norm"]
    """Normalization layer."""

    attention_bias: bool = False
    """Whether to use attention bias."""

    logits_postprocessor: Callable[[TensorValue], TensorValue] | None = None
    """Postprocessor for the logits."""

    state_dict: dict[str, Any]
    """Weights to load into the model."""

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
        """
        Args:
            pipeline_config: The configuration for this pipeline.
            session: The container for the runtime for this model.
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
        self.logprobs_device = devices[0]
        self.logprobs_model = self.load_logprobs_model(session)

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

    # TODO(zheng): Remove these wrappers once get_kv_params doesn't have to be
    # called from PipelineModel's infer_optimal_batch_size method.
    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return Llama3Config.get_kv_params(
            huggingface_config, n_devices, kv_cache_config, cache_dtype
        )

    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        return Llama3Config.get_num_layers(huggingface_config)

    def graph_inputs(self) -> tuple[Union[TensorType, BufferType], ...]:
        # Generate DeviceRef
        device_ref = DeviceRef.from_device(self.devices[0])

        # Construct general input types
        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )

        kv_inputs = self.kv_manager.input_symbols()

        # Construct Graph Inputs
        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        )

        if len(self.devices) > 1:
            # Flatten kv types for each device
            flattened_kv_types: list[TensorType] = [
                kv_type for sublist in kv_inputs for kv_type in sublist
            ]

            signals = Signals(
                devices=(DeviceRef(d.label, d.id) for d in self.devices)
            )

            # Explicitly construct tuple with mixed types
            signal_buffer_types: list[BufferType] = signals.input_types()

            # Build the complete input types list
            all_input_types: list[Union[TensorType, BufferType]] = [
                tokens_type,
                input_row_offsets_type,
                return_n_logits_type,
            ]
            all_input_types.extend(signal_buffer_types)
            all_input_types.extend(flattened_kv_types)

            return tuple(all_input_types)
        else:
            if self._lora_manager:
                lora_ids, lora_ranks = self._lora_manager.input_symbols(
                    device_ref
                )
                return (
                    tokens_type,
                    input_row_offsets_type,
                    return_n_logits_type,
                    lora_ids,
                    lora_ranks,
                    *kv_inputs[0],
                )
            else:
                return (
                    tokens_type,
                    input_row_offsets_type,
                    return_n_logits_type,
                    *kv_inputs[0],
                )

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        assert isinstance(model_inputs, Llama3Inputs)
        curr_kv_cache_inputs = model_inputs.kv_cache_inputs or ()

        if self._lora_manager:
            model_outputs = self.model.execute(
                model_inputs.tokens,
                model_inputs.input_row_offsets,
                model_inputs.return_n_logits,
                model_inputs.lora_ids,  # type: ignore
                model_inputs.lora_ranks,  # type: ignore
                *model_inputs.signal_buffers,
                *curr_kv_cache_inputs,
            )
        else:
            model_outputs = self.model.execute(
                model_inputs.tokens,
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
                logits=model_outputs[1],
                next_token_logits=model_outputs[0],
                logit_offsets=model_outputs[2],
            )
        else:
            assert isinstance(model_outputs[0], Tensor)
            return ModelOutputs(
                logits=model_outputs[0], next_token_logits=model_outputs[0]
            )

    def prepare_initial_token_inputs(
        self,
        context_batch: Sequence[TextContext],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> Llama3Inputs:
        """Prepare the inputs for the first pass in multistep execution."""
        # Get input_row_offsets: start and end position of each batch in the
        # combined total_seq_len dimension.
        input_row_offsets = np.cumsum(
            [0] + [ctx.active_length for ctx in context_batch],
            dtype=np.uint32,
        )

        # Create a ragged token vector of length: sum(len(t) for t in tokens).
        tokens = Tensor.from_numpy(
            np.concatenate([ctx.next_tokens for ctx in context_batch])
        ).to(self.devices[0])

        inputs = Llama3Inputs(
            tokens=tokens,
            input_row_offsets=Tensor.from_numpy(input_row_offsets).to(
                self.devices[0]
            ),
            signal_buffers=self.signal_buffers,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=Tensor.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
        )

        # Map model names to LoRA graph inputs
        if self._lora_manager:
            model_names: list[str | None] = [
                ctx.model_name if ctx.model_name else None
                for ctx in context_batch
            ]
            lora_ids, lora_ranks = self._lora_manager.get_lora_graph_inputs(
                model_names, self.devices[0]
            )
            inputs.lora_ids = lora_ids
            inputs.lora_ranks = lora_ranks

        return inputs

    def prepare_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: ModelInputs,
    ) -> Llama3Inputs:
        """Prepare the inputs for the next token in multistep execution.
        This should avoid any device synchronization or copy operations.
        """
        assert isinstance(prev_model_inputs, Llama3Inputs)
        row_offsets_size = prev_model_inputs.input_row_offsets.shape[0]
        next_row_offsets = self._input_row_offsets_prealloc[:row_offsets_size]

        return Llama3Inputs(
            tokens=next_tokens,
            input_row_offsets=next_row_offsets,
            signal_buffers=self.signal_buffers,
            kv_cache_inputs=prev_model_inputs.kv_cache_inputs,
            return_n_logits=prev_model_inputs.return_n_logits,
            lora_ids=prev_model_inputs.lora_ids,
            lora_ranks=prev_model_inputs.lora_ranks,
        )

    @classmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        return Llama3Config.calculate_max_seq_len(
            pipeline_config, huggingface_config
        )

    def load_kv_manager(
        self,
        session: InferenceSession,
        available_cache_memory: int,
    ) -> KVCacheManager:
        # For pipeline parallel, use layers per stage instead of total layers
        num_layers_for_cache = Llama3Config.get_num_layers(
            huggingface_config=self.huggingface_config
        )

        # For pipeline parallel, use n_devices=1 so each stage gets all heads
        n_devices_for_cache = len(self.devices)

        # Check if this is pipeline parallel mode
        pp_degree = self.pipeline_config.model_config.pipeline_parallel_degree
        if pp_degree > 1:
            # Use layers per stage for pipeline parallel

            num_layers_for_cache = ceil(num_layers_for_cache / pp_degree)
            # Use single device so each stage gets all heads (not split across devices)
            n_devices_for_cache = 1
            logger.debug(
                f"[PP KV Cache] Main KV manager using {num_layers_for_cache} layers and n_devices=1 (for stage-specific caching)"
            )

        return load_kv_manager(
            params=Llama3Config.get_kv_params(
                huggingface_config=self.huggingface_config,
                n_devices=n_devices_for_cache,
                kv_cache_config=self.kv_cache_config,
                cache_dtype=self.encoding.cache_dtype,
                pipeline_parallel_degree=pp_degree,
            ),
            max_batch_size=self.pipeline_config.max_batch_size,
            max_seq_len=self.calculate_max_seq_len(
                self.pipeline_config, huggingface_config=self.huggingface_config
            ),
            num_layers=num_layers_for_cache,
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
        return estimate_kv_cache_size(
            params=Llama3Config.get_kv_params(
                huggingface_config=huggingface_config,
                n_devices=len(devices),
                kv_cache_config=kv_cache_config,
                cache_dtype=cache_dtype,
            ),
            max_batch_size=pipeline_config.max_batch_size,
            max_seq_len=cls.calculate_max_seq_len(
                pipeline_config, huggingface_config=huggingface_config
            ),
            num_layers=Llama3Config.get_num_layers(
                huggingface_config=huggingface_config
            ),
            available_cache_memory=available_cache_memory,
            devices=devices,
        )

    @traced
    def load_model(self, session: InferenceSession) -> Model:
        # Pre-allocate a buffer for input_row_offsets in multistep execution.
        # We do this to avoid materializing and copying a buffer with each multistep step
        assert self.pipeline_config.max_batch_size, (
            "Expected max_batch_size to be set"
        )
        self._input_row_offsets_prealloc = Tensor.from_numpy(
            np.arange(self.pipeline_config.max_batch_size + 1, dtype=np.uint32)
        ).to(self.devices[0])

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

    @traced
    def load_logprobs_model(self, session: InferenceSession) -> Model:
        # TODO: Perhaps 'levels' ought to be configurable.
        graph = log_probabilities_ragged_graph(
            DeviceRef.from_device(self.logprobs_device), levels=3
        )
        return session.load(graph)

    def _unflatten_kv_inputs(
        self, kv_inputs_flat: Sequence[TensorValue]
    ) -> list[tuple[TensorValue, ...]]:
        kv_params = Llama3Config.get_kv_params(
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

    def _get_state_dict(
        self,
        weights: Weights,
        adapter: Optional[WeightsAdapter] = None,
    ) -> dict[str, WeightData]:
        # Get Config
        huggingface_config = self.huggingface_config
        if adapter:
            state_dict = adapter(
                dict(weights.items()),
                huggingface_config=huggingface_config,
                pipeline_config=self.pipeline_config,
            )
        else:
            state_dict = {key: value.data() for key, value in weights.items()}

        return state_dict

    def _build_pipeline_parallel_graph(
        self, state_dict: dict[str, WeightData], model_config: Llama3Config
    ) -> Graph:
        """Build graph for pipeline parallel model with self-contained KV cache management.

        This method handles all session management, KV cache setup, graph compilation,
        and weight loading internally to simplify the PP model interface.
        """

        # Calculate stage assignments
        pp_degree = model_config.pipeline_parallel_degree
        num_layers = model_config.num_hidden_layers

        # Use shared helper method from pipeline_parallel_llama3
        stage_assignments = PipelineParallelLlama3._compute_stage_assignments(
            num_layers, pp_degree
        )

        logger.info(
            f"[PP] Self-contained graph building for {len(stage_assignments)} stages"
        )

        # Create stage-specific KV cache managers and collections internally
        stage_kv_collections = []
        stage_kv_input_symbols = []

        for stage_idx, (start_layer, end_layer) in enumerate(stage_assignments):
            num_layers_in_stage = end_layer - start_layer
            stage_device = self.devices[stage_idx]

            # Get KV input symbols for this stage
            stage_kv_inputs = self.kv_manager.input_symbols(
                devices=[stage_device], num_layers=num_layers_in_stage
            )[0]  # Single device per stage
            stage_kv_input_symbols.append(stage_kv_inputs)

            kv_collection_func: Any
            if model_config.kv_params.cache_strategy == KVCacheStrategy.PAGED:
                kv_collection_func = FetchPagedKVCacheCollection(
                    self.kv_manager.params, num_layers=num_layers_in_stage
                )
            else:
                raise ValueError(
                    f"Unsupported cache strategy: {model_config.kv_params.cache_strategy}"
                )

            stage_kv_collections.append(kv_collection_func)

            logger.debug(
                f"[PP] Stage {stage_idx}: {num_layers_in_stage} layers "
                f"(layers {start_layer}-{end_layer - 1}), "
                f"device {stage_device.id}, {len(list(stage_kv_inputs))} KV inputs"
            )

        # Build graph input types: tokens, offsets, return_n_logits, signals, stage KV inputs
        device_ref = DeviceRef.from_device(self.devices[0])

        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        )
        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )

        signals = Signals(
            devices=(DeviceRef(d.label, d.id) for d in self.devices)
        )

        # Flatten all stage KV inputs
        all_stage_kv_inputs: list[TensorType] = []
        for stage_kv_inputs in stage_kv_input_symbols:
            all_stage_kv_inputs.extend(stage_kv_inputs)

        # Explicitly construct pipeline parallel graph inputs with mixed types
        signal_buffer_types: list[BufferType] = signals.input_types()

        pp_input_types: list[Union[TensorType, BufferType]] = [
            tokens_type,
            input_row_offsets_type,
            return_n_logits_type,
        ]
        pp_input_types.extend(signal_buffer_types)
        pp_input_types.extend(all_stage_kv_inputs)

        pp_graph_inputs = tuple(pp_input_types)

        logger.debug(
            f"[PP] Self-contained graph inputs: {len(pp_graph_inputs)}"
        )

        # Create PP model without session dependency
        pp_model: PipelineParallelLlama3 = PipelineParallelLlama3(model_config)

        # Load weights internally
        filtered_state_dict = {
            k: v for k, v in state_dict.items() if k != "rope_freqs.weight"
        }
        pp_model.load_state_dict(
            filtered_state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
            strict=False,
        )

        self.state_dict = pp_model.state_dict()

        # Build graph with stage-specific KV caches
        with Graph(
            getattr(self.huggingface_config, "model_type", "llama3"),
            input_types=pp_graph_inputs,
        ) as graph:
            tokens, input_row_offsets, return_n_logits, *variadic_args = (
                graph.inputs
            )

            # Extract signal buffers
            signal_buffers = [
                v.buffer for v in variadic_args[: len(self.devices)]
            ]

            # Extract stage-specific KV cache inputs
            kv_cache_start = len(self.devices)
            current_idx = kv_cache_start

            stage_kv_caches = []
            for stage_idx, stage_kv_inputs in enumerate(stage_kv_input_symbols):
                stage_kv_tensors = []
                stage_kv_list = list(stage_kv_inputs)
                for _ in range(len(stage_kv_list)):
                    stage_kv_tensors.append(variadic_args[current_idx].tensor)
                    current_idx += 1

                # Create stage KV cache collection using the pre-built collection function
                kv_collection = stage_kv_collections[stage_idx](
                    *stage_kv_tensors
                )
                stage_kv_caches.append(kv_collection)

            logger.debug(
                f"[PP] Self-contained graph created {len(stage_kv_caches)} stage KV caches"
            )

            # Call PP model with stage-specific KV caches
            outputs = pp_model(
                tokens.tensor,
                stage_kv_caches,  # Stage-specific KV cache collections
                return_n_logits.tensor,
                input_row_offsets.tensor,
            )

            graph.output(*outputs)
            return graph

    def _build_graph(
        self,
        weights: Weights,
        adapter: Optional[WeightsAdapter] = None,
    ) -> Graph:
        # Retrieve config
        state_dict = self._get_state_dict(weights, adapter)
        model_config = Llama3Config.generate(
            pipeline_config=self.pipeline_config,
            huggingface_config=self.huggingface_config,
            state_dict=state_dict,
            dtype=self.dtype,
            n_devices=len(self.devices),
            logits_postprocessor=self.logits_postprocessor,
            norm_method=self.norm_method,
            attention_bias=self.attention_bias,
            cache_dtype=self.encoding.cache_dtype,
            kv_cache_config=self.kv_cache_config,
            return_logits=self.return_logits,
            pipeline_parallel_degree=self.pipeline_config.model_config.pipeline_parallel_degree,
            tensor_parallel_degree=self.pipeline_config.model_config.tensor_parallel_degree,
        )

        # Pipeline Parallel case - early return to avoid changing existing logic
        if model_config.pipeline_parallel_degree > 1:
            if model_config.tensor_parallel_degree != 1:
                raise ValueError(
                    "Hybrid TP+PP not supported yet. Use either TP>1 or PP>1, not both."
                )
            logger.info(
                f"Using Pipeline Parallel with {model_config.pipeline_parallel_degree} stages"
            )

            return self._build_pipeline_parallel_graph(state_dict, model_config)

        # Tensor Parallel case
        if len(self.devices) > 1:
            dist_model: DistributedLlama3 = DistributedLlama3(model_config)

            # Load weights.
            dist_model.load_state_dict(
                state_dict,
                override_quantization_encoding=True,
                weight_alignment=1,
                strict=False,  # TODO(MODELS-550) `rope_freqs.weight` not used
            )

            self.state_dict = dist_model.state_dict()

            with Graph(
                getattr(self.huggingface_config, "model_type", "llama3"),
                input_types=self.graph_inputs(),
            ) as graph:
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

                outputs = dist_model(
                    tokens.tensor,
                    signal_buffers,
                    kv_caches_per_dev,
                    return_n_logits.tensor,
                    input_row_offsets.tensor,
                )

                graph.output(*outputs)
                return graph

        # Single GPU case
        else:
            single_model: Llama3 = Llama3(model_config)

            if self._lora_manager:
                self._lora_manager.init_weights(single_model, state_dict)

            # Load weights.
            single_model.load_state_dict(
                state_dict,
                override_quantization_encoding=True,
                weight_alignment=1,
                strict=False,  # TODO(MODELS-550) `rope_freqs.weight` not used
            )
            self.state_dict = single_model.state_dict()

            with Graph("llama3", input_types=self.graph_inputs()) as graph:
                if self._lora_manager:
                    (
                        tokens,
                        input_row_offsets,
                        return_n_logits,
                        lora_ids,
                        lora_ranks,
                        *kv_cache_inputs,
                    ) = graph.inputs
                    self._lora_manager.set_graph_info(
                        lora_ids.tensor,
                        lora_ranks.tensor,
                    )
                else:
                    (
                        tokens,
                        input_row_offsets,
                        return_n_logits,
                        *kv_cache_inputs,
                    ) = graph.inputs
                outputs = single_model(
                    tokens.tensor,
                    [inp.tensor for inp in kv_cache_inputs],
                    return_n_logits.tensor,
                    input_row_offsets.tensor,
                )
                graph.output(*outputs)
                return graph

    def compute_log_probabilities(
        self,
        session: InferenceSession,
        model_inputs: ModelInputs,
        model_outputs: ModelOutputs,
        next_tokens: Tensor,
        batch_top_n: list[int],
        batch_echo: list[bool],
    ) -> list[LogProbabilities | None] | None:
        logits = model_outputs.logits
        assert model_outputs.next_token_logits is not None
        next_token_logits = model_outputs.next_token_logits

        assert isinstance(model_inputs, Llama3Inputs)
        llama3_inputs: Llama3Inputs = model_inputs

        sampled_tokens = next_tokens.to_numpy()
        tokens = llama3_inputs.tokens.to_numpy()
        input_row_offsets = llama3_inputs.input_row_offsets.to_numpy()

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


class Llama3Model(LlamaModelBase):
    """Llama 3 pipeline model implementation."""

    norm_method: Literal["rms_norm"] | Literal["layer_norm"] = "rms_norm"
    """Normalization layer."""

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
