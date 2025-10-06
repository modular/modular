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
from typing import Any, Literal

import numpy as np
from max.driver import Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, Value
from max.graph.weights import WeightData, Weights, WeightsAdapter
from max.interfaces import LogProbabilities
from max.nn import ReturnLogits, Signals
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheParams,
    MultiPagedKVCacheManager,
    PagedCacheValues,
    PagedKVCacheManager,
    estimate_kv_cache_size,
    load_kv_manager,
)
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    KVCacheConfig,
    KVCacheMixin,
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

from .data_parallel_llama import create_graph as create_data_parallel_graph
from .distributed_llama import DistributedLlama3
from .llama3 import Llama3
from .model_config import Llama3Config

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

    data_parallel_splits: Tensor | None = None
    """Tensor containing the data parallel splits."""

    def __init__(
        self,
        tokens: Tensor,
        input_row_offsets: Tensor,
        signal_buffers: list[Tensor],
        return_n_logits: Tensor,
        kv_cache_inputs: KVCacheInputs | None = None,
        lora_ids: Tensor | None = None,
        lora_ranks: Tensor | None = None,
        lora_grouped_offsets: Tensor | None = None,
        data_parallel_splits: Tensor | None = None,
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
        self.lora_grouped_offsets = lora_grouped_offsets
        self.data_parallel_splits = data_parallel_splits


class LlamaModelBase(PipelineModel[TextContext], KVCacheMixin):
    """Base Llama pipeline model implementation."""

    model: Model
    """Compiled and initialized model ready for inference."""

    signal_buffers: list[Tensor]
    """Device buffers used for synchronization in communication collectives."""

    norm_method: Literal["rms_norm"] | Literal["layer_norm"]
    """Normalization layer."""

    attention_bias: bool = False
    """Whether to use attention bias."""

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
        adapter: WeightsAdapter | None = None,
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

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        curr_kv_cache_inputs = model_inputs.kv_cache_inputs or ()
        assert isinstance(model_inputs, Llama3Inputs)

        if self.pipeline_config.model_config.data_parallel_degree > 1:
            assert model_inputs.data_parallel_splits is not None
            model_outputs = self.model.execute(
                model_inputs.tokens,
                model_inputs.input_row_offsets,
                model_inputs.return_n_logits,
                model_inputs.data_parallel_splits,
                *curr_kv_cache_inputs,
            )
        elif self._lora_manager:
            model_outputs = self.model.execute(
                model_inputs.tokens,
                model_inputs.input_row_offsets,
                model_inputs.return_n_logits,
                model_inputs.lora_ids,  # type: ignore
                model_inputs.lora_ranks,  # type: ignore
                model_inputs.lora_grouped_offsets,  # type: ignore
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

        data_parallel_splits: Tensor | None = None
        if self.pipeline_config.model_config.data_parallel_degree > 1:
            assert isinstance(self.kv_manager, MultiPagedKVCacheManager)
            data_parallel_splits = self.kv_manager.get_data_parallel_splits(
                context_batch
            )

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
            data_parallel_splits=data_parallel_splits,
        )

        # Map model names to LoRA graph inputs
        if self._lora_manager:
            lora_ids, lora_ranks, lora_grouped_offsets = (
                self._lora_manager.get_lora_graph_inputs(
                    context_batch, input_row_offsets, self.devices[0]
                )
            )

            inputs.lora_ids = lora_ids
            inputs.lora_ranks = lora_ranks
            inputs.lora_grouped_offsets = lora_grouped_offsets

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
            lora_grouped_offsets=prev_model_inputs.lora_grouped_offsets,
            data_parallel_splits=prev_model_inputs.data_parallel_splits,
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
        available_cache_memory: int | None,
    ) -> PagedKVCacheManager:
        # For pipeline parallel, use layers per stage instead of total layers
        num_layers_for_cache = Llama3Config.get_num_layers(
            huggingface_config=self.huggingface_config
        )

        n_devices_for_cache = len(self.devices)

        return load_kv_manager(
            params=Llama3Config.get_kv_params(
                huggingface_config=self.huggingface_config,
                n_devices=n_devices_for_cache,
                kv_cache_config=self.kv_cache_config,
                cache_dtype=self.encoding.cache_dtype,
                data_parallel_degree=self.pipeline_config.model_config.data_parallel_degree,
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
        self, kv_inputs_flat: Sequence[Value[Any]]
    ) -> list[PagedCacheValues]:
        kv_params = Llama3Config.get_kv_params(
            huggingface_config=self.huggingface_config,
            n_devices=len(self.devices),
            kv_cache_config=self.kv_cache_config,
            cache_dtype=self.encoding.cache_dtype,
        )
        n_devices = kv_params.n_devices
        fetch_types = self.kv_manager.input_symbols()[0]
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

    def _get_state_dict(
        self,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
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

    def _build_graph(
        self,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
    ) -> Graph:
        # Retrieve config
        state_dict = self._get_state_dict(weights, adapter)
        model_config = Llama3Config.generate(
            pipeline_config=self.pipeline_config,
            huggingface_config=self.huggingface_config,
            state_dict=state_dict,
            dtype=self.dtype,
            n_devices=len(self.devices),
            norm_method=self.norm_method,
            attention_bias=self.attention_bias,
            cache_dtype=self.encoding.cache_dtype,
            kv_cache_config=self.kv_cache_config,
            return_logits=self.return_logits,
            data_parallel_degree=self.pipeline_config.model_config.data_parallel_degree,
        )

        if model_config.data_parallel_degree > 1:
            graph, new_state_dict = create_data_parallel_graph(
                model_config, self.kv_manager, state_dict
            )
            self.state_dict = new_state_dict
            return graph

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
                input_types=dist_model.input_types(self.kv_manager),
            ) as graph:
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

            with Graph(
                "llama3",
                input_types=single_model.input_types(
                    self.kv_manager, self._lora_manager
                ),
            ) as graph:
                if self._lora_manager:
                    (
                        tokens,
                        input_row_offsets,
                        return_n_logits,
                        lora_ids,
                        lora_ranks,
                        lora_grouped_offsets,
                        *kv_cache_inputs,
                    ) = graph.inputs
                    self._lora_manager.set_graph_info(
                        lora_ids.tensor,
                        lora_ranks.tensor,
                        lora_grouped_offsets.tensor,
                    )
                else:
                    (
                        tokens,
                        input_row_offsets,
                        return_n_logits,
                        *kv_cache_inputs,
                    ) = graph.inputs
                kv_collection = PagedCacheValues(
                    kv_blocks=kv_cache_inputs[0].buffer,
                    cache_lengths=kv_cache_inputs[1].tensor,
                    lookup_table=kv_cache_inputs[2].tensor,
                    max_lengths=kv_cache_inputs[3].tensor,
                )
                outputs = single_model(
                    tokens.tensor,
                    kv_collection,
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
        adapter: WeightsAdapter | None = None,
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
