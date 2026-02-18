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
from typing import Any, Literal

import numpy as np
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, Value
from max.graph.weights import WeightData, Weights, WeightsAdapter
from max.interfaces import LogProbabilities
from max.nn.legacy.kv_cache import (
    KVCacheInputs,
    KVCacheParams,
    PagedCacheValues,
)
from max.nn.legacy.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    CompilationTimer,
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
from max.pipelines.lib.utils import compute_data_parallel_splits
from max.profiler import traced
from max.support.algorithm import flatten2d
from transformers import AutoConfig

from .data_parallel_llama import create_graph as create_data_parallel_graph
from .distributed_llama import DistributedLlama3
from .llama3 import Llama3
from .model_config import Llama3Config

logger = logging.getLogger("max.pipelines")


@dataclass
class Llama3Inputs(ModelInputs):
    """A class representing inputs for the Llama3 model.

    This class encapsulates the input tensors required for the Llama3 model
    execution.
    """

    tokens: Buffer
    """Tensor containing the input token IDs."""

    input_row_offsets: Buffer
    """Tensor containing the offsets for each row in the ragged input
    sequence."""

    signal_buffers: list[Buffer]
    """Device buffers used for synchronization in communication collectives."""

    return_n_logits: Buffer

    lora_grouped_offsets: Buffer | None = None
    num_active_loras: Buffer | None = None
    lora_end_idx: Buffer | None = None
    batch_seq_len: Buffer | None = None
    lora_ids_kv: Buffer | None = None
    lora_grouped_offsets_kv: Buffer | None = None
    data_parallel_splits: Buffer | Sequence[Sequence[int]] | None = None
    """Tensor containing the data parallel splits."""

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        kv_cache_inputs = tuple(self.kv_cache_inputs or ())
        if self.data_parallel_splits is not None:
            if isinstance(self.data_parallel_splits, Buffer):
                splits_tensor = self.data_parallel_splits
            else:
                splits_array = np.concatenate(
                    [
                        np.array(split, dtype=np.int64)
                        for split in self.data_parallel_splits
                    ]
                )
                splits_tensor = Buffer.from_numpy(splits_array).to(
                    self.tokens.device
                )
            return (
                self.tokens,
                self.input_row_offsets,
                self.return_n_logits,
                splits_tensor,
                *kv_cache_inputs,
            )

        return (
            self.tokens,
            self.input_row_offsets,
            self.return_n_logits,
            *self.signal_buffers,
            *kv_cache_inputs,
        )


class LlamaModelBase(PipelineModel[TextContext], KVCacheMixin):
    """Base Llama pipeline model implementation."""

    model: Model
    """Compiled and initialized model ready for inference."""

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
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
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
            return_hidden_states,
        )
        self.model = self.load_model(session)
        self.logprobs_device = devices[0]
        self.logprobs_model = self.load_logprobs_model(session)
        self._execution_input_buffers: dict[
            tuple[int, int], tuple[Buffer, Buffer, Buffer, Buffer]
        ] = {}

    # TODO(zheng): Remove these wrappers once get_kv_params doesn't have to be
    # called from PipelineModel's __init__ method.
    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return Llama3Config.construct_kv_params(
            huggingface_config,
            pipeline_config,
            devices,
            kv_cache_config,
            cache_dtype,
        )

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        assert isinstance(model_inputs, Llama3Inputs)

        if self.pipeline_config.model.data_parallel_degree > 1:
            model_outputs = self.model.execute(*model_inputs.buffers)
        elif self._lora_manager:
            curr_kv_cache_inputs = model_inputs.kv_cache_inputs or ()
            model_outputs = self.model.execute(
                model_inputs.tokens,
                model_inputs.input_row_offsets,
                model_inputs.return_n_logits,
                model_inputs.lora_ids,  # type: ignore
                model_inputs.lora_ranks,  # type: ignore
                model_inputs.lora_grouped_offsets,  # type: ignore
                model_inputs.num_active_loras,  # type: ignore
                model_inputs.lora_end_idx,  # type: ignore
                model_inputs.batch_seq_len,  # type: ignore
                model_inputs.lora_ids_kv,  # type: ignore
                model_inputs.lora_grouped_offsets_kv,  # type: ignore
                *model_inputs.signal_buffers,
                *curr_kv_cache_inputs,
            )
        else:
            model_outputs = self.model.execute(*model_inputs.buffers)

        has_offsets = self.return_logits in (
            ReturnLogits.VARIABLE,
            ReturnLogits.ALL,
        )
        has_hidden_states = self.return_hidden_states != ReturnHiddenStates.NONE

        assert isinstance(model_outputs[0], Buffer)
        if has_offsets and has_hidden_states:
            assert len(model_outputs) == 4
            assert isinstance(model_outputs[1], Buffer)
            assert isinstance(model_outputs[2], Buffer)
            assert isinstance(model_outputs[3], Buffer)
            return ModelOutputs(
                logits=model_outputs[1],
                next_token_logits=model_outputs[0],
                logit_offsets=model_outputs[2],
                hidden_states=model_outputs[3],
            )
        elif has_offsets:
            assert len(model_outputs) == 3
            assert isinstance(model_outputs[1], Buffer)
            assert isinstance(model_outputs[2], Buffer)
            return ModelOutputs(
                logits=model_outputs[1],
                next_token_logits=model_outputs[0],
                logit_offsets=model_outputs[2],
            )
        elif has_hidden_states:
            assert len(model_outputs) == 2
            assert isinstance(model_outputs[1], Buffer)
            return ModelOutputs(
                logits=model_outputs[0],
                next_token_logits=model_outputs[0],
                hidden_states=model_outputs[1],
            )
        else:
            assert len(model_outputs) == 1
            return ModelOutputs(
                logits=model_outputs[0],
                next_token_logits=model_outputs[0],
            )

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> Llama3Inputs:
        """Prepare the inputs for the first pass in multistep execution."""
        dp = self.pipeline_config.model.data_parallel_degree
        if len(replica_batches) != dp:
            raise ValueError(
                "Number of replica batches must match data parallel degree"
            )

        context_batch = flatten2d(replica_batches)

        # Build the model inputs on host memory and copy to device.
        device0 = self.devices[0]
        pinned = not device0.is_host

        batch_size = len(context_batch)
        total_seq_len = sum(ctx.tokens.active_length for ctx in context_batch)
        buffer_key = (batch_size, total_seq_len)
        buffers = self._execution_input_buffers.get(buffer_key)
        if buffers is None:
            host_tokens = Buffer(
                shape=(total_seq_len,),
                dtype=DType.int64,
                device=device0,
                pinned=pinned,
            )

            if pinned:
                host_tokens.disable_auto_sync()

            host_row_offsets = Buffer(
                shape=(batch_size + 1,),
                dtype=DType.uint32,
                device=device0,
                pinned=pinned,
            )

            if pinned:
                host_row_offsets.disable_auto_sync()

            device_tokens = host_tokens.to(device0)
            device_row_offsets = host_row_offsets.to(device0)
            buffers = (
                host_tokens,
                host_row_offsets,
                device_tokens,
                device_row_offsets,
            )
            self._execution_input_buffers[buffer_key] = buffers
        (
            host_tokens,
            host_row_offsets,
            device_tokens,
            device_row_offsets,
        ) = buffers

        # Get input_row_offsets: start and end position of each batch in the
        # combined total_seq_len dimension.
        input_row_offsets_np = host_row_offsets.to_numpy()
        np.cumsum(
            [0] + [ctx.tokens.active_length for ctx in context_batch],
            dtype=np.uint32,
            out=input_row_offsets_np,
        )

        # return_n_logits_tensor does not need to be pinned since it is not
        # copied to the device.
        return_n_logits_tensor = Buffer.from_numpy(
            np.array([return_n_logits], dtype=np.int64)
        )

        # Create a ragged token vector of length: sum(len(t) for t in tokens).
        tokens_np = host_tokens.to_numpy()
        if context_batch:
            np.concatenate(
                [ctx.tokens.active for ctx in context_batch],
                out=tokens_np,
            )
        device_tokens.inplace_copy_from(host_tokens)
        device_row_offsets.inplace_copy_from(host_row_offsets)

        # Constructs splits for the data parallel execution.
        if dp > 1:
            data_parallel_splits = Buffer.from_numpy(
                compute_data_parallel_splits(replica_batches)
            )
        else:
            data_parallel_splits = None

        inputs = Llama3Inputs(
            tokens=device_tokens,
            input_row_offsets=device_row_offsets,
            signal_buffers=self.signal_buffers,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=return_n_logits_tensor,
            data_parallel_splits=data_parallel_splits,
        )

        # Map model names to LoRA graph inputs
        if self._lora_manager:
            # TODO: Move LORA graph inputs to pinned memory
            (
                lora_ids,
                lora_ranks,
                lora_grouped_offsets,
                num_active_loras,
                lora_end_idx,
                batch_seq_len,
                lora_ids_kv,
                lora_grouped_offsets_kv,
            ) = self._lora_manager.get_lora_graph_inputs(
                context_batch, input_row_offsets_np, self.devices[0]
            )

            inputs.lora_ids = lora_ids
            inputs.lora_ranks = lora_ranks
            inputs.lora_grouped_offsets = lora_grouped_offsets
            inputs.num_active_loras = num_active_loras
            inputs.lora_end_idx = lora_end_idx
            inputs.batch_seq_len = batch_seq_len
            inputs.lora_ids_kv = lora_ids_kv
            inputs.lora_grouped_offsets_kv = lora_grouped_offsets_kv

        return inputs

    def prepare_next_token_inputs(
        self,
        next_tokens: Buffer,
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
            num_active_loras=prev_model_inputs.num_active_loras,
            lora_end_idx=prev_model_inputs.lora_end_idx,
            batch_seq_len=prev_model_inputs.batch_seq_len,
            lora_ids_kv=prev_model_inputs.lora_ids_kv,
            lora_grouped_offsets_kv=prev_model_inputs.lora_grouped_offsets_kv,
            data_parallel_splits=prev_model_inputs.data_parallel_splits,
        )

    @classmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        return Llama3Config.calculate_max_seq_len(
            pipeline_config, huggingface_config
        )

    @traced
    def load_model(self, session: InferenceSession) -> Model:
        # Pre-allocate a buffer for input_row_offsets in multistep execution.
        # We do this to avoid materializing and copying a buffer with each multistep step
        assert self.pipeline_config.max_batch_size, (
            "Expected max_batch_size to be set"
        )
        self._input_row_offsets_prealloc = Buffer.from_numpy(
            np.arange(self.pipeline_config.max_batch_size + 1, dtype=np.uint32)
        ).to(self.devices[0])

        timer = CompilationTimer("model")
        graph = self._build_graph(self.weights, self.adapter)
        timer.mark_build_complete()
        model = session.load(graph, weights_registry=self.state_dict)
        timer.done()

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
        kv_params = Llama3Config.construct_kv_params(
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
        model_config = Llama3Config.initialize(self.pipeline_config)
        model_config.finalize(
            huggingface_config=self.huggingface_config,
            state_dict=state_dict,
            norm_method=self.norm_method,
            attention_bias=self.attention_bias,
            return_logits=self.return_logits,
            return_hidden_states=self.return_hidden_states,
        )

        if model_config.data_parallel_degree > 1:
            graph, new_state_dict = create_data_parallel_graph(
                model_config, self.kv_params, state_dict
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
                input_types=dist_model.input_types(self.kv_params),
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
                    self.kv_params, self._lora_manager
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
                        num_active_loras,
                        lora_end_idx,
                        batch_seq_len,
                        lora_ids_kv,
                        lora_grouped_offsets_kv,
                        *kv_cache_inputs,
                    ) = graph.inputs
                    self._lora_manager.set_graph_info(
                        lora_ids.tensor,
                        lora_ranks.tensor,
                        lora_grouped_offsets.tensor,
                        num_active_loras.tensor,
                        lora_end_idx.tensor,
                        batch_seq_len.tensor,
                        lora_ids_kv.tensor,
                        lora_grouped_offsets_kv.tensor,
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
        next_tokens: Buffer,
        batch_top_n: list[int],
        batch_echo: list[bool],
    ) -> list[LogProbabilities | None]:
        assert model_outputs.next_token_logits is not None
        next_token_logits = model_outputs.next_token_logits

        assert isinstance(model_inputs, Llama3Inputs)
        llama3_inputs: Llama3Inputs = model_inputs

        sampled_tokens = next_tokens.to_numpy()
        tokens = llama3_inputs.tokens.to_numpy()
        input_row_offsets = llama3_inputs.input_row_offsets.to_numpy()

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
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
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
            return_hidden_states,
        )
