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
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from max._core.engine import Model
from max.driver import Buffer, DevicePinnedBuffer, is_virtual_device_mode
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph
from max.graph.weights import Weights, WeightsAdapter
from max.nn.comm.ep import EPCommInitializer, EPConfig
from max.nn.comm.ep.ep_config import calculate_ep_max_tokens_per_rank
from max.nn.comm.ep.ep_manager import EPBatchManager
from max.nn.kv_cache import KVCacheInputs, KVCacheParams
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    CompilationTimer,
    KVCacheConfig,
    ModelInputs,
    PipelineConfig,
)
from max.pipelines.lib.config.config_enums import supported_encoding_dtype
from max.pipelines.lib.interfaces import AlwaysSignalBuffersMixin
from max.pipelines.lib.quant import parse_quant_config
from max.pipelines.lib.utils import (
    compute_data_parallel_splits,
    parse_state_dict_from_weights,
)
from max.support.algorithm import flatten2d
from transformers import AutoConfig
from typing_extensions import override

from ..llama3.model import Llama3Inputs, LlamaModelBase
from .model_config import Qwen3Config
from .qwen3 import Qwen3

logger = logging.getLogger("max.pipelines")


@dataclass
class Qwen3Inputs(Llama3Inputs):
    """Inputs for Qwen3 models in DP+EP mode.

    Extends Llama3Inputs with host_input_row_offsets and EP-specific buffers
    needed for the hybrid DP-attention + EP-MoE strategy.
    """

    host_input_row_offsets: Buffer | None = None
    ep_inputs: tuple[Buffer, ...] = field(default_factory=tuple)

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        base = [self.tokens, self.input_row_offsets, self.return_n_logits]
        if (
            self.host_input_row_offsets is not None
            and self.data_parallel_splits is not None
        ):
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
            base.extend([self.host_input_row_offsets, splits_tensor])
        return (
            *base,
            *self.signal_buffers,
            *(self.kv_cache_inputs.flatten() if self.kv_cache_inputs else ()),
            *self.ep_inputs,
        )


class Qwen3Model(AlwaysSignalBuffersMixin, LlamaModelBase):
    """Qwen3 pipeline model supporting single-GPU, TP, and DP+EP inference.

    Uses AlwaysSignalBuffersMixin since VocabParallelEmbedding and
    ColumnParallelLinear always require signal buffers for allreduce.
    """

    model: Model
    norm_method: Literal["rms_norm"] | Literal["layer_norm"] = "rms_norm"
    attention_bias: bool = False
    state_dict: dict[str, Any]

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return Qwen3Config.construct_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

    def _create_ep_config(
        self,
        state_dict: dict[str, Any] | None = None,
    ) -> EPConfig | None:
        """Create EP config from pipeline settings.

        Args:
            state_dict: Model weight state dict, required for non-bfloat16
                dispatch dtypes (e.g. FP8) to parse the dispatch quantization
                configuration.
        """
        ep_size = self.pipeline_config.runtime.ep_size
        if ep_size <= 1:
            return None

        n_devices = len(self.devices)
        if ep_size % n_devices != 0:
            raise ValueError(
                f"ep_size ({ep_size}) must be divisible by the number of "
                f"GPUs ({n_devices})."
            )

        config = self.huggingface_config
        n_nodes = ep_size // n_devices
        data_parallel_degree = self.pipeline_config.model.data_parallel_degree

        ep_max_rank_send_tokens = calculate_ep_max_tokens_per_rank(
            max_batch_input_tokens=self.pipeline_config.runtime.max_batch_input_tokens,
            ep_size=ep_size,
            data_parallel_degree=data_parallel_degree,
        )

        encoding = self.pipeline_config.model.quantization_encoding
        dispatch_dtype = (
            supported_encoding_dtype(encoding)
            if encoding is not None
            else DType.bfloat16
        )

        dispatch_quant_config = None
        if dispatch_dtype != DType.bfloat16 and state_dict is not None:
            dispatch_quant_config = parse_quant_config(
                config, state_dict, dispatch_dtype
            )

        return EPConfig(
            dispatch_dtype=dispatch_dtype,
            combine_dtype=DType.bfloat16,
            hidden_size=config.hidden_size,
            top_k=config.num_experts_per_tok,
            n_experts=config.num_experts,
            max_tokens_per_rank=ep_max_rank_send_tokens,
            n_gpus_per_node=n_devices,
            n_nodes=n_nodes,
            dispatch_quant_config=dispatch_quant_config,
        )

    @override
    def load_model(self, session: InferenceSession) -> Model:
        assert self.pipeline_config.runtime.max_batch_size, (
            "Expected max_batch_size to be set"
        )

        dp = self.pipeline_config.model.data_parallel_degree
        max_batch_size = self.pipeline_config.runtime.max_batch_size
        if dp > 1:
            max_batch_size *= dp

        self._input_row_offsets_prealloc: Buffer | None = None
        if not is_virtual_device_mode():
            self._input_row_offsets_prealloc = Buffer.from_numpy(
                np.arange(max_batch_size + 1, dtype=np.uint32)
            ).to(self.devices[0])

        self._host_input_row_offsets_prealloc: Buffer | None = None
        if dp > 1 and not is_virtual_device_mode():
            self._host_input_row_offsets_prealloc = Buffer.from_numpy(
                np.arange(max_batch_size + 1, dtype=np.uint32)
            )

        with CompilationTimer("model") as timer:
            graph = self._build_graph(self.weights, self.adapter, session)
            timer.mark_build_complete()
            model = session.load(graph, weights_registry=self.state_dict)

        return model

    def _build_graph(
        self,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        session: InferenceSession | None = None,
    ) -> Graph:
        state_dict = parse_state_dict_from_weights(
            self.pipeline_config, weights, adapter
        )
        model_config = Qwen3Config.initialize_from_config(
            self.pipeline_config, self.huggingface_config
        )
        model_config.finalize(
            huggingface_config=self.huggingface_config,
            state_dict=state_dict,
            return_logits=self.return_logits,
            norm_method=self.norm_method,
            attention_bias=self.attention_bias,
        )

        # Set up EP config
        ep_config = self._create_ep_config(state_dict)
        model_config.ep_config = ep_config

        # Create EP infrastructure
        ep_manager: EPBatchManager | None = None
        self.ep_comm_initializer: EPCommInitializer | None = None

        if ep_config is not None:
            ep_manager = EPBatchManager(ep_config)

            if not is_virtual_device_mode():
                self.ep_comm_initializer = EPCommInitializer(ep_config)
                if session is not None:
                    self.ep_comm_initializer.ep_init(session)
                    ep_config.node_id = self.ep_comm_initializer.config.node_id

        dp = model_config.data_parallel_degree
        use_dp = dp > 1

        if use_dp:
            logger.info(
                "Qwen3: data_parallel_degree=%d, ep_size=%s. Using "
                "DP-attention + EP-MoE strategy.",
                dp,
                self.pipeline_config.runtime.ep_size,
            )

        nn_model = Qwen3(model_config, ep_manager=ep_manager)

        graph_inputs = nn_model.input_types(self.kv_params)

        nn_model.load_state_dict(
            state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
            strict=(
                not getattr(
                    self.huggingface_config, "tie_word_embeddings", False
                )
            ),
        )

        self.state_dict = nn_model.state_dict()

        num_devices = len(self.devices)

        with Graph("qwen3", input_types=graph_inputs) as graph:
            if use_dp:
                (
                    tokens,
                    input_row_offsets,
                    return_n_logits,
                    host_input_row_offsets,
                    data_parallel_splits,
                    *variadic_args,
                ) = graph.inputs

                variadic_args_iter = iter(variadic_args)

                signal_buffers = [
                    next(variadic_args_iter).buffer for _ in range(num_devices)
                ]

                kv_input_count = len(
                    self.kv_params.get_symbolic_inputs().flatten()
                )
                kv_cache_inputs = [
                    next(variadic_args_iter) for _ in range(kv_input_count)
                ]
                kv_collections = self._unflatten_kv_inputs(kv_cache_inputs)

                ep_model_inputs = list(variadic_args_iter)
                if ep_manager is not None:
                    ep_manager.fetch_buffers(ep_model_inputs)

                outputs = nn_model(
                    tokens.tensor,
                    kv_collections,
                    return_n_logits.tensor,
                    input_row_offsets.tensor,
                    signal_buffers,
                    host_input_row_offsets=host_input_row_offsets.tensor,
                    data_parallel_splits=data_parallel_splits.tensor,
                )
            else:
                (
                    tokens,
                    input_row_offsets,
                    return_n_logits,
                    *variadic_args,
                ) = graph.inputs

                signal_buffers = [v.buffer for v in variadic_args[:num_devices]]
                kv_cache_inputs = variadic_args[num_devices:]
                kv_collections = self._unflatten_kv_inputs(kv_cache_inputs)

                outputs = nn_model(
                    tokens.tensor,
                    kv_collections,
                    return_n_logits.tensor,
                    input_row_offsets.tensor,
                    signal_buffers,
                )

            graph.output(*outputs)
            return graph

    @override
    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> Llama3Inputs | Qwen3Inputs:
        dp = self.pipeline_config.model.data_parallel_degree
        if dp <= 1:
            return super().prepare_initial_token_inputs(
                replica_batches, kv_cache_inputs, return_n_logits
            )

        if len(replica_batches) != dp:
            raise ValueError(
                "Number of replica batches must match data parallel degree"
            )

        context_batch = flatten2d(replica_batches)
        device0 = self.devices[0]
        pinned = not device0.is_host

        # Build tokens
        num_tokens = sum(ctx.tokens.active_length for ctx in context_batch)
        host_tokens: Buffer
        if pinned:
            host_tokens = DevicePinnedBuffer(
                shape=(num_tokens,), dtype=DType.int64, device=device0
            )
        else:
            host_tokens = Buffer(
                shape=(num_tokens,), dtype=DType.int64, device=device0
            )

        if context_batch:
            np.concatenate(
                [ctx.tokens.active for ctx in context_batch],
                out=host_tokens.to_numpy(),
            )
        tokens = host_tokens.to(device0)

        # Build input_row_offsets
        batch_size = len(context_batch)
        input_row_offsets_np = np.cumsum(
            [0] + [ctx.tokens.active_length for ctx in context_batch],
            dtype=np.uint32,
        )

        host_input_row_offsets = Buffer.from_numpy(input_row_offsets_np.copy())

        pinned_offsets: Buffer
        if pinned:
            pinned_offsets = DevicePinnedBuffer(
                shape=(batch_size + 1,), dtype=DType.uint32, device=device0
            )
        else:
            pinned_offsets = Buffer(
                shape=(batch_size + 1,), dtype=DType.uint32, device=device0
            )
        pinned_offsets.to_numpy()[:] = input_row_offsets_np
        device_input_row_offsets = pinned_offsets.to(device0)

        return_n_logits_tensor = Buffer.from_numpy(
            np.array([return_n_logits], dtype=np.int64)
        )

        data_parallel_splits = Buffer.from_numpy(
            compute_data_parallel_splits(replica_batches)
        )

        ep_inputs = (
            ()
            if self.ep_comm_initializer is None
            else tuple(self.ep_comm_initializer.model_inputs())
        )

        return Qwen3Inputs(
            tokens=tokens,
            input_row_offsets=device_input_row_offsets,
            return_n_logits=return_n_logits_tensor,
            host_input_row_offsets=host_input_row_offsets,
            data_parallel_splits=data_parallel_splits,
            signal_buffers=self.signal_buffers,
            kv_cache_inputs=kv_cache_inputs,
            ep_inputs=ep_inputs,
        )

    @override
    def prepare_next_token_inputs(
        self,
        next_tokens: Buffer,
        prev_model_inputs: ModelInputs,
    ) -> Llama3Inputs | Qwen3Inputs:
        if isinstance(prev_model_inputs, Qwen3Inputs):
            assert self._input_row_offsets_prealloc is not None
            row_offsets_size = prev_model_inputs.input_row_offsets.shape[0]
            next_row_offsets = self._input_row_offsets_prealloc[
                :row_offsets_size
            ]

            assert self._host_input_row_offsets_prealloc is not None
            next_host_offsets = self._host_input_row_offsets_prealloc[
                :row_offsets_size
            ]

            return Qwen3Inputs(
                tokens=next_tokens,
                input_row_offsets=next_row_offsets,
                return_n_logits=prev_model_inputs.return_n_logits,
                host_input_row_offsets=next_host_offsets,
                data_parallel_splits=prev_model_inputs.data_parallel_splits,
                signal_buffers=self.signal_buffers,
                kv_cache_inputs=prev_model_inputs.kv_cache_inputs,
                ep_inputs=prev_model_inputs.ep_inputs,
            )

        return super().prepare_next_token_inputs(next_tokens, prev_model_inputs)
