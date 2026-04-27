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
from max.driver import Buffer, DevicePinnedBuffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph
from max.graph.weights import Weights, WeightsAdapter
from max.nn.comm.ep import EPCommInitializer, EPConfig
from max.nn.comm.ep.ep_config import calculate_ep_max_tokens_per_rank
from max.nn.kv_cache import KVCacheParams
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    CompilationTimer,
    KVCacheConfig,
    PipelineConfig,
)
from max.pipelines.lib.interfaces import AlwaysSignalBuffersMixin
from max.pipelines.lib.interfaces.pipeline_model import ModelInputs
from max.pipelines.lib.utils import (
    compute_data_parallel_splits,
    parse_state_dict_from_weights,
)
from max.support.algorithm import flatten2d
from transformers import AutoConfig
from typing_extensions import override

from ..llama3.model import Llama3Inputs, LlamaModelBase
from .minimax_m2 import MiniMaxM2
from .model_config import MiniMaxM2Config

logger = logging.getLogger("max.pipelines")


@dataclass
class MiniMaxM2Inputs(Llama3Inputs):
    """Inputs for MiniMax-M2 with EP and DP support."""

    ep_inputs: tuple[Buffer, ...] = field(kw_only=True, default=())
    host_input_row_offsets: Buffer | None = field(kw_only=True, default=None)

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        data_parallel_splits = self.data_parallel_splits
        host_input_row_offsets = self.host_input_row_offsets
        assert isinstance(data_parallel_splits, Buffer)
        assert host_input_row_offsets is not None
        return (
            self.tokens,
            self.input_row_offsets,
            self.return_n_logits,
            data_parallel_splits,
            host_input_row_offsets,
            *self.signal_buffers,
            *(
                self.kv_cache_inputs.flatten()
                if self.kv_cache_inputs is not None
                else ()
            ),
            *self.ep_inputs,
        )


class MiniMaxM2Model(AlwaysSignalBuffersMixin, LlamaModelBase):
    """MiniMax-M2 pipeline model for text generation.

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
        return MiniMaxM2Config.construct_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

    @override
    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: Any = None,
        return_n_logits: int = 1,
    ) -> MiniMaxM2Inputs:
        dp = self.pipeline_config.model.data_parallel_degree
        if len(replica_batches) != dp:
            raise ValueError(
                "Number of replica batches must match data parallel degree"
            )

        context_batch = flatten2d(replica_batches)
        device0 = self.devices[0]
        pinned = not device0.is_host

        batch_size = len(context_batch)
        total_seq_len = sum(ctx.tokens.active_length for ctx in context_batch)
        buffer_key = (batch_size, total_seq_len)
        buffers = self._execution_input_buffers.get(buffer_key)
        if buffers is None:
            host_tokens: Buffer
            if pinned:
                host_tokens = DevicePinnedBuffer(
                    dtype=DType.int64,
                    shape=(total_seq_len,),
                    device=device0,
                )
            else:
                host_tokens = Buffer(
                    shape=(total_seq_len,),
                    dtype=DType.int64,
                    device=device0,
                )

            host_row_offsets: Buffer
            if pinned:
                host_row_offsets = DevicePinnedBuffer(
                    dtype=DType.uint32,
                    shape=(batch_size + 1,),
                    device=device0,
                )
            else:
                host_row_offsets = Buffer(
                    shape=(batch_size + 1,),
                    dtype=DType.uint32,
                    device=device0,
                )

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

        np.cumsum(
            [0] + [ctx.tokens.active_length for ctx in context_batch],
            dtype=np.uint32,
            out=host_row_offsets.to_numpy(),
        )

        return_n_logits_tensor = Buffer.from_numpy(
            np.array([return_n_logits], dtype=np.int64)
        )

        tokens_np = host_tokens.to_numpy()
        if context_batch:
            np.concatenate(
                [ctx.tokens.active for ctx in context_batch],
                out=tokens_np,
            )
        device_tokens.inplace_copy_from(host_tokens)
        device_row_offsets.inplace_copy_from(host_row_offsets)

        if dp > 1:
            data_parallel_splits = Buffer.from_numpy(
                compute_data_parallel_splits(replica_batches)
            )
        else:
            data_parallel_splits = None

        ep_inputs = (
            ()
            if self.ep_comm_initializer is None
            else tuple(self.ep_comm_initializer.model_inputs())
        )

        return MiniMaxM2Inputs(
            tokens=device_tokens,
            input_row_offsets=device_row_offsets,
            signal_buffers=self.signal_buffers,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=return_n_logits_tensor,
            data_parallel_splits=data_parallel_splits,
            ep_inputs=ep_inputs,
            host_input_row_offsets=host_row_offsets,
        )

    @override
    def prepare_next_token_inputs(
        self,
        next_tokens: Buffer,
        prev_model_inputs: ModelInputs,
    ) -> MiniMaxM2Inputs:
        assert isinstance(prev_model_inputs, MiniMaxM2Inputs)
        llama_inputs = super().prepare_next_token_inputs(
            next_tokens, prev_model_inputs
        )
        host_input_row_offsets = Buffer.from_numpy(
            np.arange(llama_inputs.input_row_offsets.shape[0], dtype=np.uint32)
        )
        return MiniMaxM2Inputs(
            tokens=llama_inputs.tokens,
            input_row_offsets=llama_inputs.input_row_offsets,
            signal_buffers=llama_inputs.signal_buffers,
            kv_cache_inputs=llama_inputs.kv_cache_inputs,
            return_n_logits=llama_inputs.return_n_logits,
            data_parallel_splits=llama_inputs.data_parallel_splits,
            ep_inputs=prev_model_inputs.ep_inputs,
            host_input_row_offsets=host_input_row_offsets,
        )

    @override
    def load_model(self, session: InferenceSession) -> Model:
        assert self.pipeline_config.runtime.max_batch_size, (
            "Expected max_batch_size to be set"
        )
        self._input_row_offsets_prealloc = Buffer.from_numpy(
            np.arange(
                self.pipeline_config.runtime.max_batch_size + 1,
                dtype=np.uint32,
            )
        ).to(self.devices[0])

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
        model_config = MiniMaxM2Config.initialize_from_config(
            self.pipeline_config, self.huggingface_config
        )
        model_config.finalize(
            huggingface_config=self.huggingface_config,
            state_dict=state_dict,
            return_logits=self.return_logits,
            norm_method=self.norm_method,
            attention_bias=self.attention_bias,
        )

        # Detect dtypes from state dict
        for k, v in state_dict.items():
            if k.endswith("mlp.gate.gate_score.weight"):
                model_config.gate_dtype = v.dtype
                break

        for k, v in state_dict.items():
            if k.endswith("mlp.gate.e_score_correction_bias"):
                model_config.correction_bias_dtype = v.dtype
                break

        for k, v in state_dict.items():
            if k.endswith("self_attn.q_proj.weight"):
                model_config.attn_dtype = v.dtype
                break

        # Create EP config for multi-GPU MoE
        # Each GPU sends the full batch through EP dispatch (attention is
        # replicated, so all GPUs have identical tokens).
        num_devices = len(self.devices)
        ep_size = num_devices
        ep_max_rank_send_tokens = calculate_ep_max_tokens_per_rank(
            max_batch_input_tokens=self.pipeline_config.runtime.max_batch_input_tokens,
            ep_size=ep_size,
            data_parallel_degree=self.pipeline_config.model.data_parallel_degree,
        )
        model_config.ep_config = EPConfig(
            dispatch_dtype=model_config.dtype,
            combine_dtype=DType.bfloat16,
            hidden_size=model_config.hidden_size,
            top_k=model_config.num_experts_per_tok,
            n_experts=model_config.num_local_experts,
            max_tokens_per_rank=ep_max_rank_send_tokens,
            n_gpus_per_node=num_devices,
            n_nodes=1,
            dispatch_quant_config=model_config.quant_config,
        )

        assert session is not None
        self.ep_comm_initializer = EPCommInitializer(model_config.ep_config)
        self.ep_comm_initializer.ep_init(session)
        logger.info(
            f"EP initialized: node_id={model_config.ep_config.node_id}, "
            f"n_gpus={model_config.ep_config.n_gpus_per_node}, "
            f"n_nodes={model_config.ep_config.n_nodes}, "
            f"n_experts={model_config.ep_config.n_experts}, "
            f"max_tokens_per_rank={model_config.ep_config.max_tokens_per_rank}"
        )

        nn_model = MiniMaxM2(model_config)

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

        with Graph("minimax_m2", input_types=graph_inputs) as graph:
            (
                tokens,
                input_row_offsets,
                return_n_logits,
                data_parallel_splits,
                host_input_row_offsets,
                *variadic_args,
            ) = graph.inputs

            variadic_args_iter = iter(variadic_args)

            # Unmarshal signal buffers
            signal_buffers = [
                next(variadic_args_iter).buffer for _ in range(num_devices)
            ]

            # Unmarshal KV cache inputs
            num_kv_inputs = len(
                nn_model.kv_params.get_symbolic_inputs().flatten()
            )
            kv_cache_inputs = [
                next(variadic_args_iter) for _ in range(num_kv_inputs)
            ]
            kv_collections = self._unflatten_kv_inputs(kv_cache_inputs)

            # Remaining args are EP inputs (empty list if no EP)
            ep_inputs = list(variadic_args_iter)

            outputs = nn_model(
                tokens.tensor,
                kv_collections,
                return_n_logits.tensor,
                input_row_offsets.tensor,
                signal_buffers,
                ep_inputs,  # type: ignore[arg-type]
                data_parallel_splits.tensor,
                host_input_row_offsets.tensor,
            )

            graph.output(*outputs)
            return graph
