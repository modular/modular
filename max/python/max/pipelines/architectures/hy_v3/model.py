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
"""Hy3-preview pipeline shell."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import numpy as np
from max._core.engine import Model
from max.driver import Buffer, DevicePinnedBuffer, is_virtual_device_mode
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph
from max.graph.weights import Weights, WeightsAdapter
from max.nn.comm.ep import EPCommInitializer, EPConfig
from max.nn.comm.ep.ep_config import (
    calculate_ep_max_tokens_per_rank,
    estimate_ep_memory_usage,
)
from max.pipelines.architectures.llama3.model import (
    Llama3Inputs,
    LlamaModelBase,
)
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    CompilationTimer,
    PipelineConfig,
)
from max.pipelines.lib.interfaces import AlwaysSignalBuffersMixin
from max.pipelines.lib.utils import (
    compute_data_parallel_splits,
    parse_state_dict_from_weights,
)
from max.pipelines.modeling.config_enums import supported_encoding_dtype
from max.support.algorithm import flatten2d
from transformers import AutoConfig
from typing_extensions import override

from .hy_v3 import HYV3
from .model_config import HYV3Config, hyv3_num_experts_from_config


@dataclass
class HYV3Inputs(Llama3Inputs):
    """Inputs with EP and DP support."""

    ep_inputs: tuple[Buffer, ...] = field(kw_only=True, default=())
    host_input_row_offsets: Buffer | None = field(kw_only=True, default=None)

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        data_parallel_splits = self.data_parallel_splits
        host_input_row_offsets = self.host_input_row_offsets
        dp_buffers: tuple[Buffer, ...] = ()
        if (
            isinstance(data_parallel_splits, Buffer)
            and host_input_row_offsets is not None
        ):
            dp_buffers = (data_parallel_splits, host_input_row_offsets)
        return (
            self.tokens,
            self.input_row_offsets,
            self.return_n_logits,
            *dp_buffers,
            *self.signal_buffers,
            *(
                self.kv_cache_inputs.flatten()
                if self.kv_cache_inputs is not None
                else ()
            ),
            *self.ep_inputs,
        )


class HYV3Model(AlwaysSignalBuffersMixin, LlamaModelBase):
    """Hy3-preview pipeline model."""

    model_config_cls: ClassVar[type[Any]] = HYV3Config

    model: Model
    norm_method: Literal["rms_norm"] | Literal["layer_norm"] = "rms_norm"
    attention_bias: bool = False
    state_dict: dict[str, Any]

    _GRAPH_CAPTURE_HEADROOM_BYTES_PER_DEVICE = 8 * 1024**3

    @classmethod
    def estimate_activation_memory(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        encoding = pipeline_config.model.quantization_encoding
        n_gpus_per_node = len(pipeline_config.model.device_specs)
        # Use moe_intermediate_size for EP buffer math, not the
        # dense-layer intermediate_size (the latter is ~9x larger on
        # Hy3 and would over-reserve).
        num_experts = hyv3_num_experts_from_config(huggingface_config)
        moe_dim = int(huggingface_config.moe_intermediate_size)
        hidden_size = int(huggingface_config.hidden_size)
        top_k = int(huggingface_config.num_experts_per_tok)

        ep_buffer_memory = 0
        moe_activation_memory = 0
        ep_size = pipeline_config.runtime.ep_size
        if ep_size > 1 and encoding is not None:
            ep_max_rank_send_tokens = calculate_ep_max_tokens_per_rank(
                max_batch_input_tokens=pipeline_config.runtime.max_batch_input_tokens,
                ep_size=ep_size,
                data_parallel_degree=pipeline_config.model.data_parallel_degree,
            )
            ep_dispatch_dtype = supported_encoding_dtype(encoding)

            max_recv_tokens_per_rank = ep_max_rank_send_tokens * min(
                num_experts,
                ep_size * top_k,
            )

            moe_activation_memory += (
                max_recv_tokens_per_rank
                * moe_dim
                * ep_dispatch_dtype.size_in_bytes
            )
            moe_activation_memory += (
                max_recv_tokens_per_rank
                * hidden_size
                * DType.bfloat16.size_in_bytes
            )
            moe_activation_memory += 256 * 1024 * 1024
            moe_activation_memory *= n_gpus_per_node

            n_nodes = max(ep_size // n_gpus_per_node, 1)
            per_device_ep_memory = estimate_ep_memory_usage(
                hidden_size=hidden_size,
                dispatch_dtype=ep_dispatch_dtype,
                combine_dtype=DType.bfloat16,
                max_tokens_per_rank=ep_max_rank_send_tokens,
                n_experts=num_experts,
                n_nodes=n_nodes,
                n_gpus_per_node=n_gpus_per_node,
                top_k=top_k,
            )
            ep_buffer_memory = per_device_ep_memory * n_gpus_per_node * 2

        activation_memory = moe_activation_memory + ep_buffer_memory

        graph_capture_headroom = 0
        if pipeline_config.runtime.device_graph_capture:
            graph_capture_headroom = (
                cls._GRAPH_CAPTURE_HEADROOM_BYTES_PER_DEVICE * n_gpus_per_node
            )
            activation_memory += graph_capture_headroom

        return activation_memory

    @override
    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: Any = None,
        return_n_logits: int = 1,
    ) -> HYV3Inputs:
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

        return HYV3Inputs(
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
    def load_model(self, session: InferenceSession) -> Model:
        assert self.pipeline_config.runtime.max_batch_size, (
            "Expected max_batch_size to be set"
        )
        self._input_row_offsets_prealloc = None
        if not is_virtual_device_mode():
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
        model_config = HYV3Config.initialize_from_config(
            self.pipeline_config, self.huggingface_config
        )
        model_config.finalize(
            huggingface_config=self.huggingface_config,
            state_dict=state_dict,
            return_logits=self.return_logits,
            norm_method=self.norm_method,
            attention_bias=self.attention_bias,
        )

        # Hy3 checkpoint: gate weight is BF16, correction bias is FP32.
        for k, v in state_dict.items():
            if k.endswith("mlp.gate.gate_score.weight"):
                model_config.gate_dtype = v.dtype
                break
        for k, v in state_dict.items():
            if k.endswith("mlp.gate.e_score_correction_bias"):
                model_config.correction_bias_dtype = v.dtype
                break

        # EP only matters multi-GPU (SHMEM kernels reject n_ranks=1).
        num_devices = len(self.devices)
        self.ep_comm_initializer: EPCommInitializer | None = None
        if num_devices > 1:
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
                dispatch_quant_config=None,
            )
            # Skip EP init in virtual device mode (compile-only): NVSHMEM
            # needs real GPUs, but keep ep_config for graph structure.
            if session is not None and not is_virtual_device_mode():
                self.ep_comm_initializer = EPCommInitializer(
                    model_config.ep_config
                )
                self.ep_comm_initializer.ep_init(session)
                model_config.ep_config.node_id = (
                    self.ep_comm_initializer.config.node_id
                )
                if model_config.ep_config.node_id == -1:
                    raise ValueError(
                        "EP node ID is not set. Please check if the EP "
                        "initialization is successful."
                    )
        else:
            model_config.ep_config = None

        nn_model = HYV3(model_config)
        graph_inputs = nn_model.input_types(self.kv_params)

        nn_model.load_state_dict(
            state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
            strict=True,
        )
        self.state_dict = nn_model.state_dict()

        with Graph("hy_v3", input_types=graph_inputs) as graph:
            inputs_iter = iter(graph.inputs)
            tokens = next(inputs_iter)
            input_row_offsets = next(inputs_iter)
            return_n_logits = next(inputs_iter)
            if model_config.data_parallel_degree > 1:
                data_parallel_splits = next(inputs_iter).tensor
                host_input_row_offsets = next(inputs_iter).tensor
            else:
                data_parallel_splits = None
                host_input_row_offsets = None

            signal_buffers = [
                next(inputs_iter).buffer for _ in range(num_devices)
            ]

            num_kv_inputs = len(
                nn_model.kv_params.get_symbolic_inputs().flatten()
            )
            kv_cache_inputs = [next(inputs_iter) for _ in range(num_kv_inputs)]
            kv_collections = self._unflatten_kv_inputs(kv_cache_inputs)

            ep_inputs = list(inputs_iter)

            outputs = nn_model(
                tokens.tensor,
                kv_collections,
                return_n_logits.tensor,
                input_row_offsets.tensor,
                signal_buffers,
                ep_inputs,  # type: ignore[arg-type]
                data_parallel_splits,
                host_input_row_offsets,
            )

            graph.output(*outputs)
            return graph
