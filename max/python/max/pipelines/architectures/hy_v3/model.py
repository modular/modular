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

from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from max._core.engine import Model
from max.driver import Buffer, is_virtual_device_mode
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph
from max.graph.weights import Weights, WeightsAdapter
from max.nn.comm.ep import EPCommInitializer, EPConfig
from max.nn.comm.ep.ep_config import calculate_ep_max_tokens_per_rank
from max.pipelines.architectures.llama3.model import (
    Llama3Inputs,
    LlamaModelBase,
)
from max.pipelines.lib import CompilationTimer
from max.pipelines.lib.interfaces import AlwaysSignalBuffersMixin
from max.pipelines.lib.utils import parse_state_dict_from_weights
from typing_extensions import override

from .batch_processor import HyV3BatchProcessor
from .hy_v3 import HYV3
from .model_config import HYV3Config


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
    batch_processor_cls: ClassVar[type[HyV3BatchProcessor]] = HyV3BatchProcessor

    model: Model
    norm_method: Literal["rms_norm"] | Literal["layer_norm"] = "rms_norm"
    attention_bias: bool = False
    state_dict: dict[str, Any]

    @override
    def load_model(self, session: InferenceSession) -> Model:
        with CompilationTimer("model") as timer:
            graph = self._build_graph(self.weights, self.adapter, session)
            timer.mark_build_complete()
            model = session.load(graph, weights_registry=self.state_dict)
        if self._batch_processor is not None:
            bind = getattr(
                self._batch_processor, "bind_ep_comm_initializer", None
            )
            if bind is not None:
                bind(self.ep_comm_initializer)
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

            num_kv_inputs = len(nn_model.kv_params.flattened_kv_inputs())
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
