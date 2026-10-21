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
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import numpy as np
from max import tree
from max._core.engine import Model
from max.driver import Buffer, is_virtual_device_mode
from max.dtype import DType
from max.engine import InferenceSession
from max.nn.comm.ep import EPCommInitializer, EPConfig
from max.nn.comm.ep.ep_config import calculate_ep_max_tokens_per_rank
from max.nn.comm.ep.ep_manager import EPBatchManager
from max.pipelines.lib.config.model_config import (
    _select_quantization_encoding,
)
from max.pipelines.lib.graph_specs import GraphRole, GraphSpec
from max.pipelines.lib.interfaces import AlwaysSignalBuffersMixin
from max.pipelines.modeling.config_enums import supported_encoding_dtype
from max.pipelines.weights.quant import parse_quant_config
from typing_extensions import override

from ..llama3.model import Llama3Inputs, LlamaModelBase
from .batch_processor import Qwen3BatchProcessor
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
            *(
                tree.leaves(self.kv_cache_inputs)
                if self.kv_cache_inputs
                else ()
            ),
            *self.ep_inputs,
        )


class Qwen3Model(AlwaysSignalBuffersMixin, LlamaModelBase):
    """Qwen3 pipeline model supporting single-GPU, TP, and DP+EP inference.

    Uses AlwaysSignalBuffersMixin since VocabParallelEmbedding and
    ColumnParallelLinear always require signal buffers for allreduce.
    """

    model_config_cls: ClassVar[type[Any]] = Qwen3Config
    batch_processor_cls: ClassVar[type[Qwen3BatchProcessor]] = (
        Qwen3BatchProcessor
    )

    model: Model
    norm_method: Literal["rms_norm", "layer_norm"] = "rms_norm"
    attention_bias: bool = False
    state_dict: dict[str, Any]

    def _create_ep_config(
        self,
        state_dict: dict[str, Any] | None = None,
    ) -> EPConfig | None:
        """Create EP config from pipeline settings.

        Args:
            state_dict: Model weight state dict, required for non-bfloat16
                dispatch dtypes (such as FP8) to parse the dispatch quantization
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

        encoding = _select_quantization_encoding(
            self.pipeline_config.model, Qwen3Config.DEFAULT_ENCODING
        )
        dispatch_dtype = supported_encoding_dtype(encoding)

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
    def _create_model_config(self, state_dict: dict[str, Any]) -> Qwen3Config:
        model_config = Qwen3Config.initialize_from_config(
            self.pipeline_config,
            self.huggingface_config,
            max_seq_len=self.max_seq_len,
        )
        model_config.finalize(
            huggingface_config=self.huggingface_config,
            state_dict=state_dict,
            return_logits=self.return_logits,
            norm_method=self.norm_method,
            attention_bias=self.attention_bias,
        )
        # Set up EP config
        model_config.ep_config = self._create_ep_config(state_dict)
        return model_config

    @override
    def _init_distributed_runtime(
        self,
        session: InferenceSession,
        model_config: Qwen3Config,
    ) -> None:
        self.ep_comm_initializer = None
        ep_config = model_config.ep_config
        if ep_config is None:
            return
        if is_virtual_device_mode():
            return
        # Create EP infrastructure
        self.ep_comm_initializer = EPCommInitializer(ep_config)
        self.ep_comm_initializer.ep_init(session)
        ep_config.node_id = self.ep_comm_initializer.config.node_id

    @override
    def graph_specs(self, model_config: Qwen3Config) -> dict[str, GraphSpec]:
        """Returns the decoder: single-device, tensor-parallel, or
        data-parallel attention with expert-parallel MoE."""
        ep_config = model_config.ep_config
        # Create EP batch manager for graph buffer wiring (runtime init is
        # in `_init_distributed_runtime`).
        ep_manager: EPBatchManager | None = None
        if ep_config is not None:
            ep_manager = EPBatchManager(ep_config)

        dp = model_config.data_parallel_degree
        if dp > 1:
            logger.info(
                "Qwen3: data_parallel_degree=%d, ep_size=%s. Using "
                "DP-attention + EP-MoE strategy.",
                dp,
                self.pipeline_config.runtime.ep_size,
            )

        nn_model = Qwen3(model_config, ep_manager=ep_manager)
        graph_inputs = nn_model.input_types(self.kv_params)

        nn_model.load_state_dict(
            self.state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
            strict=(
                not getattr(
                    self.huggingface_config, "tie_word_embeddings", False
                )
            ),
        )
        return {
            GraphRole.DECODER: GraphSpec(
                nn_model,
                "qwen3",
                graph_inputs,
                weights=nn_model.state_dict(),
            )
        }
