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
"""Implements the DeepseekV3.2 PipelineModel."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, ClassVar

from max.driver import is_virtual_device_mode
from max.dtype import DType
from max.engine import InferenceSession
from max.graph.weights import WeightData
from max.nn.legacy.comm.ep import EPConfig
from max.nn.legacy.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.lib import GraphRole, GraphSpec, PipelineConfig
from max.pipelines.weights.quant import parse_quant_config
from transformers import AutoConfig
from typing_extensions import override

from ..deepseekV3.memory_planner import (
    _ep_max_rank_send_tokens_for_pipeline,
    ep_fuse_ffn_combine_send_for_pipeline,
)
from ..deepseekV3.model import DeepseekV3Model
from .deepseekV3_2 import DeepseekV3_2
from .model_config import DeepseekV3_2Config

logger = logging.getLogger("max.pipelines")


class DeepseekV3_2Model(DeepseekV3Model):
    """A DeepseekV3.2 model."""

    model_config_cls: ClassVar[type[Any]] = DeepseekV3_2Config

    @classmethod
    def _first_mla_layer(cls, huggingface_config: AutoConfig) -> int:
        """Index of the first layer that actually has multi-latent attention.

        Zero for an all-MLA decoder, which is every model on this path today.
        A hybrid schedule that interleaves another attention family overrides
        this: probing layer 0 for an MLA-only tensor raises ``KeyError`` when
        layer 0 belongs to the other family.
        """
        del huggingface_config
        return 0

    def _create_model_config(
        self, state_dict: dict[str, WeightData]
    ) -> DeepseekV3_2Config:
        """Create model configuration from huggingface config."""
        return self.finalized_config(
            self.pipeline_config,
            state_dict,
            max_seq_len=self.max_seq_len,
            max_batch_total_tokens=self.planned_max_batch_total_tokens,
            return_logits=self.return_logits,
            return_hidden_states=self.return_hidden_states,
            model_config=self.arch_config,
        )

    @override
    @classmethod
    def finalized_config(
        cls,
        pipeline_config: PipelineConfig,
        state_dict: Mapping[str, Any],
        *,
        max_seq_len: int,
        max_batch_total_tokens: int | None,
        return_logits: ReturnLogits,
        return_hidden_states: ReturnHiddenStates,
        model_config: DeepseekV3_2Config | None = None,
    ) -> DeepseekV3_2Config:
        """Builds the model config, finalized against ``state_dict``.

        A caller without a pipeline model, such as a speculative pairing
        that builds the module alone, leaves ``model_config`` unset to
        initialize one from ``pipeline_config``.
        """
        config = pipeline_config.model.huggingface_config
        num_devices = len(pipeline_config.model.device_specs)
        if model_config is None:
            model_config = cls.model_config_cls.initialize(
                pipeline_config, max_seq_len=max_seq_len
            )
        assert isinstance(model_config, DeepseekV3_2Config)

        # data_parallel_degree controls the attention strategy:
        #   == num_devices  ->  DP attention  (each device owns a batch shard)
        #   == 1            ->  TP attention  (heads sharded, tokens replicated)
        data_parallel_degree = pipeline_config.model.data_parallel_degree
        # PipelineConfig would automatically resolve it if not set by user.
        assert max_batch_total_tokens is not None, "max_length must be set"

        if pipeline_config.runtime.pipeline_role == "prefill_only":
            graph_mode = "prefill"
        elif pipeline_config.runtime.pipeline_role == "decode_only":
            graph_mode = "decode"
        else:
            graph_mode = "auto"

        dtype = model_config.dtype
        if dtype in (DType.float8_e4m3fn, DType.uint8, DType.float4_e2m1fn):
            quant_config = parse_quant_config(config, state_dict, dtype)
        else:
            quant_config = None

        ep_size = pipeline_config.runtime.ep_size
        if ep_size == 1:
            ep_config = None
        else:
            if ep_size % num_devices != 0:
                raise ValueError(
                    f"ep_size={ep_size} is not divisible by the number of GPUs"
                    f" on this node ({num_devices}). ep_size must equal"
                    f" n_gpus_per_node * n_nodes. For a single-node deployment"
                    f" set ep_size={num_devices}."
                )
            n_nodes = ep_size // num_devices

            ep_max_rank_send_tokens = _ep_max_rank_send_tokens_for_pipeline(
                pipeline_config
            )

            ep_kwargs: dict[str, Any] = dict(
                dispatch_dtype=dtype,
                combine_dtype=DType.bfloat16,
                hidden_size=config.hidden_size,
                top_k=config.num_experts_per_tok,
                n_experts=config.n_routed_experts,
                moe_dim=config.moe_intermediate_size,
                max_tokens_per_rank=ep_max_rank_send_tokens,
                n_gpus_per_node=num_devices,
                n_nodes=n_nodes,
                dispatch_quant_config=None,
                use_allreduce=pipeline_config.runtime.ep_use_allreduce,
                fuse_ffn_combine_send=ep_fuse_ffn_combine_send_for_pipeline(
                    pipeline_config, config
                ),
            )

            if config.n_shared_experts == 1:
                # Fuse into EP dispatch only when shared experts use the same
                # quantized layout as routed experts (modelopt ``*shared_experts*``
                # ignore leaves them bf16 → separate unfused path).
                if quant_config is None:
                    ep_kwargs["fused_shared_expert"] = True
                else:
                    ep_kwargs["fused_shared_expert"] = (
                        quant_config.shared_experts_dtype(DType.bfloat16)
                        == dtype
                    )

            if quant_config is not None:
                ep_kwargs["dispatch_quant_config"] = quant_config

            ep_config = EPConfig(**ep_kwargs)

        first_mla = cls._first_mla_layer(config)
        norm_dtype = state_dict[
            f"layers.{first_mla}.self_attn.kv_a_layernorm.weight"
        ].dtype

        if config.topk_method == "noaux_tc":
            correction_bias_key = None
            for k in state_dict:
                if k.endswith("e_score_correction_bias"):
                    correction_bias_key = k
                    break
            if correction_bias_key is None:
                raise KeyError("Expected e_score_correction_bias in state_dict")
            correction_bias_dtype = state_dict[correction_bias_key].dtype
        else:
            correction_bias_dtype = None

        # Finalize config with state_dict-dependent parameters
        model_config.norm_dtype = norm_dtype
        model_config.correction_bias_dtype = correction_bias_dtype
        model_config.max_batch_context_length = max_batch_total_tokens
        model_config.quant_config = quant_config
        model_config.ep_config = ep_config
        model_config.graph_mode = graph_mode
        model_config.data_parallel_degree = data_parallel_degree
        model_config.return_logits = return_logits
        model_config.return_hidden_states = return_hidden_states

        if ep_size > 1:
            attn_strategy = "TP" if data_parallel_degree == 1 else "DP"
            logger.info(
                f"DeepSeekV3.2: data_parallel_degree={data_parallel_degree},"
                f" ep_size={ep_size}. Use {attn_strategy}-attention + EP-MoE"
                f" strategy."
            )

        return model_config

    @override
    def _init_distributed_runtime(
        self,
        session: InferenceSession,
        model_config: DeepseekV3_2Config,
    ) -> None:
        self.ep_comm_initializer = None
        if model_config.ep_config is None or is_virtual_device_mode():
            return
        self.ep_comm_initializer = self._init_ep_comm(
            session, model_config.ep_config
        )
        if model_config.ep_config.node_id == -1:
            raise ValueError(
                "EP node ID is not set. Please check if the EP initialization is successful."
            )

    @override
    def graph_specs(
        self, model_config: DeepseekV3_2Config
    ) -> dict[str, GraphSpec]:
        """Returns the decoder graph."""
        nn_model = DeepseekV3_2(model_config)
        nn_model.load_state_dict(
            self._weights_dict, weight_alignment=1, strict=True
        )
        return {
            GraphRole.DECODER: GraphSpec(
                nn_model,
                "deepseekV3_2_graph",
                self._bind_graph_inputs(
                    nn_model.input_tree(model_config.kv_params)
                ),
                weights=nn_model.state_dict(),
            )
        }
