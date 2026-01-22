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
"""Implements the DeepseekV3.2 PipelineModel."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph
from max.graph.weights import WeightData
from max.nn.legacy.comm.ep import EPCommInitializer, EPConfig
from max.pipelines.lib import (
    CompilationTimer,
)
from max.pipelines.lib.config_enums import PipelineRole
from max.pipelines.lib.float8 import parse_float8_config
from typing_extensions import override

from ..deepseekV3.model import DeepseekV3Model
from .deepseekV3_2 import DeepseekV3_2
from .model_config import DeepseekV3_2Config

logger = logging.getLogger("max.pipelines")


class DeepseekV3_2Model(DeepseekV3Model):
    """A DeepseekV3.2 model."""

    def _create_model_config(
        self, state_dict: dict[str, WeightData]
    ) -> DeepseekV3_2Config:
        """Create model configuration from huggingface config."""
        config = self.huggingface_config

        max_batch_total_tokens = self.pipeline_config.max_batch_total_tokens
        # PipelineConfig would automatically resolve it if not set by user.
        assert max_batch_total_tokens is not None, "max_length must be set"

        if self.pipeline_config.pipeline_role is PipelineRole.PrefillOnly:
            graph_mode = "prefill"
        elif self.pipeline_config.pipeline_role is PipelineRole.DecodeOnly:
            graph_mode = "decode"
        else:
            graph_mode = "auto"

        kv_params = DeepseekV3_2Config.construct_kv_params(
            huggingface_config=self.huggingface_config,
            pipeline_config=self.pipeline_config,
            devices=[DeviceRef.from_device(d) for d in self.devices],
            kv_cache_config=self.kv_cache_config,
            cache_dtype=self.encoding.cache_dtype,
        )

        dtype = self.encoding.dtype
        if dtype == DType.float8_e4m3fn:
            float8_config = parse_float8_config(config, state_dict, dtype)
        else:
            float8_config = None

        if self.pipeline_config.ep_size == 1:
            ep_config = None
        else:
            if self.pipeline_config.ep_size % len(self.devices) != 0:
                raise ValueError(
                    "If you are running with expert parallelism, ep_size must"
                    " be set to the total number of GPUs across nodes."
                )
            n_nodes = self.pipeline_config.ep_size // len(self.devices)
            ep_kwargs: dict[str, Any] = dict(
                dispatch_dtype=dtype,
                combine_dtype=DType.bfloat16,
                hidden_size=config.hidden_size,
                top_k=config.num_experts_per_tok,
                n_experts=config.n_routed_experts,
                max_tokens_per_rank=self.pipeline_config.max_batch_input_tokens,
                n_gpus_per_node=len(self.devices),
                n_nodes=n_nodes,
                dispatch_fp8_config=None,
            )

            if config.n_shared_experts == 1:
                # Only enable shared expert fusion if the shared expert is of
                # the same shape as routed experts.
                ep_kwargs["fused_shared_expert"] = True

            if float8_config is not None:
                ep_kwargs["dispatch_fp8_config"] = float8_config.input_scale

            ep_config = EPConfig(**ep_kwargs)

        norm_dtype = state_dict[
            "layers.0.self_attn.kv_a_layernorm.weight"
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
        return DeepseekV3_2Config(
            dtype=self.encoding.dtype,
            norm_dtype=norm_dtype,
            correction_bias_dtype=correction_bias_dtype,
            kv_params=kv_params,
            devices=[DeviceRef.from_device(dev) for dev in self.devices],
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            moe_intermediate_size=config.moe_intermediate_size,
            moe_layer_freq=config.moe_layer_freq,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            n_shared_experts=config.n_shared_experts,
            n_routed_experts=config.n_routed_experts,
            routed_scaling_factor=config.routed_scaling_factor,
            kv_lora_rank=config.kv_lora_rank,
            q_lora_rank=config.q_lora_rank,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            qk_nope_head_dim=config.qk_nope_head_dim,
            topk_method=config.topk_method,
            n_group=config.n_group,
            topk_group=config.topk_group,
            num_experts_per_tok=config.num_experts_per_tok,
            first_k_dense_replace=config.first_k_dense_replace,
            norm_topk_prob=config.norm_topk_prob,
            hidden_act=config.hidden_act,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=config.tie_word_embeddings,
            rope_theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
            rope_interleave=getattr(config, "rope_interleave", True),
            scoring_func=config.scoring_func,
            attention_bias=config.attention_bias,
            attention_dropout=config.attention_dropout,
            max_batch_context_length=max_batch_total_tokens,
            float8_config=float8_config,
            ep_config=ep_config,
            graph_mode=graph_mode,
            data_parallel_degree=self.pipeline_config.model.data_parallel_degree,
            use_subgraphs=self.pipeline_config.model.use_subgraphs,
            return_logits=self.return_logits,
            index_head_dim=config.index_head_dim,
            index_n_heads=config.index_n_heads,
            index_topk=config.index_topk,
        )

    @override
    def load_model(self, session: InferenceSession) -> Model:
        """Load the model with the given weights."""

        max_batch_size = self.pipeline_config.max_batch_size
        assert max_batch_size, "Expected max_batch_size to be set"

        # `_host_input_row_offsets_prealloc` tensor needs to reserve space for
        # `max_batch_size` of requests on each DP rank.
        dp_size = self.pipeline_config.model.data_parallel_degree
        max_batch_size *= dp_size

        self._host_input_row_offsets_prealloc = Buffer.from_numpy(
            np.arange(max_batch_size + 1, dtype=np.uint32)
        )
        self._device_input_row_offsets_prealloc = (
            self._host_input_row_offsets_prealloc.to(self.devices[0])
        )

        # create batch context lengths tensor for each device
        self._batch_context_lengths_prealloc_cpu = [
            Buffer.zeros(shape=[1], dtype=DType.int32)
            for _ in range(len(self.devices))
        ]

        timer = CompilationTimer("model")
        if self.adapter:
            state_dict = self.adapter(
                dict(self.weights.items()),
                huggingface_config=self.huggingface_config,
                pipeline_config=self.pipeline_config,
            )
        else:
            state_dict = {
                key: value.data() for key, value in self.weights.items()
            }
        # Create the model
        config = self._create_model_config(state_dict)

        self.ep_comm_initializer: EPCommInitializer | None = None
        if config.ep_config is not None:
            self.ep_comm_initializer = EPCommInitializer(config.ep_config)
            self.ep_comm_initializer.ep_init(session)
            if config.ep_config.node_id == -1:
                raise ValueError(
                    "EP node ID is not set. Please check if the EP initialization is successful."
                )

        nn_model = DeepseekV3_2(config)
        nn_model.load_state_dict(state_dict, weight_alignment=1, strict=True)

        # Create the graph
        with Graph(
            "deepseekV3_2_graph",
            input_types=nn_model.input_types(self.kv_params),
        ) as graph:
            (
                tokens,
                devices_input_row_offsets,
                host_input_row_offsets,
                return_n_logits,
                data_parallel_splits,
                *variadic_args,
            ) = graph.inputs

            variadic_args_iter = iter(variadic_args)
            # Multi-GPU passes a signal buffer per device: unmarshal these.
            signal_buffers = [
                next(variadic_args_iter).buffer
                for _ in range(len(self.devices))
            ]

            # Unmarshal the KV cache arguments.
            fetch_types = self.kv_params.get_symbolic_inputs()[0]
            len_of_kv_inputs = len(list(fetch_types)) * len(self.devices)
            kv_caches_per_dev = self._unflatten_kv_inputs(
                [next(variadic_args_iter) for _ in range(len_of_kv_inputs)]
            )

            # Unmarshal the batch context lengths
            batch_context_lengths = [
                next(variadic_args_iter).tensor
                for _ in range(len(self.devices))
            ]

            # all remaining arguments are for EP inputs
            ep_model_inputs = list(variadic_args_iter)

            outputs = nn_model(
                tokens.tensor,
                signal_buffers,
                kv_caches_per_dev,
                return_n_logits.tensor,
                devices_input_row_offsets.tensor,
                host_input_row_offsets.tensor,
                data_parallel_splits.tensor,
                batch_context_lengths,
                ep_model_inputs,
            )

            graph.output(*outputs)

        timer.mark_build_complete()
        model = session.load(graph, weights_registry=nn_model.state_dict())
        timer.done()

        return model
