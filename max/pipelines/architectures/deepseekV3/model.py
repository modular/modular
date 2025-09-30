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
"""Implements the DeepseekV3 nn.model."""

from __future__ import annotations

import logging
import time

from max.driver import Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph
from max.nn import Signals
from typing_extensions import override

from ..deepseekV2.model import DeepseekV2Model
from .deepseekV3 import DeepseekV3
from .model_config import DeepseekV3Config

logger = logging.getLogger("max.pipelines")


class DeepseekV3Model(DeepseekV2Model):
    """A DeepseekV3 model."""

    def _create_model_config(self) -> DeepseekV3Config:
        """Create model configuration from huggingface config."""
        config = self.huggingface_config

        kv_params = DeepseekV3Config.get_kv_params(
            huggingface_config=self.huggingface_config,
            n_devices=len(self.devices),
            kv_cache_config=self.kv_cache_config,
            cache_dtype=self.encoding.cache_dtype,
        )

        return DeepseekV3Config(
            dtype=self.encoding.dtype,
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
            rope_interleave=config.rope_interleave,
            scoring_func=config.scoring_func,
            attention_bias=config.attention_bias,
            attention_dropout=config.attention_dropout,
        )

    @override
    def load_model(self, session: InferenceSession) -> Model:
        """Load the model with the given weights."""

        # Override signal buffers from DeepSeekV2 model, because this model
        # always requires the signal buffer input. This can be removed once
        # we delete non-distributed Deepseek V2 (the distributed version
        # should already be able to handle a single device).
        self.signal_buffers = [
            Tensor.zeros(
                shape=(Signals.NUM_BYTES,), dtype=DType.uint8, device=dev
            )
            for dev in self.devices
        ]

        logger.info("Building DeepseekV3 model...")
        before = time.perf_counter()
        # Create the model
        config = self._create_model_config()
        nn_model = DeepseekV3(config)

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
        nn_model.load_state_dict(state_dict, weight_alignment=1, strict=True)

        # Create the graph
        with Graph(
            "deepseekV3_graph",
            input_types=nn_model.input_types(self.kv_manager),
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

            outputs = nn_model(
                tokens.tensor,
                signal_buffers,
                kv_caches_per_dev,
                return_n_logits.tensor,
                input_row_offsets.tensor,
            )

            graph.output(*outputs)
        after_build = time.perf_counter()
        logger.info(
            f"Building graph took {after_build - before:.6f} seconds. Compiling..."
        )

        # Compile the graph
        before_compile = time.perf_counter()

        model = session.load(graph, weights_registry=nn_model.state_dict())
        after = time.perf_counter()

        logger.info(
            f"Compiling model took {after - before_compile:.6f} seconds"
        )

        load_time = after - before
        logging.info(f"DeepseekV3 model loaded in {load_time:.6f} seconds")
        return model
