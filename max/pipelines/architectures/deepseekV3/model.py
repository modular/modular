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
from collections.abc import Sequence

import numpy as np
from max.driver import Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph
from max.graph.weights import WeightData
from max.nn import Signals
from max.nn.float8_config import parse_float8_config
from max.nn.kv_cache import (
    KVCacheInputs,
    MultiPagedKVCacheManager,
    PagedKVCacheManager,
    load_kv_manager,
)
from max.pipelines.core import TextContext
from max.pipelines.lib import ModelInputs, ModelOutputs
from max.pipelines.lib.config_enums import PipelineRole
from typing_extensions import override

from ..deepseekV2.model import DeepseekV2Inputs, DeepseekV2Model
from .deepseekV3 import DeepseekV3
from .model_config import DeepseekV3Config

logger = logging.getLogger("max.pipelines")


class DeepseekV3Inputs(DeepseekV2Inputs):
    """A class representing inputs for the DeepseekV3 model."""

    data_parallel_splits: Tensor
    """Tensor containing the data parallel splits for the MLA layer."""

    def __init__(
        self,
        tokens: Tensor,
        input_row_offsets: Tensor,
        signal_buffers: list[Tensor],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: Tensor | None = None,
        data_parallel_splits: Tensor | None = None,
    ) -> None:
        if data_parallel_splits is None:
            raise ValueError("data_parallel_splits must be provided")
        self.data_parallel_splits = data_parallel_splits
        super().__init__(
            tokens,
            input_row_offsets,
            signal_buffers,
            kv_cache_inputs,
            return_n_logits,
        )


class DeepseekV3Model(DeepseekV2Model):
    """A DeepseekV3 model."""

    def _create_model_config(
        self, state_dict: dict[str, WeightData]
    ) -> DeepseekV3Config:
        """Create model configuration from huggingface config."""
        config = self.huggingface_config

        if self.pipeline_config.pipeline_role is PipelineRole.PrefillOnly:
            graph_mode = "prefill"
        elif self.pipeline_config.pipeline_role is PipelineRole.DecodeOnly:
            graph_mode = "decode"
        else:
            graph_mode = "auto"

        kv_params = DeepseekV3Config.get_kv_params(
            huggingface_config=self.huggingface_config,
            n_devices=len(self.devices),
            kv_cache_config=self.kv_cache_config,
            cache_dtype=self.encoding.cache_dtype,
        )

        dtype = self.encoding.dtype
        if dtype == DType.float8_e4m3fn:
            float8_config = parse_float8_config(config, state_dict, dtype)
        else:
            float8_config = None

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
        return DeepseekV3Config(
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
            rope_interleave=config.rope_interleave,
            scoring_func=config.scoring_func,
            attention_bias=config.attention_bias,
            attention_dropout=config.attention_dropout,
            float8_config=float8_config,
            graph_mode=graph_mode,
            data_parallel_degree=self.pipeline_config.model_config.data_parallel_degree,
            use_subgraphs=self.pipeline_config.model_config.use_subgraphs,
        )

    @override
    def load_model(self, session: InferenceSession) -> Model:
        """Load the model with the given weights."""

        max_batch_size = self.pipeline_config.max_batch_size
        assert max_batch_size, "Expected max_batch_size to be set"
        self._input_row_offsets_prealloc_cpu = Tensor.from_numpy(
            np.arange(max_batch_size + 1, dtype=np.uint32)
        )

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
        nn_model = DeepseekV3(config)
        nn_model.load_state_dict(state_dict, weight_alignment=1, strict=True)

        # Create the graph
        with Graph(
            "deepseekV3_graph",
            input_types=nn_model.input_types(self.kv_manager),
        ) as graph:
            (
                tokens,
                input_row_offsets,
                return_n_logits,
                data_parallel_splits,
                *variadic_args,
            ) = graph.inputs
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
                data_parallel_splits.tensor,
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

    def execute(
        self,
        model_inputs: ModelInputs,
    ) -> ModelOutputs:
        assert isinstance(model_inputs, DeepseekV3Inputs)
        curr_kv_cache_inputs = model_inputs.kv_cache_inputs or ()

        model_outputs = self.model.execute(
            model_inputs.tokens,
            model_inputs.input_row_offsets,
            model_inputs.return_n_logits,
            model_inputs.data_parallel_splits,
            *model_inputs.signal_buffers,
            *curr_kv_cache_inputs,
        )
        if len(model_outputs) == 3:
            assert isinstance(model_outputs[0], Tensor)
            assert isinstance(model_outputs[1], Tensor)
            assert isinstance(model_outputs[2], Tensor)
            return ModelOutputs(
                next_token_logits=model_outputs[0],
                logits=model_outputs[1],
                logit_offsets=model_outputs[2],
            )
        else:
            assert isinstance(model_outputs[0], Tensor)
            return ModelOutputs(
                next_token_logits=model_outputs[0],
                logits=model_outputs[0],
            )

    def prepare_initial_token_inputs(
        self,
        context_batch: Sequence[TextContext],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> DeepseekV3Inputs:
        # Get input_row_offsets: start and end position of each batch in the
        # combined total_seq_len dimension.
        input_row_offsets = np.cumsum(
            [0] + [ctx.active_length for ctx in context_batch], dtype=np.uint32
        )

        # Create a ragged token vector of length: sum(len(t) for t in tokens).
        tokens = np.concatenate([ctx.next_tokens for ctx in context_batch])

        data_parallel_splits: Tensor
        if self.pipeline_config.model_config.data_parallel_degree > 1:
            assert isinstance(self.kv_manager, MultiPagedKVCacheManager)
            data_parallel_splits = self.kv_manager.get_data_parallel_splits(
                context_batch
            )
        else:
            data_parallel_splits = Tensor.from_numpy(
                np.array([0, len(context_batch)], dtype=np.int64)
            )

        return DeepseekV3Inputs(
            tokens=Tensor.from_numpy(tokens).to(self.devices[0]),
            input_row_offsets=Tensor.from_numpy(input_row_offsets),
            signal_buffers=self.signal_buffers,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=Tensor.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ).to(self.devices[0]),
            data_parallel_splits=data_parallel_splits,
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: ModelInputs,
    ) -> DeepseekV3Inputs:
        assert isinstance(prev_model_inputs, DeepseekV3Inputs)
        row_offsets_size = prev_model_inputs.input_row_offsets.shape[0]
        next_row_offsets = self._input_row_offsets_prealloc_cpu[
            :row_offsets_size
        ]
        return DeepseekV3Inputs(
            tokens=next_tokens,
            input_row_offsets=next_row_offsets,
            signal_buffers=self.signal_buffers,
            kv_cache_inputs=prev_model_inputs.kv_cache_inputs,
            return_n_logits=prev_model_inputs.return_n_logits,
            data_parallel_splits=prev_model_inputs.data_parallel_splits,
        )

    def load_kv_manager(
        self,
        session: InferenceSession,
        available_cache_memory: int,
    ) -> PagedKVCacheManager:
        return load_kv_manager(
            params=DeepseekV3Config.get_kv_params(
                huggingface_config=self.huggingface_config,
                n_devices=len(self.devices),
                kv_cache_config=self.kv_cache_config,
                cache_dtype=self.encoding.cache_dtype,
                data_parallel_degree=self.pipeline_config.model_config.data_parallel_degree,
            ),
            max_batch_size=self.pipeline_config.max_batch_size,
            max_seq_len=self.calculate_max_seq_len(
                self.pipeline_config, huggingface_config=self.huggingface_config
            ),
            num_layers=self.huggingface_config.num_hidden_layers,
            devices=self.devices,
            available_cache_memory=available_cache_memory,
            page_size=self.kv_cache_config.kv_cache_page_size,
            session=session,
        )
