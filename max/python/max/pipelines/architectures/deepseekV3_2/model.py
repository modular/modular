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
from collections.abc import Sequence
from typing import Any

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph
from max.graph.weights import WeightData
from max.interfaces.request import RequestID
from max.kv_cache import PagedKVCacheManager
from max.nn.comm.ep import EPCommInitializer, EPConfig
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheInputsSequence,
    KVCacheParamInterface,
    MultiKVCacheParams,
)
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    CompilationTimer,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
)
from max.pipelines.lib.config.config_enums import is_float4_encoding
from max.pipelines.lib.float8 import parse_float8_config
from transformers import AutoConfig
from typing_extensions import override

from ..deepseekV3.model import DeepseekV3Inputs, DeepseekV3Model
from .deepseekV3_2 import DeepseekV3_2
from .model_config import DeepseekV3_2Config

logger = logging.getLogger("max.pipelines")


class DeepseekV3_2Model(DeepseekV3Model):
    """A DeepseekV3.2 model."""

    # Set by pipeline for extra KV cache managers
    extra_kv_managers: list[PagedKVCacheManager]
    # Stored for step() call in execute()
    _current_batches: Sequence[Sequence[TextContext]]

    @property
    def indexer_kv_manager(self) -> PagedKVCacheManager:
        """Returns the indexer KV cache manager (the single extra manager)."""
        assert len(self.extra_kv_managers) == 1, (
            "Expected exactly one extra KV manager (indexer cache)"
        )
        return self.extra_kv_managers[0]

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParamInterface:
        encoding = pipeline_config.model.quantization_encoding
        if encoding is not None and is_float4_encoding(encoding):
            cache_dtype = DType.bfloat16
        return DeepseekV3_2Config.construct_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

    def _create_model_config(
        self, state_dict: dict[str, WeightData]
    ) -> DeepseekV3_2Config:
        """Create model configuration from huggingface config."""
        config = self.huggingface_config

        max_batch_total_tokens = self.pipeline_config.max_batch_total_tokens
        # PipelineConfig would automatically resolve it if not set by user.
        assert max_batch_total_tokens is not None, "max_length must be set"

        if self.pipeline_config.pipeline_role == "prefill_only":
            graph_mode = "prefill"
        elif self.pipeline_config.pipeline_role == "decode_only":
            graph_mode = "decode"
        else:
            graph_mode = "auto"

        dtype = self.dtype
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
                ep_kwargs["dispatch_fp8_config"] = float8_config

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

        # Initialize config with parameters from pipeline_config
        model_config = DeepseekV3_2Config.initialize(self.pipeline_config)

        # Finalize config with state_dict-dependent parameters
        model_config.norm_dtype = norm_dtype
        model_config.correction_bias_dtype = correction_bias_dtype
        model_config.max_batch_context_length = max_batch_total_tokens
        model_config.float8_config = float8_config
        model_config.ep_config = ep_config
        model_config.graph_mode = graph_mode
        model_config.data_parallel_degree = (
            self.pipeline_config.model.data_parallel_degree
        )
        model_config.return_logits = self.return_logits

        return model_config

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
            assert isinstance(self.kv_params, MultiKVCacheParams)
            len_of_mla_kv_inputs = len(
                self.kv_params.get_symbolic_inputs()[0].flatten()
            )
            mla_kv_caches_per_dev = self._unflatten_kv_inputs(
                [next(variadic_args_iter) for _ in range(len_of_mla_kv_inputs)],
                self.kv_params.params[0],
            )

            len_of_indexer_kv_inputs = len(
                self.kv_params.get_symbolic_inputs()[1].flatten()
            )
            indexer_kv_caches_per_dev = self._unflatten_kv_inputs(
                [
                    next(variadic_args_iter)
                    for _ in range(len_of_indexer_kv_inputs)
                ],
                self.kv_params.params[1],
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
                mla_kv_caches_per_dev,
                indexer_kv_caches_per_dev,
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

    @override
    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> DeepseekV3Inputs:
        # Store batches for step() call in execute()
        self._current_batches = replica_batches

        # Get base inputs from parent (contains MLA KV cache inputs)
        model_inputs = super().prepare_initial_token_inputs(
            replica_batches, kv_cache_inputs, return_n_logits
        )

        # Claim and allocate blocks for indexer KV cache
        for replica_idx, batch in enumerate(replica_batches):
            for ctx in batch:
                if not self.indexer_kv_manager.contains(
                    ctx.request_id, replica_idx=replica_idx
                ):
                    self.indexer_kv_manager.claim(
                        ctx.request_id, replica_idx=replica_idx
                    )
                self.indexer_kv_manager.alloc(
                    ctx, replica_idx=replica_idx, num_steps=1
                )

        # Combine primary (MLA) and indexer KV cache inputs
        indexer_kv_inputs = self.indexer_kv_manager.runtime_inputs(
            replica_batches
        )
        assert model_inputs.kv_cache_inputs is not None
        combined_inputs = [model_inputs.kv_cache_inputs] + indexer_kv_inputs
        model_inputs.kv_cache_inputs = KVCacheInputsSequence(
            kv_cache_inputs=combined_inputs
        )

        return model_inputs

    @override
    def prepare_next_token_inputs(
        self,
        next_tokens: Buffer,
        prev_model_inputs: ModelInputs,
    ) -> DeepseekV3Inputs:
        model_inputs = super().prepare_next_token_inputs(
            next_tokens, prev_model_inputs
        )

        # Allocate blocks for indexer KV cache for next step
        for replica_idx, batch in enumerate(self._current_batches):
            for ctx in batch:
                self.indexer_kv_manager.alloc(
                    ctx, replica_idx=replica_idx, num_steps=1
                )

        # Get updated indexer KV inputs
        indexer_kv_inputs = self.indexer_kv_manager.runtime_inputs(
            self._current_batches
        )

        # Extract MLA inputs from previous inputs and combine with new indexer inputs
        assert isinstance(
            prev_model_inputs.kv_cache_inputs, KVCacheInputsSequence
        )
        prev_kv_inputs = prev_model_inputs.kv_cache_inputs.kv_cache_inputs
        # MLA inputs are at the beginning, indexer inputs are at the end
        num_indexer_inputs = len(indexer_kv_inputs)
        mla_kv_inputs = prev_kv_inputs[:-num_indexer_inputs]
        combined_inputs = list(mla_kv_inputs) + indexer_kv_inputs
        model_inputs.kv_cache_inputs = KVCacheInputsSequence(
            kv_cache_inputs=combined_inputs
        )

        return model_inputs

    @override
    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        outputs = super().execute(model_inputs)

        # Step indexer KV manager (commit new tokens)
        self.indexer_kv_manager.step(self._current_batches)

        return outputs

    def release(self, request_id: RequestID) -> None:
        """Release indexer KV cache resources for the request."""
        for replica_idx in range(len(self.devices)):
            if self.indexer_kv_manager.contains(request_id, replica_idx):
                self.indexer_kv_manager.release(request_id, replica_idx)
