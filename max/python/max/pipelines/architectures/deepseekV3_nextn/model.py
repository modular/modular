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
"""Implements the DeepseekV3 NextN pipeline model."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, fields

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph
from max.graph.weights import WeightData
from max.nn.legacy.comm.ep import EPCommInitializer
from max.nn.legacy.kv_cache import KVCacheInputs, KVCacheParams
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    AlwaysSignalBuffersMixin,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
)
from max.pipelines.lib.config_enums import PipelineRole
from max.pipelines.lib.utils import compute_data_parallel_splits
from max.support.algorithm import flatten2d
from max.support.human_readable_formatter import to_human_readable_bytes
from transformers import AutoConfig
from typing_extensions import override

from ..deepseekV2.model import DeepseekV2Model
from ..deepseekV3.model import (
    DeepseekV3Inputs,
    DeepseekV3Model,
    _choose_correct_data_parallel_degree,
)
from .deepseekV3_nextn import DeepseekV3NextN
from .model_config import DeepseekV3NextNConfig

logger = logging.getLogger("max.pipelines")


@dataclass
class DeepseekV3NextNInputs(DeepseekV3Inputs):
    """A class representing inputs for the DeepseekV3 NextN model.

    Inherits from DeepseekV3Inputs so that the target model's isinstance check
    passes during EAGLE verification (when draft_inputs is passed to the target).
    """

    hidden_states: Buffer | None = field(default=None, kw_only=True)


class DeepseekV3NextNModel(AlwaysSignalBuffersMixin, DeepseekV2Model):
    @classmethod
    def finalize_pipeline_config(cls, pipeline_config: PipelineConfig) -> None:
        """Finalizes the pipeline configuration."""
        _choose_correct_data_parallel_degree(
            pipeline_config, len(pipeline_config.model.device_specs)
        )

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return DeepseekV3NextNConfig.construct_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

    @classmethod
    def estimate_weights_size(cls, pipeline_config: PipelineConfig) -> int:
        """Calculates the estimated memory consumption of the DeepseekV3 NextN model.

        The NextN model consists of:
        - embed_tokens: VocabParallelEmbedding (shared in EAGLE/MTP mode)
        - lm_head: ColumnParallelLinear (shared in EAGLE/MTP mode)
        - enorm, hnorm, shared_head_norm: RMSNorm layers
        - eh_proj: Linear layer (hidden_size * 2 -> hidden_size)
        - decoder_layer: Single DeepseekV3DecoderLayer (MoE layer)

        Args:
            pipeline_config: The pipeline configuration containing model settings.

        Returns:
            Estimated weight memory in bytes.
        """
        draft_model_config = pipeline_config.draft_model
        assert draft_model_config is not None, (
            "draft_model must be set for NextN"
        )
        encoding = draft_model_config.quantization_encoding
        assert encoding is not None
        dtype_bytes = encoding.dtype.size_in_bytes
        config = draft_model_config.huggingface_config
        assert config is not None
        n_gpus_per_node = len(draft_model_config.device_specs)

        # Check if weights are shared (EAGLE or MTP mode)
        # In these modes, embed_tokens and lm_head are shared with the target model
        weights_are_shared = pipeline_config.speculative is not None and (
            pipeline_config.speculative.is_eagle()
            or pipeline_config.speculative.is_mtp()
        )

        total_size = 0

        # 1. Embedding and LM head (only if not shared)
        # These are always in BF16
        if not weights_are_shared:
            embedding_size = (
                config.vocab_size
                * config.hidden_size
                * DType.bfloat16.size_in_bytes
            )
            lm_head_size = embedding_size
            total_size += embedding_size + lm_head_size

        # 2. NextN-specific norms (enorm, hnorm, shared_head_norm) - always BF16
        norm_size = config.hidden_size * DType.bfloat16.size_in_bytes
        total_size += 3 * norm_size

        # 3. eh_proj: Linear(hidden_size * 2, hidden_size)
        eh_proj_size = config.hidden_size * 2 * config.hidden_size * dtype_bytes
        total_size += eh_proj_size

        # 4. Single decoder layer components

        # 4a. Layer norms (input_layernorm, post_attention_layernorm)
        total_size += 2 * norm_size

        # 4b. MLA attention weights
        num_heads = config.num_attention_heads
        # kv_a_proj: hidden_size -> kv_lora_rank + qk_rope_head_dim
        kv_a_proj_size = (
            config.hidden_size
            * (config.kv_lora_rank + config.qk_rope_head_dim)
            * dtype_bytes
        )
        # kv_a_layernorm: kv_lora_rank
        kv_a_layernorm_size = config.kv_lora_rank * DType.bfloat16.size_in_bytes
        # kv_b_proj: kv_lora_rank -> num_heads * (qk_nope_head_dim + v_head_dim)
        kv_b_proj_size = (
            config.kv_lora_rank
            * num_heads
            * (config.qk_nope_head_dim + config.v_head_dim)
            * dtype_bytes
        )
        # q_proj: hidden_size -> num_heads * (qk_nope_head_dim + qk_rope_head_dim)
        q_proj_size = (
            config.hidden_size
            * num_heads
            * (config.qk_nope_head_dim + config.qk_rope_head_dim)
            * dtype_bytes
        )
        # o_proj: num_heads * v_head_dim -> hidden_size
        o_proj_size = (
            num_heads * config.v_head_dim * config.hidden_size * dtype_bytes
        )

        attn_size = (
            kv_a_proj_size
            + kv_a_layernorm_size
            + kv_b_proj_size
            + q_proj_size
            + o_proj_size
        )
        total_size += attn_size

        # 4c. MoE weights (single layer)
        # Expert FFN: gate_proj, up_proj, down_proj
        expert_size = (
            config.moe_intermediate_size * config.hidden_size * 3 * dtype_bytes
        )
        routing_experts_size = config.n_routed_experts * expert_size
        shared_experts_size = config.n_shared_experts * expert_size

        # Router gate weights
        router_size = config.hidden_size * config.n_routed_experts * dtype_bytes
        total_size += router_size

        # Handle expert parallelism
        ep_size = max(pipeline_config.ep_size, 1)
        if ep_size == 1:
            total_size += routing_experts_size
        else:
            # Routing experts are sharded across nodes
            n_nodes = ep_size // n_gpus_per_node
            total_size += routing_experts_size // n_nodes

        # Shared experts are replicated on each device
        total_size += shared_experts_size * n_gpus_per_node

        logger.info(
            f"Estimated NextN weights size: {to_human_readable_bytes(total_size)}"
        )

        return total_size

    @classmethod
    def estimate_activation_memory(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        """Estimates the activation memory required for DeepseekV3 NextN model.

        This accounts for temporary memory buffers used during model execution,
        such as intermediate activations and working buffers. The NextN model
        has a single decoder layer, so activation memory is proportionally smaller.

        Args:
            pipeline_config: Pipeline configuration
            huggingface_config: HuggingFace model configuration (from draft_model)

        Returns:
            Estimated activation memory in bytes
        """
        draft_model_config = pipeline_config.draft_model
        assert draft_model_config is not None, (
            "draft_model must be set for NextN"
        )
        encoding = draft_model_config.quantization_encoding
        assert encoding is not None

        mla_activation_memory: int = 0
        moe_activation_memory: int = 0

        if pipeline_config.pipeline_role != PipelineRole.DecodeOnly:
            max_kv_length: int = 0

            if pipeline_config.max_batch_total_tokens is None:
                # If max_batch_total_tokens is not set, we use max_length.
                max_kv_length = pipeline_config.max_length or 0
            else:
                max_kv_length = pipeline_config.max_batch_total_tokens

            # MLA prefill activation for KV up-projection
            mla_activation_memory += (
                draft_model_config.data_parallel_degree
                * 2  # K and V
                * max_kv_length
                * huggingface_config.num_attention_heads
                * huggingface_config.qk_nope_head_dim
                * draft_model_config.kv_cache.cache_dtype.size_in_bytes
            )

        # Estimate activation memory during Expert Parallel MoE
        if pipeline_config.ep_size > 1:
            n_gpus_per_node = len(draft_model_config.device_specs)
            max_input_len_per_rank = pipeline_config.max_batch_input_tokens

            # Maximum tokens a rank may receive during all-to-all routing
            max_recv_tokens_per_rank = (
                max_input_len_per_rank * huggingface_config.n_routed_experts
            )

            # Input for grouped_matmul: [max_recv_tokens_per_rank, moe_intermediate_size]
            moe_activation_memory += (
                max_recv_tokens_per_rank
                * huggingface_config.moe_intermediate_size
                * encoding.dtype.size_in_bytes
            )

            # Output: [max_recv_tokens_per_rank, hidden_size]
            moe_activation_memory += (
                max_recv_tokens_per_rank
                * huggingface_config.hidden_size
                * DType.bfloat16.size_in_bytes  # output is always bfloat16
            )

            # Misc overhead (FP8 scalars, etc.)
            moe_activation_memory += 256 * 1024 * 1024

            moe_activation_memory *= n_gpus_per_node

        # MLA and MoE execute sequentially, so take the max
        activation_memory = max(mla_activation_memory, moe_activation_memory)

        if activation_memory != 0:
            logger.info(
                f"Estimated NextN activation memory: {to_human_readable_bytes(activation_memory)}"
            )

        return activation_memory

    def _create_model_config(
        self, state_dict: dict[str, WeightData]
    ) -> DeepseekV3NextNConfig:
        """Create model configuration from huggingface config.

        The weight adapter converts keys to NextN format (decoder_layer.*),
        but the base model's _create_model_config expects base format (layers.0.*).
        We temporarily remap the key so the base implementation can extract dtype info.
        """
        # Temporarily add the base format key for norm dtype extraction
        # NextN format: "decoder_layer.self_attn.kv_a_layernorm.weight"
        # Base format:  "layers.0.self_attn.kv_a_layernorm.weight"
        nextn_key = "decoder_layer.self_attn.kv_a_layernorm.weight"
        base_key = "layers.0.self_attn.kv_a_layernorm.weight"

        if nextn_key not in state_dict:
            raise KeyError(
                f"Expected NextN norm key '{nextn_key}' not found in state_dict. "
                f"This may indicate the weights are not in the correct NextN format. "
                f"Available keys sample: {list(state_dict.keys())[:10]}..."
            )

        state_dict[base_key] = state_dict[nextn_key]

        # Call DeepseekV3Model's _create_model_config to compute
        # state-dict-dependent fields (ep_config, float8_config, etc.)
        base_config = DeepseekV3Model._create_model_config(self, state_dict)  # type: ignore[arg-type]

        # Remove temporary key
        if base_key in state_dict and nextn_key in state_dict:
            del state_dict[base_key]

        # Build NextN config from the base config's fields, avoiding
        # asdict() which recursively converts nested dataclasses to dicts.
        model_config = DeepseekV3NextNConfig(
            **{
                f.name: getattr(base_config, f.name)
                for f in fields(base_config)
            }
        )
        return model_config

    @override
    def load_model(self, session: InferenceSession) -> Model:
        """Load the NextN model with the given weights."""

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

        self._batch_context_lengths_prealloc_cpu = [
            Buffer.zeros(shape=[1], dtype=DType.int32)
            for _ in range(len(self.devices))
        ]

        logger.info("Building DeepseekV3 NextN model...")
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

        config = self._create_model_config(state_dict)

        self.ep_comm_initializer: EPCommInitializer | None = None
        if config.ep_config is not None:
            self.ep_comm_initializer = EPCommInitializer(config.ep_config)
            self.ep_comm_initializer.ep_init(session)
            if config.ep_config.node_id == -1:
                raise ValueError(
                    "EP node ID is not set. Please check if the EP initialization is successful."
                )

        nn_model = DeepseekV3NextN(config)
        nn_model.load_state_dict(state_dict, weight_alignment=1, strict=True)

        num_devices = len(self.devices)

        with Graph(
            "deepseekV3_nextn_graph",
            input_types=nn_model.input_types(self.kv_params),
        ) as graph:
            graph_inputs_iter = iter(graph.inputs)

            tokens = next(graph_inputs_iter)

            hidden_states = [
                next(graph_inputs_iter).tensor for _ in range(num_devices)
            ]

            device_input_row_offsets = next(graph_inputs_iter)
            host_input_row_offsets = next(graph_inputs_iter)
            return_n_logits = next(graph_inputs_iter)
            data_parallel_splits = next(graph_inputs_iter)

            signal_buffers = [
                next(graph_inputs_iter).buffer for _ in range(num_devices)
            ]

            fetch_types = self.kv_params.get_symbolic_inputs()[0]
            len_of_kv_inputs = len(list(fetch_types)) * num_devices
            kv_caches_per_dev = self._unflatten_kv_inputs(
                [next(graph_inputs_iter) for _ in range(len_of_kv_inputs)]
            )

            batch_context_lengths = [
                next(graph_inputs_iter).tensor for _ in range(num_devices)
            ]

            ep_model_inputs = list(graph_inputs_iter)

            outputs = nn_model(
                tokens.tensor,
                hidden_states,
                signal_buffers,
                kv_caches_per_dev,
                return_n_logits.tensor,
                device_input_row_offsets.tensor,
                host_input_row_offsets.tensor,
                data_parallel_splits.tensor,
                batch_context_lengths,
                ep_model_inputs,
            )

            graph.output(*outputs)

        after_build = time.perf_counter()
        logger.info(
            f"Building graph took {after_build - before:.6f} seconds. Compiling..."
        )

        before_compile = time.perf_counter()
        model = session.load(graph, weights_registry=nn_model.state_dict())
        after = time.perf_counter()

        logger.info(
            f"Compiling model took {after - before_compile:.6f} seconds"
        )

        load_time = after - before
        logging.info(
            f"DeepseekV3 NextN model loaded in {load_time:.6f} seconds"
        )
        return model

    def execute(
        self,
        model_inputs: ModelInputs,
    ) -> ModelOutputs:
        assert isinstance(model_inputs, DeepseekV3NextNInputs)

        if model_inputs.hidden_states is None:
            raise ValueError(
                "hidden_states must be set before executing DeepSeekV3 NextN model. "
                "EAGLE pipeline should set this field after calling prepare_initial_token_inputs()."
            )

        num_devices = len(self.devices)

        hidden_states_list: list[Buffer]
        if isinstance(model_inputs.hidden_states, list):
            if len(model_inputs.hidden_states) != num_devices:
                raise ValueError(
                    f"hidden_states list length ({len(model_inputs.hidden_states)}) "
                    f"must match number of devices ({num_devices})"
                )
            hidden_states_list = model_inputs.hidden_states
        else:
            raise ValueError(
                "hidden_states must be a list of Buffers (one per device) "
                "for data parallel execution"
            )

        curr_kv_cache_inputs = model_inputs.kv_cache_inputs or ()
        ep_inputs = (
            ()
            if self.ep_comm_initializer is None
            else self.ep_comm_initializer.model_inputs()
        )

        model_outputs = self.model.execute(
            model_inputs.tokens,
            *hidden_states_list,
            model_inputs.input_row_offsets,
            model_inputs.host_input_row_offsets,
            model_inputs.return_n_logits,
            model_inputs.data_parallel_splits,
            *model_inputs.signal_buffers,
            *curr_kv_cache_inputs,
            *model_inputs.batch_context_lengths,
            *ep_inputs,
        )

        num_hidden_state_outputs = len(self.devices)
        if len(model_outputs) == 1 + num_hidden_state_outputs:
            assert isinstance(model_outputs[0], Buffer)
            output_hidden_states: list[Buffer] = []
            for i in range(num_hidden_state_outputs):
                hs = model_outputs[1 + i]
                assert isinstance(hs, Buffer)
                output_hidden_states.append(hs)
            return ModelOutputs(
                next_token_logits=model_outputs[0],
                logits=model_outputs[0],
                hidden_states=output_hidden_states,
            )
        else:
            assert isinstance(model_outputs[0], Buffer)
            return ModelOutputs(
                next_token_logits=model_outputs[0],
                logits=model_outputs[0],
            )

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
        hidden_states: Buffer | None = None,
    ) -> DeepseekV3NextNInputs:
        """Prepare initial inputs for the NextN model.

        Args:
            replica_batches: Batches of text contexts per replica
            kv_cache_inputs: KV cache inputs (optional)
            return_n_logits: Number of logits to return
            hidden_states: Hidden states from the base or draft model

        Returns:
            NextN model inputs
        """

        dp = self.pipeline_config.model.data_parallel_degree
        if len(replica_batches) != dp:
            raise ValueError(
                "Number of replica batches must match data parallel degree"
            )

        # If we are not in decode only mode, we need to create a list of
        # tensors containing the context length of each batch. Needed by MLA prefill.
        if self.pipeline_config.pipeline_role is not PipelineRole.DecodeOnly:
            for i, batch in enumerate(replica_batches):
                curr_length = sum([ctx.tokens.active_length for ctx in batch])
                self._batch_context_lengths_prealloc_cpu[i][0] = curr_length

            if dp != len(self.devices):
                assert dp == 1
                # Duplicate the batch context lengths for each device.
                for dev_idx in range(1, len(self.devices)):
                    self._batch_context_lengths_prealloc_cpu[dev_idx][0] = (
                        self._batch_context_lengths_prealloc_cpu[0][0].item()
                    )

        context_batch = flatten2d(replica_batches)
        if len(context_batch) == 0:
            tokens = Buffer(shape=[0], dtype=DType.int64).to(self.devices[0])
            host_input_row_offsets = Buffer.zeros(shape=[1], dtype=DType.uint32)
        else:
            # Create a ragged token vector of length: sum(len(t) for t in tokens).
            tokens = Buffer.from_numpy(
                np.concatenate([ctx.tokens.active for ctx in context_batch])
            ).to(self.devices[0])

            host_input_row_offsets = Buffer.from_numpy(
                np.cumsum(
                    [0] + [ctx.tokens.active_length for ctx in context_batch],
                    dtype=np.uint32,
                )
            )

        device_input_row_offsets = host_input_row_offsets.to(self.devices[0])

        data_parallel_splits = compute_data_parallel_splits(replica_batches)

        return DeepseekV3NextNInputs(
            tokens=tokens,
            hidden_states=hidden_states,
            input_row_offsets=device_input_row_offsets,
            host_input_row_offsets=host_input_row_offsets,
            signal_buffers=self.signal_buffers,
            batch_context_lengths=self._batch_context_lengths_prealloc_cpu,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            data_parallel_splits=Buffer.from_numpy(data_parallel_splits),
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Buffer,
        prev_model_inputs: ModelInputs,
        hidden_states: Buffer | None = None,
    ) -> DeepseekV3NextNInputs:
        """Prepare inputs for next token generation.

        Args:
            next_tokens: Next tokens to process
            prev_model_inputs: Previous model inputs
            hidden_states: Hidden states from the base model (optional, will use
                          hidden_states from prev_model_inputs if not provided)

        Returns:
            NextN model inputs for next token
        """
        assert isinstance(prev_model_inputs, DeepseekV3NextNInputs)

        # Use provided hidden_states, or fall back to previous hidden_states
        # This allows EAGLE pipeline to set hidden_states after calling this method
        if hidden_states is None:
            hidden_states = prev_model_inputs.hidden_states

        # If still None after fallback, that's a real error
        if hidden_states is None:
            raise ValueError(
                "hidden_states must be provided for DeepSeekV3 NextN model"
            )
        row_offsets_size = prev_model_inputs.input_row_offsets.shape[0]
        next_device_row_offsets = self._device_input_row_offsets_prealloc[
            :row_offsets_size
        ]
        next_host_row_offsets = self._host_input_row_offsets_prealloc[
            :row_offsets_size
        ]
        return DeepseekV3NextNInputs(
            tokens=next_tokens,
            hidden_states=hidden_states,
            input_row_offsets=next_device_row_offsets,
            host_input_row_offsets=next_host_row_offsets,
            signal_buffers=self.signal_buffers,
            batch_context_lengths=self._batch_context_lengths_prealloc_cpu,
            kv_cache_inputs=prev_model_inputs.kv_cache_inputs,
            return_n_logits=prev_model_inputs.return_n_logits,
            data_parallel_splits=prev_model_inputs.data_parallel_splits,
        )
