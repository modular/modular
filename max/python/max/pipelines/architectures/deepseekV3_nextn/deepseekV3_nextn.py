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
"""Implements the DeepseekV3 NextN (Next-N token prediction) model."""

from __future__ import annotations

from typing import Any

from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorType,
    TensorValue,
    Value,
    ops,
)
from max.nn.legacy.attention.multi_latent_attention import MLAPrefillMetadata
from max.nn.legacy.comm import Signals
from max.nn.legacy.comm.ep import EPBatchManager
from max.nn.legacy.data_parallelism import split_batch_replicated
from max.nn.legacy.embedding import VocabParallelEmbedding
from max.nn.legacy.kv_cache import KVCacheParams, PagedCacheValues
from max.nn.legacy.layer import Module
from max.nn.legacy.linear import ColumnParallelLinear, Linear
from max.nn.legacy.norm import RMSNorm
from max.nn.legacy.rotary_embedding import (
    DeepseekYarnRopeScalingParams,
    DeepseekYarnRotaryEmbedding,
)
from max.nn.legacy.transformer import ReturnHiddenStates
from max.nn.legacy.transformer.distributed_transformer import (
    distribute_value,
    forward_sharded_layers,
)

from ..deepseekV3.deepseekV3 import DeepseekV3DecoderLayer
from .model_config import DeepseekV3NextNConfig


class DeepseekV3NextN(Module):
    def __init__(self, config: DeepseekV3NextNConfig) -> None:
        super().__init__()
        self.config = config
        num_devices = len(config.devices)
        devices = config.devices

        self.use_data_parallel_attention = (
            num_devices > 1 and config.data_parallel_degree == num_devices
        )

        embedding_output_dtype = config.dtype
        if config.float8_config and config.float8_config.embedding_output_dtype:
            embedding_output_dtype = config.float8_config.embedding_output_dtype
        self.embedding_output_dtype = embedding_output_dtype
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            dtype=embedding_output_dtype,
            devices=config.devices,
            quantization_encoding=None,
        )

        self.enorm = RMSNorm(
            config.hidden_size,
            config.norm_dtype,
            config.rms_norm_eps,
            multiply_before_cast=False,
        )
        self.enorm.sharding_strategy = ShardingStrategy.replicate(num_devices)
        self.enorm_shards = self.enorm.shard(devices)

        self.hnorm = RMSNorm(
            config.hidden_size,
            config.norm_dtype,
            config.rms_norm_eps,
            multiply_before_cast=False,
        )
        self.hnorm.sharding_strategy = ShardingStrategy.replicate(num_devices)
        self.hnorm_shards = self.hnorm.shard(devices)

        self.eh_proj = Linear(
            config.hidden_size * 2,
            config.hidden_size,
            embedding_output_dtype,
            devices[0],
            quantization_encoding=None,
            has_bias=False,
        )
        self.eh_proj.sharding_strategy = ShardingStrategy.replicate(num_devices)
        self.eh_proj_shards = self.eh_proj.shard(devices)

        assert config.rope_scaling is not None
        scaling_params = DeepseekYarnRopeScalingParams(
            scaling_factor=config.rope_scaling["factor"],
            original_max_position_embeddings=config.rope_scaling[
                "original_max_position_embeddings"
            ],
            beta_fast=config.rope_scaling["beta_fast"],
            beta_slow=config.rope_scaling["beta_slow"],
            mscale=config.rope_scaling["mscale"],
            mscale_all_dim=config.rope_scaling["mscale_all_dim"],
        )
        self.rope = DeepseekYarnRotaryEmbedding(
            config.qk_rope_head_dim,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_position_embeddings,
            scaling_params=scaling_params,
        )

        self.ep_manager: EPBatchManager | None = None
        if config.ep_config is not None:
            self.ep_manager = EPBatchManager(config.ep_config)

        # Ensure MoE layer creation by using layer_idx >= first_k_dense_replace
        nextn_layer_idx = max(
            config.num_hidden_layers, config.first_k_dense_replace
        )
        self.decoder_layer = DeepseekV3DecoderLayer(
            self.rope,
            config,
            layer_idx=nextn_layer_idx,
            ep_manager=self.ep_manager,
            is_nextn=True,
        )

        self.shared_head_norm = RMSNorm(
            config.hidden_size,
            config.norm_dtype,
            config.rms_norm_eps,
            multiply_before_cast=False,
        )
        self.shared_head_norm.sharding_strategy = ShardingStrategy.replicate(
            num_devices
        )
        self.shared_head_norm_shards = self.shared_head_norm.shard(devices)

        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            embedding_output_dtype,
            devices=config.devices,
            quantization_encoding=None,
        )

        self.return_logits = config.return_logits
        self.return_hidden_states = config.return_hidden_states
        self.logits_scaling = 1.0

    def __call__(
        self,
        tokens: TensorValue,
        hidden_states: list[TensorValue],
        signal_buffers: list[BufferValue],
        kv_collections: list[PagedCacheValues],
        return_n_logits: TensorValue,
        input_row_offsets: TensorValue,
        host_input_row_offsets: TensorValue,
        data_parallel_splits: TensorValue,
        batch_context_lengths: list[TensorValue],
        ep_inputs: list[Value[Any]] | None = None,
    ) -> tuple[TensorValue, ...]:
        if not host_input_row_offsets.device == DeviceRef.CPU():
            raise ValueError("host_input_row_offsets must be located on CPU")
        if not data_parallel_splits.device == DeviceRef.CPU():
            raise ValueError("data_parallel_splits must be located on CPU")

        devices = self.config.devices

        if len(hidden_states) != len(devices):
            raise ValueError(
                f"hidden_states list length ({len(hidden_states)}) must match "
                f"number of devices ({len(devices)})"
            )

        h_embed = self.embed_tokens(tokens, signal_buffers)
        norm_embed = forward_sharded_layers(self.enorm_shards, h_embed)
        norm_hidden = forward_sharded_layers(self.hnorm_shards, hidden_states)
        freqs_cis = distribute_value(self.rope.freqs_cis, devices)
        input_row_offsets_ = distribute_value(input_row_offsets, devices)
        # Split embeddings FIRST to match already-split hidden_states from target model
        # hidden_states are already data-parallel split (different sizes per device),
        # while norm_embed is replicated (same size on all devices).
        # We must split norm_embed before concatenating with norm_hidden.
        if self.use_data_parallel_attention:
            norm_embed, input_row_offsets_ = split_batch_replicated(
                devices,
                norm_embed,
                input_row_offsets_,
                host_input_row_offsets.cast(DType.int64),
                data_parallel_splits,
            )
            # Rebind split embeddings to match hidden_states dimension names
            # split_batch_replicated uses 'input_split_{i}' but hidden_states use
            # 'seq_len_device_{i}' - they're logically the same size at runtime
            norm_embed = [
                ops.rebind(
                    norm_embed[i],
                    [f"seq_len_device_{i}", self.config.hidden_size],
                )
                for i in range(len(devices))
            ]
        else:
            # Single-device case: rebind norm_embed to match hidden_states dimension names
            # hidden_states uses 'seq_len_device_0' but norm_embed uses 'total_seq_len'
            norm_embed = [
                ops.rebind(
                    norm_embed[i],
                    [f"seq_len_device_{i}", self.config.hidden_size],
                )
                for i in range(len(devices))
            ]

        concat_inputs = [
            ops.concat([norm_embed[i], norm_hidden[i]], axis=-1)
            for i in range(len(devices))
        ]
        h = forward_sharded_layers(self.eh_proj_shards, concat_inputs)

        # Create MLA prefill metadata if not in decode mode (similar to base DeepSeek V3)
        mla_prefill_metadata: list[MLAPrefillMetadata] = []
        if self.config.graph_mode != "decode":
            mla_prefill_metadata = (
                self.decoder_layer.self_attn.create_mla_prefill_metadata(
                    input_row_offsets_, kv_collections
                )
            )

            assert len(mla_prefill_metadata) == len(batch_context_lengths)
            for i in range(len(batch_context_lengths)):
                mla_prefill_metadata[i].buffer_lengths = batch_context_lengths[
                    i
                ]

        mla_inputs: list[TensorValue] = []
        for metadata in mla_prefill_metadata:
            mla_inputs.extend(
                [
                    metadata.buffer_row_offsets,
                    metadata.cache_offsets,
                    metadata.buffer_lengths,
                ]
            )

        h = self.decoder_layer(
            ops.constant(0, DType.uint32, device=DeviceRef.CPU()),
            h,
            signal_buffers,
            [kv_collection.kv_blocks for kv_collection in kv_collections],
            [kv_collection.cache_lengths for kv_collection in kv_collections],
            [kv_collection.lookup_table for kv_collection in kv_collections],
            [kv_collection.max_lengths for kv_collection in kv_collections],
            freqs_cis=freqs_cis,
            mla_prefill_metadata_flat=mla_inputs,
            input_row_offsets=input_row_offsets_,
            ep_inputs=ep_inputs,
        )

        if self.config.data_parallel_degree > 1:
            last_token_per_dev: list[TensorValue] = []
            for dev_idx in range(len(devices)):
                h0 = h[dev_idx]
                last_token_indices = input_row_offsets_[dev_idx][1:] - 1
                last_token_h = ops.gather(h0, last_token_indices, axis=0)
                last_token_per_dev.append(last_token_h)
            last_token_distributed = ops.allgather(
                last_token_per_dev, signal_buffers
            )
        else:
            h0 = h[0]
            last_token_indices = input_row_offsets_[0][1:] - 1
            last_token_h = ops.gather(h0, last_token_indices, axis=0)
            last_token_distributed = distribute_value(last_token_h, devices)

        norm_last_token = forward_sharded_layers(
            self.shared_head_norm_shards, last_token_distributed
        )
        last_logits = ops.cast(
            self.lm_head(norm_last_token, signal_buffers)[0],
            DType.float32,
        )
        if self.logits_scaling != 1.0:
            last_logits = last_logits / self.logits_scaling

        ret_val: tuple[TensorValue, ...] = (last_logits,)

        if self.return_hidden_states == ReturnHiddenStates.ALL:
            ret_val += tuple(h)
        elif self.return_hidden_states == ReturnHiddenStates.LAST:
            if self.config.data_parallel_degree > 1:
                ret_val += tuple(last_token_per_dev)
            else:
                # For non-data-parallel case, distribute the single tensor to match interface
                ret_val += tuple(distribute_value(last_token_h, devices))
        elif self.return_hidden_states == ReturnHiddenStates.ALL_NORMALIZED:
            norm_h = forward_sharded_layers(self.shared_head_norm_shards, h)
            ret_val += tuple(norm_h)
        elif self.return_hidden_states == ReturnHiddenStates.LAST_NORMALIZED:
            ret_val += tuple(norm_last_token)

        return ret_val

    def input_types(
        self, kv_params: KVCacheParams
    ) -> tuple[TensorType | BufferType, ...]:
        devices = self.config.devices
        device_ref = devices[0]

        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )

        # Hidden states input types - one per device for data parallelism
        # Each device receives a different slice of the batch (data parallel split)
        # Using device-specific dimension names allows different sizes per device
        hidden_states_types = [
            TensorType(
                self.embedding_output_dtype,
                shape=[f"seq_len_device_{i}", self.config.hidden_size],
                device=dev,
            )
            for i, dev in enumerate(devices)
        ]

        device_input_row_offsets_type = TensorType(
            DType.uint32,
            shape=["input_row_offsets_len"],
            device=device_ref,
        )
        host_input_row_offsets_type = TensorType(
            DType.uint32,
            shape=["input_row_offsets_len"],
            device=DeviceRef.CPU(),
        )
        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )
        data_parallel_splits_type = TensorType(
            DType.int64,
            shape=[self.config.data_parallel_degree + 1],
            device=DeviceRef.CPU(),
        )

        signals = Signals(devices=devices)
        signal_buffer_types: list[BufferType] = signals.input_types()

        all_input_types: list[TensorType | BufferType] = [tokens_type]
        all_input_types.extend(hidden_states_types)
        all_input_types.extend(
            [
                device_input_row_offsets_type,
                host_input_row_offsets_type,
                return_n_logits_type,
                data_parallel_splits_type,
            ]
        )
        all_input_types.extend(signal_buffer_types)
        all_input_types.extend(kv_params.get_symbolic_inputs().flatten())

        # Add batch context lengths (one per device)
        batch_context_length_type = TensorType(
            DType.int32, shape=[1], device=DeviceRef.CPU()
        )
        all_input_types.extend(
            [batch_context_length_type for _ in range(len(devices))]
        )

        if self.ep_manager is not None:
            all_input_types.extend(self.ep_manager.input_types())

        return tuple(all_input_types)
