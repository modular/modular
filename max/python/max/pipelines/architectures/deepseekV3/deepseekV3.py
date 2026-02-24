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
"""Implements the DeepseekV3 model."""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import Any

from max._core.driver import is_virtual_device_mode
from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorType,
    TensorValue,
    Type,
    Value,
    ops,
)
from max.nn.legacy.attention.multi_latent_attention import (
    DataParallelLatentAttentionWithRope,
    MLAPrefillMetadata,
)
from max.nn.legacy.attention.multi_latent_attention_fp8 import (
    DataParallelLatentAttentionWithRopeFp8,
)
from max.nn.legacy.comm import Signals
from max.nn.legacy.comm.ep import EPBatchManager
from max.nn.legacy.data_parallelism import split_batch_replicated
from max.nn.legacy.embedding import VocabParallelEmbedding
from max.nn.legacy.kv_cache import KVCacheParams, PagedCacheValues
from max.nn.legacy.layer import LayerList, Module
from max.nn.legacy.linear import (
    MLP,
    ColumnParallelLinear,
)
from max.nn.legacy.moe import MoE, MoEQuantized
from max.nn.legacy.norm import RMSNorm
from max.nn.legacy.rotary_embedding import (
    DeepseekYarnRopeScalingParams,
    DeepseekYarnRotaryEmbedding,
)
from max.nn.legacy.transformer import ReturnHiddenStates, ReturnLogits
from max.nn.legacy.transformer.distributed_transformer import (
    forward_sharded_layers,
)

from .layers.moe_gate import DeepseekV3TopKRouter
from .model_config import DeepseekV3Config


def _unpack_kv_collections(
    kv_collections: Sequence[PagedCacheValues],
) -> tuple[
    list[BufferValue], list[TensorValue], list[TensorValue], list[TensorValue]
]:
    """Unpack KV collections into component lists.

    Returns:
        Tuple of (kv_blocks, cache_lengths, lookup_tables, max_lengths).
    """
    return (
        [kv.kv_blocks for kv in kv_collections],
        [kv.cache_lengths for kv in kv_collections],
        [kv.lookup_table for kv in kv_collections],
        [kv.max_lengths for kv in kv_collections],
    )


def _validate_parallelism_config(config: DeepseekV3Config) -> None:
    """Validate parallelism configuration for DeepseekV3."""
    num_devices = len(config.devices)
    if config.data_parallel_degree != num_devices:
        raise ValueError(
            f"data_parallel_degree must match the number of devices ({num_devices}). "
            "Tensor-parallel attention is not supported for DeepseekV3."
        )
    # Skip EP validation in virtual device mode (compilation-only) since EP
    # will be disabled later due to NVSHMEM linking requirements
    if (
        num_devices > 1
        and config.ep_config is None
        and not is_virtual_device_mode()
    ):
        raise ValueError(
            "Expert-parallel (ep_config) must be enabled for multi-GPU DeepseekV3."
        )


class DeepseekV3DecoderLayer(Module):
    def __init__(
        self,
        rope: DeepseekYarnRotaryEmbedding,
        config: DeepseekV3Config,
        layer_idx: int,
        ep_manager: EPBatchManager | None = None,
        is_nextn: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.ep_manager = ep_manager
        self.is_nextn = is_nextn
        num_devices = len(config.devices)

        # Create Multi-head Latent Attention layer.
        mla_kwargs: dict[str, Any] = dict(
            rope=rope,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            hidden_size=config.hidden_size,
            kv_params=config.kv_params,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            devices=config.devices,
            graph_mode=config.graph_mode,
            buffer_size=config.max_batch_context_length,
        )

        nvfp4_enabled = (
            config.float8_config is not None and config.float8_config.is_nvfp4
        )
        use_fp8_mla = config.float8_config is not None and not nvfp4_enabled

        if config.float8_config is not None and nvfp4_enabled:
            mla_kwargs["o_proj_float8_config"] = config.float8_config
            mla_kwargs["o_proj_dtype"] = config.dtype

        mla_cls: (
            type[DataParallelLatentAttentionWithRope]
            | type[DataParallelLatentAttentionWithRopeFp8]
        )
        if use_fp8_mla:
            mla_kwargs["float8_config"] = config.float8_config
            mla_cls = DataParallelLatentAttentionWithRopeFp8
        else:
            mla_kwargs["dtype"] = DType.bfloat16
            mla_cls = DataParallelLatentAttentionWithRope

        self.self_attn = mla_cls(**mla_kwargs)

        # Create MLP or MoE layer
        self.mlp = self._get_mlp(config, layer_idx)

        self.mlp_shards: list[MLP | MoE]
        if self.mlp.sharding_strategy is not None:
            self.mlp_shards = list(self.mlp.shard(config.devices))
        else:
            self.mlp_shards = [self.mlp]

        # Create normalization layers
        create_norm = functools.partial(
            RMSNorm,
            dim=config.hidden_size,
            dtype=config.norm_dtype,
            eps=config.rms_norm_eps,
            multiply_before_cast=False,
        )
        self.input_layernorm = create_norm()
        self.input_layernorm.sharding_strategy = ShardingStrategy.replicate(
            num_devices
        )
        self.input_layernorm_shards = self.input_layernorm.shard(config.devices)

        self.post_attention_layernorm = create_norm()
        self.post_attention_layernorm.sharding_strategy = (
            ShardingStrategy.replicate(num_devices)
        )
        self.post_attention_layernorm_shards = (
            self.post_attention_layernorm.shard(config.devices)
        )

    def _get_mlp(self, config: DeepseekV3Config, layer_idx: int) -> MLP | MoE:
        """Helper function to return a mixture of experts layer or traditional multi-layer perceptron layer
        for the TransformerBlock's mlp depending on the layer idx.

        Args:
            config: Configuration object containing model parameters
            layer_idx: Layer index

        Returns:
            List of MLP shards or MoE modules depending on the layer index and config
        """
        if (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        ):
            if config.ep_config is not None:
                ep_size = (
                    config.ep_config.n_gpus_per_node * config.ep_config.n_nodes
                )
            else:
                ep_size = 1

            moe_kwargs: dict[str, Any] = dict(
                devices=config.devices,
                hidden_dim=config.hidden_size,
                num_experts=config.n_routed_experts,
                num_experts_per_token=config.num_experts_per_tok,
                moe_dim=config.moe_intermediate_size,
                gate_cls=functools.partial(
                    DeepseekV3TopKRouter,
                    routed_scaling_factor=config.routed_scaling_factor,
                    scoring_func=config.scoring_func,
                    topk_method=config.topk_method,
                    n_group=config.n_group,
                    topk_group=config.topk_group,
                    norm_topk_prob=config.norm_topk_prob,
                    # Use the same dtype for the gate as the norm
                    gate_dtype=config.norm_dtype,
                    correction_bias_dtype=config.correction_bias_dtype,
                ),
                has_shared_experts=True,
                shared_experts_dim=config.n_shared_experts
                * config.moe_intermediate_size,
                dtype=config.dtype,
                ep_size=ep_size,
                apply_router_weight_first=False,
                ep_batch_manager=self.ep_manager,
                float8_config=config.float8_config,
            )

            moe: MoE
            if config.float8_config is not None:
                moe = MoEQuantized(**moe_kwargs)
            else:
                moe = MoE(**moe_kwargs)

            num_devices = len(config.devices)
            if num_devices > 1:
                moe.sharding_strategy = ShardingStrategy.expert_parallel(
                    num_devices
                )
            return moe
        else:
            mlp = MLP(
                dtype=config.dtype,
                quantization_encoding=None,
                hidden_dim=config.hidden_size,
                feed_forward_length=config.intermediate_size,
                devices=config.devices,
                float8_config=config.float8_config,
            )
            mlp.sharding_strategy = ShardingStrategy.replicate(
                len(config.devices)
            )
            return mlp

    def __call__(
        self,
        layer_idx: TensorValue,
        xs: list[TensorValue],
        signal_buffers: list[BufferValue],
        kv_blocks: list[BufferValue],
        kv_cache_lengths: list[TensorValue],
        kv_lookup_table: list[TensorValue],
        kv_max_lengths: list[TensorValue],
        freqs_cis: list[TensorValue],
        mla_prefill_metadata_flat: list[TensorValue],
        input_row_offsets: list[TensorValue],
        ep_inputs: list[Value[Any]] | None = None,
    ) -> list[TensorValue]:
        # We have to unpack our PagedCacheValues into constituent parts so
        # subgraphs have only max.graph.Values as arguments.
        # Re-pack those arguments into a nice structured type.
        kv_collections = [
            PagedCacheValues(
                kv_blocks[i],
                kv_cache_lengths[i],
                kv_lookup_table[i],
                kv_max_lengths[i],
            )
            for i in range(len(kv_blocks))
        ]

        # Re-pack flat MLA inputs into MLAPrefillMetadata dataclasses
        num_devices = len(kv_blocks)
        mla_prefill_metadata: list[MLAPrefillMetadata] = []
        if self.config.graph_mode != "decode":
            assert len(mla_prefill_metadata_flat) == 3 * num_devices
            for i in range(num_devices):
                mla_prefill_metadata.append(
                    MLAPrefillMetadata(
                        buffer_row_offsets=mla_prefill_metadata_flat[3 * i],
                        cache_offsets=mla_prefill_metadata_flat[3 * i + 1],
                        buffer_lengths=mla_prefill_metadata_flat[3 * i + 2],
                    )
                )

        # Apply input layer norm to each shard
        norm_xs = forward_sharded_layers(self.input_layernorm_shards, xs)

        attn_outs = self.self_attn(
            layer_idx,
            norm_xs,
            signal_buffers,
            kv_collections,
            freqs_cis=freqs_cis,
            input_row_offsets=input_row_offsets,
            mla_prefill_metadata=mla_prefill_metadata,
        )

        hs = [x + attn_out for x, attn_out in zip(xs, attn_outs, strict=True)]

        # Post-attention norm (per-device)
        norm_outs = forward_sharded_layers(
            self.post_attention_layernorm_shards, hs
        )

        if self.config.ep_config is not None:
            assert ep_inputs is not None
            if self.ep_manager is not None:
                self.ep_manager.fetch_buffers(ep_inputs)

            mlp_outs = forward_sharded_layers(self.mlp_shards, norm_outs)

        else:
            # Single-GPU non-EP path
            mlp_outs = forward_sharded_layers(self.mlp_shards, norm_outs)

        if self.is_nextn:
            # NextN/MTP decoder: skip the second residual connection.
            # The MoE output is used directly as hidden_states.
            hs = mlp_outs
        else:
            hs = [h + mlp_out for h, mlp_out in zip(hs, mlp_outs, strict=True)]

        return hs


class DeepseekV3(Module):
    """Defines the DeepseekV3 transformer model.

    This is a combination of the DeepseekV3Model and the DeepseekV3ForCausalLM
    classes from the HuggingFace Transformers implementation.
    """

    def __init__(self, config: DeepseekV3Config) -> None:
        super().__init__()
        self.config = config
        num_devices = len(config.devices)
        devices = config.devices

        _validate_parallelism_config(config)

        embedding_output_dtype = config.dtype
        if embedding_output_dtype == DType.uint8:
            embedding_output_dtype = DType.bfloat16
        if config.float8_config and config.float8_config.embedding_output_dtype:
            embedding_output_dtype = config.float8_config.embedding_output_dtype
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            dtype=embedding_output_dtype,
            devices=config.devices,
            quantization_encoding=None,
        )

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

        self.layers = LayerList(
            [
                DeepseekV3DecoderLayer(
                    self.rope,
                    config,
                    i,
                    None
                    if i < config.first_k_dense_replace
                    else self.ep_manager,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(
            config.hidden_size,
            config.norm_dtype,
            config.rms_norm_eps,
        )
        self.norm.sharding_strategy = ShardingStrategy.replicate(num_devices)
        self.norm_shards = self.norm.shard(devices)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            embedding_output_dtype,
            devices=config.devices,
            quantization_encoding=None,
        )

        if config.use_subgraphs:
            self.subgraph_layer_groups = [
                [
                    i
                    for i in range(
                        config.first_k_dense_replace, config.num_hidden_layers
                    )
                ]
            ]
        else:
            self.subgraph_layer_groups = []
        self.return_logits = config.return_logits
        self.return_hidden_states = config.return_hidden_states
        self.logits_scaling = 1.0

    def __call__(
        self,
        tokens: TensorValue,
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
            raise ValueError("input_row_offsets must be located on CPU")
        if not data_parallel_splits.device == DeviceRef.CPU():
            raise ValueError("data_parallel_splits must be located on CPU")

        devices = self.config.devices
        h = self.embed_tokens(tokens, signal_buffers)

        mla_prefill_metadata: list[MLAPrefillMetadata] = []
        # Keep this as explicit per-device `.to()` copies.
        # Broadcasting graph-time constants can hang when chained after
        # runtime-dependent collectives (GEX-3200).
        freqs_cis = [self.rope.freqs_cis.to(device) for device in devices]
        if not input_row_offsets.device == devices[0]:
            raise ValueError(
                f"input_row_offsets must be located on {devices[0]}"
            )
        input_row_offsets_ = ops.distributed_broadcast(
            input_row_offsets, signal_buffers
        )

        if len(devices) > 1:
            # Split batch across devices for data-parallel attention.
            h, input_row_offsets_ = split_batch_replicated(
                devices,
                h,
                input_row_offsets_,
                host_input_row_offsets.cast(DType.int64),
                data_parallel_splits,
            )

        # Create MLA prefill metadata if not in decode mode
        if self.config.graph_mode != "decode":
            mla_prefill_metadata = self.layers[
                0
            ].self_attn.create_mla_prefill_metadata(  # type: ignore
                input_row_offsets_, kv_collections
            )

            # replace each device's buffer_lengths with the batch context length
            assert len(mla_prefill_metadata) == len(batch_context_lengths)
            for i in range(len(batch_context_lengths)):
                mla_prefill_metadata[i].buffer_lengths = batch_context_lengths[
                    i
                ]

        # Flatten MLAPrefillMetadata to list of TensorValues for subgraph calls
        mla_prefill_metadata_flat: list[TensorValue] = []
        for metadata in mla_prefill_metadata:
            mla_prefill_metadata_flat.extend(
                [
                    metadata.buffer_row_offsets,
                    metadata.cache_offsets,
                    metadata.buffer_lengths,
                ]
            )

        # Unpack KV collections once for use throughout the method
        kv_blocks, cache_lengths, lookup_tables, max_lengths = (
            _unpack_kv_collections(kv_collections)
        )

        subgraph_input_types: list[Type[Any] | list[Type[Any]]] = [
            TensorType(DType.uint32, shape=(), device=DeviceRef.CPU()),
            [hidden.type for hidden in h],
            [signal_buffer.type for signal_buffer in signal_buffers],
            [block.type for block in kv_blocks],
            [length.type for length in cache_lengths],
            [table.type for table in lookup_tables],
            [length.type for length in max_lengths],
            [freq.type for freq in freqs_cis],
            [val.type for val in mla_prefill_metadata_flat],
            [offset.type for offset in input_row_offsets_],
        ]

        if self.ep_manager is not None:
            subgraph_input_types.append(list(self.ep_manager.input_types()))

        subgraphs = []
        for group_idx, layer_group in enumerate(self.subgraph_layer_groups):
            assert len(layer_group) > 0, (
                "Subgraph layer groups must contain at least one layer"
            )
            subgraph_layer = self.layers[layer_group[0]]
            assert isinstance(subgraph_layer, DeepseekV3DecoderLayer), (
                "Subgraph layer must be a DeepseekV3DecoderLayer"
            )
            subgraphs.append(
                subgraph_layer.build_subgraph(
                    f"dist_transformer_block_{group_idx}",
                    subgraph_input_types,
                    f"layers.{layer_group[0]}.",
                )
            )

        for idx, layer in enumerate(self.layers):
            has_subgraph = False
            for group_idx, layer_group in enumerate(self.subgraph_layer_groups):
                if idx in layer_group:
                    has_subgraph = True
                    h = [
                        x.tensor
                        for x in ops.call(
                            subgraphs[group_idx],
                            ops.constant(
                                idx, DType.uint32, device=DeviceRef.CPU()
                            ),
                            *h,
                            *signal_buffers,
                            *kv_blocks,
                            *cache_lengths,
                            *lookup_tables,
                            *max_lengths,
                            *freqs_cis,
                            *mla_prefill_metadata_flat,
                            *input_row_offsets_,
                            *(ep_inputs if ep_inputs is not None else ()),
                            prefix=f"layers.{idx}.",
                        )
                    ]
                    break
            if not has_subgraph:
                h = layer(
                    ops.constant(idx, DType.uint32, device=DeviceRef.CPU()),
                    h,
                    signal_buffers,
                    kv_blocks,
                    cache_lengths,
                    lookup_tables,
                    max_lengths,
                    freqs_cis=freqs_cis,
                    mla_prefill_metadata_flat=mla_prefill_metadata_flat,
                    input_row_offsets=input_row_offsets_,
                    ep_inputs=ep_inputs,
                )
                assert isinstance(h, list)

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
            last_token_distributed = [
                ops.gather(h_i, offsets_i[1:] - 1, axis=0)
                for h_i, offsets_i in zip(h, input_row_offsets_, strict=True)
            ]

        # Apply norm to each shard
        norm_last_token = forward_sharded_layers(
            self.norm_shards, last_token_distributed
        )
        last_logits = ops.cast(
            self.lm_head(norm_last_token, signal_buffers)[0],
            DType.float32,
        )

        logits = None
        offsets = None

        if self.return_logits == ReturnLogits.VARIABLE:
            if self.config.data_parallel_degree > 1:
                # Data-parallel case: gather variable tokens per device, then allgather
                # Create the range once on device 0 (range inputs must be on CPU)
                return_n_logits_range = ops.range(
                    start=return_n_logits[0],
                    stop=0,
                    step=-1,
                    out_dim="return_n_logits_range",
                    dtype=DType.int64,
                    device=devices[0],
                )
                variable_tokens_per_dev: list[TensorValue] = []
                for dev_idx in range(len(devices)):
                    h0 = h[dev_idx]
                    dev_return_n_logits_range = return_n_logits_range.to(
                        devices[dev_idx]
                    )
                    # Compute indices for last return_n_logits tokens per
                    # sequence on this device
                    dev_offsets = (
                        ops.unsqueeze(input_row_offsets_[dev_idx][1:], -1)
                        - dev_return_n_logits_range
                    )
                    indices = ops.reshape(dev_offsets, shape=(-1,))
                    variable_h = ops.gather(h0, indices, axis=0)
                    variable_tokens_per_dev.append(variable_h)

                variable_tokens_distributed = ops.allgather(
                    variable_tokens_per_dev, signal_buffers
                )

                norm_variable_tokens = forward_sharded_layers(
                    self.norm_shards, variable_tokens_distributed
                )
                logits = ops.cast(
                    self.lm_head(norm_variable_tokens, signal_buffers)[0],
                    DType.float32,
                )

                offsets = ops.range(
                    0,
                    TensorValue(logits.shape[0]) + return_n_logits[0],
                    return_n_logits[0],
                    out_dim="logit_offsets",
                    dtype=DType.int64,
                    device=devices[0],
                )
            else:
                # Non-EP case: keep existing single-device implementation
                return_n_logits_range = ops.range(
                    start=return_n_logits[0],
                    stop=0,
                    step=-1,
                    out_dim="return_n_logits_range",
                    dtype=DType.int64,
                    device=devices[0],
                )
                last_offsets = (
                    ops.unsqueeze(input_row_offsets_[0][1:], -1)
                    - return_n_logits_range
                )
                last_indices = ops.reshape(last_offsets, shape=(-1,))
                logits = ops.gather(
                    ops.cast(
                        self.lm_head(
                            forward_sharded_layers(self.norm_shards, h),
                            signal_buffers,
                        )[0],
                        DType.float32,
                    ),
                    last_indices,
                    axis=0,
                )
                offsets = ops.range(
                    0,
                    TensorValue(last_indices.shape[0]) + return_n_logits[0],
                    return_n_logits[0],
                    out_dim="logit_offsets",
                    dtype=DType.int64,
                    device=devices[0],
                )
        elif self.return_logits == ReturnLogits.ALL:
            logits = ops.cast(
                self.lm_head(
                    forward_sharded_layers(self.norm_shards, h),
                    signal_buffers,
                )[0],
                DType.float32,
            )
            offsets = input_row_offsets_[0]

        if self.logits_scaling != 1.0:
            last_logits = last_logits / self.logits_scaling
            if logits is not None:
                logits = logits / self.logits_scaling

        ret_val: tuple[TensorValue, ...] = (last_logits,)
        if logits is not None and offsets is not None:
            ret_val += (logits, offsets)

        if self.return_hidden_states == ReturnHiddenStates.ALL:
            ret_val += tuple(h)
        elif self.return_hidden_states == ReturnHiddenStates.LAST:
            if self.config.data_parallel_degree > 1:
                ret_val += tuple(last_token_per_dev)
            else:
                ret_val += tuple(last_token_distributed)
        elif self.return_hidden_states == ReturnHiddenStates.ALL_NORMALIZED:
            norm_h = forward_sharded_layers(self.norm_shards, h)
            ret_val += tuple(norm_h)
        elif self.return_hidden_states == ReturnHiddenStates.LAST_NORMALIZED:
            ret_val += tuple(norm_last_token)

        return ret_val

    def input_types(
        self, kv_params: KVCacheParams
    ) -> tuple[TensorType | BufferType, ...]:
        # TODO: Move input symbol computation from the manager classes.
        # It should be possible to compute the input symbols from the model
        # config.
        device_ref = self.config.devices[0]

        # Construct Graph Inputs
        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        device_input_row_offsets_type = TensorType(
            DType.uint32,
            shape=["input_row_offsets_len"],
            device=device_ref,
        )

        # Add host input row offsets type, this is used to split the
        # concatenated DP inputs.
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

        signals = Signals(devices=self.config.devices)
        signal_buffer_types: list[BufferType] = signals.input_types()

        all_input_types: list[TensorType | BufferType] = [
            tokens_type,
            device_input_row_offsets_type,
            host_input_row_offsets_type,
            return_n_logits_type,
            data_parallel_splits_type,
        ]
        all_input_types.extend(signal_buffer_types)
        all_input_types.extend(kv_params.get_symbolic_inputs().flatten())

        # Add batch context lengths
        batch_context_length_type = TensorType(
            DType.int32, shape=[1], device=DeviceRef.CPU()
        )
        all_input_types.extend(
            [batch_context_length_type for _ in range(len(self.config.devices))]
        )

        if self.ep_manager is not None:
            all_input_types.extend(self.ep_manager.input_types())
        return tuple(all_input_types)
