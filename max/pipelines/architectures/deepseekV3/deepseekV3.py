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
"""Implements the DeepseekV3 model."""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import Any

from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorType,
    TensorValue,
    Type,
    ops,
)
from max.graph.ops.allgather import allgather
from max.graph.ops.allreduce import matmul_allreduce
from max.nn import (
    MLP,
    ColumnParallelLinear,
    DistributedGemmConfig,
    LayerList,
    Module,
    RMSNorm,
    Signals,
    VocabParallelEmbedding,
)
from max.nn.attention.multi_latent_attention import (
    DataParallelLatentAttentionWithRope,
)
from max.nn.comm.allreduce import Allreduce
from max.nn.kv_cache import (
    PagedCacheValues,
    PagedKVCacheManager,
)
from max.nn.moe import MoE
from max.nn.rotary_embedding import (
    DeepseekYarnRopeScalingParams,
    DeepseekYarnRotaryEmbedding,
)
from max.nn.transformer import ReturnLogits
from max.nn.transformer.distributed_transformer import (
    distribute_value,
    forward_sharded_layers,
)

from .layers.moe_gate import DeepseekV3TopKRouter
from .model_config import DeepseekV3Config


class DeepseekV3DecoderLayer(Module):
    def __init__(
        self,
        rope: DeepseekYarnRotaryEmbedding,
        config: DeepseekV3Config,
        layer_idx: int,
        distributed_gemm_config: DistributedGemmConfig | None = None,
    ) -> None:
        super().__init__()
        num_devices = len(config.devices)

        # Create self-attention layer
        self.self_attn = DataParallelLatentAttentionWithRope(
            rope=rope,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            hidden_size=config.hidden_size,
            kv_params=config.kv_params,
            dtype=config.dtype,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            devices=config.devices,
        )

        # Create MLP or MoE layer
        self.mlp = self._get_mlp(config, layer_idx)
        self.mlp.sharding_strategy = ShardingStrategy.tensor_parallel(
            num_devices
        )
        self.mlp_shards = self.mlp.shard(config.devices)

        # Create normalization layers
        create_norm = functools.partial(
            RMSNorm,
            dim=config.hidden_size,
            dtype=config.dtype,
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

        self.distributed_gemm_config = distributed_gemm_config
        self.allreduce = Allreduce(num_accelerators=num_devices)

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
            moe = MoE(
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
                ),
                has_shared_experts=True,
                shared_experts_dim=config.n_shared_experts
                * config.moe_intermediate_size,
                dtype=config.dtype,
                apply_router_weight_first=False,
            )
            moe.sharding_strategy = ShardingStrategy.tensor_parallel(
                len(config.devices)
            )
            return moe
        else:
            mlp = MLP(
                dtype=config.dtype,
                quantization_encoding=None,
                hidden_dim=config.hidden_size,
                feed_forward_length=config.intermediate_size,
                devices=config.devices,
            )
            mlp.sharding_strategy = ShardingStrategy.tensor_parallel(
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
        input_row_offsets: list[TensorValue],
    ) -> list[TensorValue]:
        # Apply input layer norm to each shard
        norm_xs = forward_sharded_layers(self.input_layernorm_shards, xs)

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

        # Data-parallel attention (per-device)
        attn_outs = self.self_attn(
            layer_idx,
            norm_xs,
            signal_buffers,
            kv_collections,
            freqs_cis=freqs_cis,
            input_row_offsets=input_row_offsets,
        )

        # Residual add (still per-device)
        hs = [x + attn_out for x, attn_out in zip(xs, attn_outs, strict=True)]

        # Post-attention norm (per-device)
        norm_outs = forward_sharded_layers(
            self.post_attention_layernorm_shards, hs
        )

        # Allgather across data-parallel shards (batch axis)
        devices = [t.device for t in xs]
        gathered_list = allgather(norm_outs, signal_buffers, axis=0)

        # Tensor-parallel MLP runs on the replicated full batch
        mlp_full = forward_sharded_layers(self.mlp_shards, gathered_list)

        if (
            self.distributed_gemm_config is None
            or not self.distributed_gemm_config.enable_matmul_allreduce
        ):
            mlp_full = self.allreduce(mlp_full, signal_buffers)
        else:
            # Special matmul + allreduce split version
            weights = [layer.down_proj.weight for layer in self.mlp_shards]  # type: ignore[union-attr]
            mlp_full = matmul_allreduce(
                mlp_full,
                weights,
                signal_buffers,
            )

        # Re-split the full-batch MLP output back to per-device shards
        lengths_cpu: list[TensorValue] = []
        for offs in input_row_offsets:
            last_len = offs[-1]
            lengths_cpu.append(last_len.to(DeviceRef.CPU()))

        starts_cpu: list[TensorValue] = []
        running = ops.constant(0, DType.uint32, device=DeviceRef.CPU())
        for ln in lengths_cpu:
            starts_cpu.append(running)
            running = running + ln

        mlp_local: list[TensorValue] = []
        for i, (start_i, len_i) in enumerate(
            zip(starts_cpu, lengths_cpu, strict=True)
        ):
            end_i = start_i + len_i
            local_slice = ops.slice_tensor(
                mlp_full[i],
                [(slice(start_i, end_i), f"mlp_local_{i}")],
            ).to(devices[i])
            mlp_local.append(local_slice)

        # Final residual add stays in data-parallel layout
        hs = [h + m for h, m in zip(hs, mlp_local, strict=True)]

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

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.dtype,
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
            device=config.devices[0],
        )

        self.layers = LayerList(
            [
                DeepseekV3DecoderLayer(self.rope, config, i)
                for i in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(
            config.hidden_size,
            config.dtype,
            config.rms_norm_eps,
        )
        self.norm.sharding_strategy = ShardingStrategy.replicate(num_devices)
        self.norm_shards = self.norm.shard(devices)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            config.dtype,
            devices=config.devices,
            quantization_encoding=None,
        )

        self.subgraph_layer_groups = [
            [
                i
                for i in range(
                    config.first_k_dense_replace, config.num_hidden_layers
                )
            ]
        ]
        self.return_logits = ReturnLogits.LAST_TOKEN
        self.logits_scaling = 1.0

    def __call__(
        self,
        tokens: TensorValue,
        signal_buffers: list[BufferValue],
        kv_collections: list[PagedCacheValues],
        return_n_logits: TensorValue,
        input_row_offsets: TensorValue,
    ) -> tuple[TensorValue, ...]:
        devices = self.config.devices
        h = self.embed_tokens(tokens, signal_buffers)

        freqs_cis = distribute_value(self.rope.freqs_cis, devices)
        input_row_offsets_ = distribute_value(input_row_offsets, devices)

        subgraph_input_types: Sequence[Type[Any] | list[Type[Any]]] = [
            TensorType(DType.uint32, shape=(), device=DeviceRef.CPU()),
            [hidden.type for hidden in h],
            [signal_buffer.type for signal_buffer in signal_buffers],
            [kv_collection[0].type for kv_collection in kv_collections],
            [kv_collection[1].type for kv_collection in kv_collections],
            [kv_collection[2].type for kv_collection in kv_collections],
            [kv_collection[3].type for kv_collection in kv_collections],
            [freq.type for freq in freqs_cis],
            [offset.type for offset in input_row_offsets_],
        ]

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
                            *[
                                kv_collection[0]
                                for kv_collection in kv_collections
                            ],
                            *[
                                kv_collection[1]
                                for kv_collection in kv_collections
                            ],
                            *[
                                kv_collection[2]
                                for kv_collection in kv_collections
                            ],
                            *[
                                kv_collection[3]
                                for kv_collection in kv_collections
                            ],
                            *freqs_cis,
                            *input_row_offsets_,
                            prefix=f"layers.{idx}.",
                        )
                    ]
                    break
            if not has_subgraph:
                h = layer(
                    ops.constant(idx, DType.uint32, device=DeviceRef.CPU()),
                    h,
                    signal_buffers,
                    [kv_collection[0] for kv_collection in kv_collections],
                    [kv_collection[1] for kv_collection in kv_collections],
                    [kv_collection[2] for kv_collection in kv_collections],
                    [kv_collection[3] for kv_collection in kv_collections],
                    freqs_cis=freqs_cis,
                    input_row_offsets=input_row_offsets_,
                )
                assert isinstance(h, list)

        h0 = h[0]
        last_token_indices = input_row_offsets[1:] - 1
        last_token_h = ops.gather(h0, last_token_indices, axis=0)
        last_token_distributed = distribute_value(last_token_h, devices)

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
            return_n_logits_range = ops.range(
                start=return_n_logits[0],
                stop=0,
                step=-1,
                out_dim="return_n_logits_range",
                dtype=DType.int64,
                device=devices[0],
            )
            offsets = (
                ops.unsqueeze(input_row_offsets[1:], -1) - return_n_logits_range
            )
            last_indices = ops.reshape(offsets, shape=(-1,))
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
            offsets = input_row_offsets

        if self.logits_scaling != 1.0:
            last_logits = last_logits / self.logits_scaling
            if logits is not None:
                logits = logits / self.logits_scaling

        if logits is not None and offsets is not None:
            return (last_logits, logits, offsets)
        else:
            return (last_logits,)

    def input_types(
        self, kv_manager: PagedKVCacheManager
    ) -> tuple[TensorType | BufferType, ...]:
        # TODO: Move input symbol computation from the manager classes.
        # It should be possible to compute the input symbols from the model
        # config.
        device_ref = self.config.devices[0]

        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=device_ref
        )

        kv_inputs = kv_manager.input_symbols()

        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        )
        flattened_kv_types: list[TensorType] = [
            kv_type for sublist in kv_inputs for kv_type in sublist
        ]

        signals = Signals(devices=self.config.devices)
        signal_buffer_types: list[BufferType] = signals.input_types()

        all_input_types: list[TensorType | BufferType] = [
            tokens_type,
            input_row_offsets_type,
            return_n_logits_type,
        ]
        all_input_types.extend(signal_buffer_types)
        all_input_types.extend(flattened_kv_types)
        return tuple(all_input_types)
