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

from collections.abc import Callable, Iterable, Sequence
from typing import Any

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    ops,
)
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.attention.multi_latent_attention import MLAPrefillMetadata
from max.nn.attention.multi_latent_attention_fp8 import (
    LatentAttentionWithRopeFp8,
)
from max.nn.kernels import (
    mla_decode_graph,
    mla_prefill_decode_graph,
    mla_prefill_graph,
)
from max.nn.kv_cache import KVCacheParams, PagedCacheValues
from max.nn.linear import Linear
from max.nn.quant_config import QuantConfig
from max.nn.quant_ops import quantized_matmul
from max.nn.rotary_embedding import RotaryEmbedding

from .indexer import Indexer


class SparseLatentAttentionWithRopeFp8(LatentAttentionWithRopeFp8):
    """FP8 latent attention with optional sparse decode (physical KV row indices)."""

    def __init__(
        self,
        *,
        rope: RotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        quant_config: QuantConfig,
        devices: list[DeviceRef] | None = None,
        linear_cls: Callable[..., Linear] = Linear,
        scale: float | None = None,
        q_lora_rank: int = 1536,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        buffer_size: int = 16384,
        graph_mode: str | None = None,
        norm_dtype: DType = DType.bfloat16,
        index_n_heads: int = 64,
        index_head_dim: int = 128,
        index_topk: int = 2048,
    ):
        super().__init__(
            rope=rope,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_size=hidden_size,
            kv_params=kv_params,
            quant_config=quant_config,
            devices=devices,
            linear_cls=linear_cls,
            scale=scale,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            buffer_size=buffer_size,
            graph_mode=graph_mode,
            norm_dtype=norm_dtype,
        )
        self.indexer = Indexer(
            dim=hidden_size,
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            index_topk=index_topk,
            q_lora_rank=q_lora_rank,
            devices=self.devices,
            quant_config=self.quant_config,
        )

    @LatentAttentionWithRopeFp8.sharding_strategy.setter  # type: ignore[attr-defined]
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Extends the base setter so indexer weights participate in replicate DP."""
        LatentAttentionWithRopeFp8.sharding_strategy.fset(self, strategy)  # type: ignore[attr-defined]
        if strategy.is_replicate:
            rep = ShardingStrategy.replicate(strategy.num_devices)
            for linear in (
                self.indexer.wq_b,
                self.indexer.wk,
                self.indexer.weights_proj,
            ):
                linear.weight.sharding_strategy = rep
                if linear.weight_scale is not None:
                    linear.weight_scale.sharding_strategy = rep
                if linear.input_scale is not None:
                    linear.input_scale.sharding_strategy = rep
            self.indexer.k_norm.sharding_strategy = rep

    def _mla_impl(
        self,
        xq: TensorValue,
        kv: TensorValue,
        kv_collection: PagedCacheValues,
        layer_idx: TensorValue,
        input_row_offsets: TensorValue,
        freqs_cis: TensorValue,
        kv_a_proj_layernorm: TensorValue,
        _mla_prefill_metadata: MLAPrefillMetadata | None = None,
        *,
        sparse_indices: TensorValue | None = None,
        sparse_topk_lengths: TensorValue | None = None,
        sparse_attn_sink: TensorValue | None = None,
        sparse_indices_stride: int | None = None,
    ) -> TensorValue:
        attn_kwargs: dict[str, Any] = {
            "q": xq,
            "kv": kv,
            "input_row_offsets": input_row_offsets,
            "freqs_cis": freqs_cis,
            "kv_norm_gamma": kv_a_proj_layernorm,
            "kv_params": self.kv_params,
            "kv_collection": kv_collection,
            "layer_idx": layer_idx,
            "epsilon": 1e-6,
            "mask_variant": MHAMaskVariant.CAUSAL_MASK,
            "scale": self.scale,
            "v_head_dim": self.v_head_dim,
            "quant_config": self.quant_config,
        }

        w_k, w_k_scale = self.w_k
        w_uk, w_uk_scale = self.w_uk
        w_uv, w_uv_scale = self.w_uv
        if self.graph_mode in ["prefill", "auto"]:
            if _mla_prefill_metadata is None:
                mla_prefill_metadata = self.create_mla_prefill_metadata(
                    input_row_offsets, kv_collection
                )
            else:
                mla_prefill_metadata = _mla_prefill_metadata

            attn_kwargs["buffer_row_offsets"] = (
                mla_prefill_metadata.buffer_row_offsets
            )
            attn_kwargs["cache_offsets"] = mla_prefill_metadata.cache_offsets
            attn_kwargs["buffer_length"] = (
                mla_prefill_metadata.buffer_lengths.to(DeviceRef.CPU())
            )
            attn_kwargs["w_k"] = w_k
            attn_kwargs["w_k_scale"] = w_k_scale
            attn_kwargs["w_uv"] = w_uv
            attn_kwargs["w_uv_scale"] = w_uv_scale

        if self.graph_mode in ["decode", "auto"]:
            attn_kwargs["w_uk"] = w_uk
            attn_kwargs["w_uk_scale"] = w_uk_scale
            attn_kwargs["w_uv"] = w_uv
            attn_kwargs["w_uv_scale"] = w_uv_scale
            assert kv_collection.attention_dispatch_metadata is not None
            attn_kwargs["scalar_args"] = (
                kv_collection.attention_dispatch_metadata
            )

        sparse_kw: dict[str, Any] = {}
        if sparse_indices is not None:
            sparse_kw = {
                "sparse_indices": sparse_indices,
                "sparse_topk_lengths": sparse_topk_lengths,
                "sparse_attn_sink": sparse_attn_sink,
                "sparse_indices_stride": sparse_indices_stride,
            }

        if self.graph_mode == "prefill":
            result = mla_prefill_graph(**attn_kwargs)
        elif self.graph_mode == "decode":
            result = mla_decode_graph(**attn_kwargs, **sparse_kw)
        else:
            result = mla_prefill_decode_graph(**attn_kwargs, **sparse_kw)

        return result.reshape((-1, self.n_heads * self.v_head_dim))

    def __call__(  # type: ignore[override]
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        indexer_kv_collection: PagedCacheValues,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
        mla_prefill_metadata: MLAPrefillMetadata | None = None,
    ) -> TensorValue:
        wqkv, wqkv_scale = self.wqkv
        qkv = quantized_matmul(
            x=x,
            weight=wqkv,
            weight_scale=wqkv_scale,
            input_scale=None,
            quant_config=self.quant_config,
        )

        q_a_out, kv = ops.split(
            qkv, [self.q_lora_rank, self.cache_head_dim], axis=1
        )

        q_a_normed = self.q_a_layernorm(q_a_out)

        xq = quantized_matmul(
            x=q_a_normed,
            weight=self.q_b_proj,
            weight_scale=self.q_b_proj_scale,
            input_scale=None,
            quant_config=self.quant_config,
        )

        xq = xq.reshape((-1, self.n_heads, self.qk_head_dim))

        freqs_cis = ops.cast(freqs_cis, xq.dtype).to(xq.device)

        topk_indices = self.indexer(
            x,
            q_a_normed,
            freqs_cis,
            input_row_offsets,
            indexer_kv_collection,
            layer_idx,
            mask_variant=MHAMaskVariant.CAUSAL_MASK
            if self.graph_mode in ["prefill", "auto"]
            else MHAMaskVariant.NULL_MASK,
        )
        # TODO: restore mapping to physical cache blocks
        topk_indices = ops.where(
            (topk_indices != -1),
            topk_indices,
            ops.broadcast_to(
                ops.constant(
                    0, dtype=topk_indices.dtype, device=topk_indices.device
                ),
                topk_indices.shape,
            ),
        )

        batch_dim = kv_collection.lookup_table.shape[0]
        sparse_topk_lengths = ops.broadcast_to(
            ops.constant(
                self.indexer.index_topk,
                dtype=DType.int32,
                device=xq.device,
            ),
            (batch_dim,),
        )
        sparse_attn_sink = ops.broadcast_to(
            ops.constant(float("-inf"), dtype=DType.float32, device=xq.device),
            (self.n_heads,),
        )

        attn_out = self._mla_impl(
            xq,
            kv,
            kv_collection,
            layer_idx,
            input_row_offsets,
            freqs_cis,
            self.kv_a_proj_layernorm,
            mla_prefill_metadata,
            sparse_indices=topk_indices,
            sparse_topk_lengths=sparse_topk_lengths,
            sparse_attn_sink=sparse_attn_sink,
            sparse_indices_stride=self.indexer.index_topk,
        )

        return self.o_proj(attn_out)

    def shard(  # type: ignore[override]
        self, devices: Iterable[DeviceRef]
    ) -> list[SparseLatentAttentionWithRopeFp8]:
        """Replicate full weights across devices (data parallel)."""
        if not self.sharding_strategy:
            raise ValueError(
                "SparseLatentAttentionWithRopeFp8 cannot be sharded because no "
                "sharding strategy was provided."
            )
        if self.sharding_strategy.is_tensor_parallel:
            raise ValueError(
                "SparseLatentAttentionWithRopeFp8 only supports replicate "
                "sharding for data parallelism."
            )
        if not self.sharding_strategy.is_replicate:
            raise ValueError(
                "Only replicate sharding is supported for "
                "SparseLatentAttentionWithRopeFp8"
            )

        q_a_proj_shards = self.q_a_proj.shard(devices)
        q_a_proj_scale_shards = self.q_a_proj_scale.shard(devices)
        q_a_layernorm_weight_shards = self.q_a_layernorm.weight.shard(devices)
        q_b_proj_shards = self.q_b_proj.shard(devices)
        q_b_proj_scale_shards = self.q_b_proj_scale.shard(devices)

        kv_a_proj_layernorm_shards = self.kv_a_proj_layernorm.shard(devices)
        kv_a_proj_with_mqa_shards = self.kv_a_proj_with_mqa.shard(devices)
        kv_a_proj_with_mqa_scale_shards = self.kv_a_proj_with_mqa_scale.shard(
            devices
        )
        kv_b_proj_shards = self.kv_b_proj.shard(devices)
        kv_b_proj_scale_shards = self.kv_b_proj_scale.shard(devices)
        o_proj_weight_shards = self.o_proj.weight.shard(devices)

        if self.o_proj.input_scale is not None:
            o_proj_scale_shards = self.o_proj.input_scale.shard(devices)
        if self.o_proj.weight_scale is not None:
            o_proj_weight_scale_shards = self.o_proj.weight_scale.shard(devices)

        indexer_wq_b_shards = self.indexer.wq_b.shard(devices)
        indexer_wk_shards = self.indexer.wk.shard(devices)
        indexer_weights_proj_shards = self.indexer.weights_proj.shard(devices)
        indexer_k_norm_shards = self.indexer.k_norm.shard(devices)

        replicas: list[SparseLatentAttentionWithRopeFp8] = []
        for shard_idx, device in enumerate(devices):
            replica = SparseLatentAttentionWithRopeFp8(
                rope=self.rope,
                num_attention_heads=self.n_heads,
                num_key_value_heads=self.num_key_value_heads,
                hidden_size=self.hidden_size,
                kv_params=self.kv_params,
                quant_config=self.quant_config,
                devices=[device],
                graph_mode=self.graph_mode,
                linear_cls=self.linear_cls,
                scale=self._scale,
                q_lora_rank=self.q_lora_rank,
                kv_lora_rank=self.kv_lora_rank,
                qk_nope_head_dim=self.qk_nope_head_dim,
                qk_rope_head_dim=self.qk_rope_head_dim,
                v_head_dim=self.v_head_dim,
                buffer_size=self.BUFFER_TOK_SIZE,
                norm_dtype=self.norm_dtype,
            )

            replica.q_a_proj = q_a_proj_shards[shard_idx]
            replica.q_a_proj_scale = q_a_proj_scale_shards[shard_idx]
            replica.q_a_layernorm.weight = q_a_layernorm_weight_shards[
                shard_idx
            ]
            replica.q_b_proj = q_b_proj_shards[shard_idx]
            replica.q_b_proj_scale = q_b_proj_scale_shards[shard_idx]

            replica.kv_a_proj_layernorm = kv_a_proj_layernorm_shards[shard_idx]
            replica.kv_a_proj_with_mqa = kv_a_proj_with_mqa_shards[shard_idx]
            replica.kv_a_proj_with_mqa_scale = kv_a_proj_with_mqa_scale_shards[
                shard_idx
            ]
            replica.kv_b_proj = kv_b_proj_shards[shard_idx]
            replica.kv_b_proj_scale = kv_b_proj_scale_shards[shard_idx]
            replica.o_proj.weight = o_proj_weight_shards[shard_idx]
            if self.o_proj.input_scale is not None:
                replica.o_proj.input_scale = o_proj_scale_shards[shard_idx]
            if self.o_proj.weight_scale is not None:
                replica.o_proj.weight_scale = o_proj_weight_scale_shards[
                    shard_idx
                ]

            replica.indexer.wq_b = indexer_wq_b_shards[shard_idx]
            replica.indexer.wk = indexer_wk_shards[shard_idx]
            replica.indexer.weights_proj = indexer_weights_proj_shards[
                shard_idx
            ]
            replica.indexer.k_norm = indexer_k_norm_shards[shard_idx]

            replicas.append(replica)

        return replicas


class DataParallelSparseLatentAttentionWithRopeFp8(
    SparseLatentAttentionWithRopeFp8
):
    """Data-parallel sparse FP8 MLA: per-device optional sparse index tensors."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if not self.devices:
            raise ValueError("devices cannot be None or empty")

        num_devices = len(self.devices)
        self.sharding_strategy = ShardingStrategy.replicate(num_devices)
        self.list_of_attentions = self.shard(self.devices)

    def create_mla_prefill_metadata(  # type: ignore[override]
        self,
        input_row_offsets_: list[TensorValue],
        kv_collections: list[PagedCacheValues],
    ) -> list[MLAPrefillMetadata]:
        """Creates per-device FP8 MLA prefill metadata for data-parallel execution."""
        multi_mla_prefill_metadata: list[MLAPrefillMetadata] = []

        for input_row_offsets, kv_collection in zip(
            input_row_offsets_, kv_collections, strict=True
        ):
            multi_mla_prefill_metadata.append(
                super().create_mla_prefill_metadata(
                    input_row_offsets, kv_collection
                )
            )

        return multi_mla_prefill_metadata

    def __call__(  # type: ignore[override]
        self,
        layer_idx: TensorValue,
        xs: Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue],
        kv_collections: Sequence[PagedCacheValues],
        indexer_kv_collections: Sequence[PagedCacheValues],
        freqs_cis: list[TensorValue],
        input_row_offsets: Sequence[TensorValue],
        mla_prefill_metadata: list[MLAPrefillMetadata] | None = None,
    ) -> list[TensorValue]:
        if not self.devices:
            raise ValueError("devices cannot be None or empty")

        n = len(self.devices)
        if not (
            len(xs)
            == len(kv_collections)
            == len(indexer_kv_collections)
            == len(freqs_cis)
            == len(input_row_offsets)
            == n
        ):
            raise ValueError(
                "xs, kv_collections, indexer_kv_collections, freqs_cis, and "
                f"input_row_offsets must all have length equal to number of devices "
                f"({n})"
            )

        outs: list[TensorValue] = []
        for i in range(n):
            if xs[i].shape[0] == 0:
                outs.append(xs[i])
                continue

            mla_prefill_metadata_i: MLAPrefillMetadata | None
            if (
                mla_prefill_metadata is not None
                and len(mla_prefill_metadata) == n
            ):
                mla_prefill_metadata_i = mla_prefill_metadata[i]
            else:
                assert (
                    mla_prefill_metadata is None
                    or len(mla_prefill_metadata) == 0
                )
                mla_prefill_metadata_i = None

            outs.append(
                self.list_of_attentions[i](
                    layer_idx=layer_idx,
                    x=xs[i],
                    kv_collection=kv_collections[i],
                    indexer_kv_collection=indexer_kv_collections[i],
                    freqs_cis=freqs_cis[i],
                    input_row_offsets=input_row_offsets[i],
                    mla_prefill_metadata=mla_prefill_metadata_i,
                )
            )
        return outs
