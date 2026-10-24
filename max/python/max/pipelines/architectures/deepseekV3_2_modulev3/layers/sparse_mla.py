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
"""Sparse Multi-Latent Attention for DeepseekV3.2, in the ModuleV3 API."""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn.common_layers import quant_ops
from max.experimental.nn.common_layers.functional_kernels import (
    mla_decode_graph,
    mla_prefill_decode_graph,
)
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.nn.common_layers.multi_latent_attention import (
    MLAPrefillMetadata,
)
from max.experimental.nn.common_layers.quant_mla import (
    QuantizedLatentAttention,
    tensor_parallel_latent_attention_with_rope,
)
from max.experimental.nn.common_layers.quant_tensor import FP8BlockTensor
from max.experimental.sharding import DeviceMapping
from max.experimental.tensor import Tensor
from max.graph import TensorValue, ops
from max.nn.attention import MHAMaskVariant
from max.nn.quant_config import QuantConfig

from ...deepseekV3_2.layers.sparse_mla import sparse_selection_width
from .indexer import Indexer

logger = logging.getLogger("max.pipelines")


# Head counts that route prefill through the combined prefill/decode op
# unconditionally (landed behavior; the sparse-prefill kernel handles 128 or
# any multiple of 8 in (0, 64]).  Other counts fall back to decode rather
# than tripping the kernel's comptime assert.
_SPARSE_PREFILL_SUPPORTED_HEADS = (64, 128)
# GLM 5.2's TP-sharded counts (64 // {8, 4, 2}) route to the combined op over a
# bfloat16, float8_e4m3fn or NVFP4 (uint8) latent cache.  The combined op's
# prefill arm takes the sparse-prefill kernel for each cache dtype, so the
# absorbed sparse path -- not the dense unabsorbed FP8 prefill, whose extra
# Q/K/V requantization cost accuracy -- runs for FP8-cache prefill at these
# head counts.
_SPARSE_PREFILL_TP_SHARDED_HEADS = (8, 16, 32)

# Master gate for the sparse-MLA *prefill* kernel. When False, prefill is
# routed through the sparse *decode* kernel (the same fallback used for
# unsupported head counts) instead of the sparse-prefill kernel.
_ENABLE_SPARSE_MLA_PREFILL_KERNEL = True

# Dedup the one-time "prefill kernel gated off" notice (guard runs per layer).
_WARNED_PREFILL_KERNEL_DISABLED: set[str] = set()

# Head counts already warned about; the guard runs per layer, so dedup the log.
_WARNED_FALLBACK_HEADS: set[int] = set()


def _warn_prefill_kernel_disabled() -> None:
    """Log once (deduped across layers) that the prefill kernel is off."""
    if "logged" in _WARNED_PREFILL_KERNEL_DISABLED:
        return
    _WARNED_PREFILL_KERNEL_DISABLED.add("logged")
    logger.info(
        "Sparse MLA prefill kernel disabled "
        "(_ENABLE_SPARSE_MLA_PREFILL_KERNEL=False); routing prefill "
        "through the sparse decode kernel."
    )


def _sparse_prefill_head_count_supported(
    n_heads: int, cache_dtype: DType
) -> bool:
    """Whether prefill may route through the combined prefill/decode op."""
    if n_heads in _SPARSE_PREFILL_SUPPORTED_HEADS:
        return True
    return n_heads in _SPARSE_PREFILL_TP_SHARDED_HEADS and cache_dtype in (
        DType.bfloat16,
        DType.float8_e4m3fn,
        DType.uint8,
    )


def _placed_like(scale: Tensor, weight: Tensor) -> Tensor:
    """Places a head-major gathered scale with its weight's head sharding.

    ``scale`` and ``weight`` share rank and lead with the head axis, so the
    weight's mapping hands each device exactly its own heads' scales. A
    replicated-to-sharded move is a local slice, not a collective.
    """
    if weight.mesh.num_devices == 1:
        return scale
    return scale.to(weight.mapping)


def _zero_prefill_plan(
    input_row_offsets: TensorValue,
) -> tuple[TensorValue, TensorValue, TensorValue]:
    """Returns zeroed outputs shaped like a one-chunk dense prefill plan."""
    if input_row_offsets.dtype != DType.uint32:
        raise ValueError("sparse metadata requires uint32 input row offsets")
    offsets = ops.broadcast_to(
        ops.constant(0, DType.uint32, device=input_row_offsets.device),
        (1, input_row_offsets.shape[0]),
    )
    lengths = ops.broadcast_to(
        ops.constant(0, DType.int32, device=input_row_offsets.device), (1,)
    )
    return offsets, offsets, lengths


_sparse_prefill_plan = F.functional(_zero_prefill_plan)


class QuantizedSparseLatentAttentionWithRope(QuantizedLatentAttention):
    """Latent attention with a lightning indexer selecting the keys to attend.

    Unifies what V2 split across a bf16 and an FP8 class: the quantized path is
    selected by ``quant_config``, exactly as in the dense base. ``skip_topk``
    layers carry no indexer weights and reuse the previous full layer's
    selection (cross-layer index sharing).
    """

    def __init__(
        self,
        *,
        index_n_heads: int = 64,
        index_head_dim: int = 128,
        index_topk: int = 2048,
        skip_topk: bool = False,
        indexer_rope_interleave: bool = False,
        indexer_quant_config: QuantConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if self.q_lora_rank is None:
            raise ValueError(
                "q_lora_rank is required for sparse latent attention"
            )
        self._b_scale_granularity = 0
        if self.quantized:
            # A head's rows need not align with the on-disk scale block:
            # GLM-5.x has qk_nope + v = 448 against a 128-row block, so head
            # boundaries fall mid-block. The kernel is then driven at the
            # finest granularity where every per-head chunk sits inside one
            # on-disk block, and the scales are gathered to match. When the
            # rows do divide the block (DeepSeek-V3.2) this is just block_m
            # and the base class's reshape already produces the right layout.
            assert self.weight_block_size is not None
            block_m = int(self.weight_block_size[0])
            per_head = self.qk_nope_head_dim + self.v_head_dim
            residue = per_head % block_m
            if residue == 0 and self.qk_nope_head_dim % block_m == 0:
                self._b_scale_granularity = block_m
            else:
                self._b_scale_granularity = math.gcd(residue, block_m)

        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.skip_topk = skip_topk
        self.indexer_rope_interleave = indexer_rope_interleave

        # ``shared`` layers carry no indexer weights, and instead reuse the
        # previous full layer's top-k selection.
        self.indexer: Indexer | None
        if skip_topk:
            self.indexer = None
        else:
            self.indexer = Indexer(
                dim=self.hidden_size,
                index_n_heads=index_n_heads,
                index_head_dim=index_head_dim,
                qk_rope_head_dim=self.qk_rope_head_dim,
                index_topk=index_topk,
                q_lora_rank=self.q_lora_rank,
                rope_interleaved=indexer_rope_interleave,
                quant_config=indexer_quant_config,
            )

    @property
    def _ragged_scales(self) -> bool:
        """Whether per-head rows straddle the on-disk scale block.

        Only an FP8 block-quantized layer has an on-disk scale block at all:
        a bf16 or NVFP4 layer carries no ``weight_block_size``, so the
        question does not arise and the per-head gather must stay unused.
        """
        if not self.quantized:
            return False
        assert self.weight_block_size is not None
        return self._b_scale_granularity != self.weight_block_size[0]

    def _gather_per_head_scale(
        self, start_row_offset: int, n_rows: int
    ) -> Tensor:
        """Gathers per-head B-scale chunks from the flat on-disk scale.

        For head ``h`` and chunk ``k`` of ``g`` rows, the on-disk scale row is
        ``(h * per_head_row + start_row_offset + k * g) // block_m``. When
        ``block_k > g`` the gathered columns are replicated ``block_k // g``
        times so the kernel -- driven at granularity ``g`` -- sees a matching
        block count along the on-disk K axis.

        Returns a ``[H, n_chunks, n_cols_kernel]`` tensor over all ``H`` heads,
        replicated on the scale's mesh. Under tensor parallelism the caller
        places it like its weight (see :func:`_placed_like`).
        """
        assert self.weight_block_size is not None
        assert isinstance(self.kv_b_proj, FP8BlockTensor)
        scale = self.kv_b_proj.weight_scale_inv
        index_mapping = DeviceMapping.replicated(scale.mesh)

        g = self._b_scale_granularity
        block_m, block_k = (int(b) for b in self.weight_block_size)
        per_head_row = self.qk_nope_head_dim + self.v_head_dim
        n_chunks = -(-n_rows // g)

        heads = np.arange(self.n_heads, dtype=np.int32)
        chunks = np.arange(n_chunks, dtype=np.int32)
        row_indices = (
            (heads[:, None] * per_head_row + start_row_offset + chunks * g)
            // block_m
        ).reshape(-1)
        gathered = F.gather(
            scale,
            F.constant(row_indices, DType.int32, device=index_mapping),
            axis=0,
        )

        n_cols_on_disk = int(scale.shape[1])
        gathered = gathered.reshape((self.n_heads, n_chunks, n_cols_on_disk))
        col_repeat = block_k // g
        if col_repeat == 1:
            return gathered
        col_indices = np.repeat(
            np.arange(n_cols_on_disk, dtype=np.int32), col_repeat
        )
        return F.gather(
            gathered,
            F.constant(col_indices, DType.int32, device=index_mapping),
            axis=2,
        )

    @property
    def w_uk(self) -> tuple[Tensor, Tensor | None]:
        """Decode K-projection. Scale is ``[H, N_blk, K_blk]`` for ``Q @ w_uk``."""
        if not self._ragged_scales:
            return super().w_uk
        w_uk = self._kv_b_proj_weight[..., : self.qk_nope_head_dim].transpose(
            0, 1
        )
        scale = self._gather_per_head_scale(
            start_row_offset=0, n_rows=self.qk_nope_head_dim
        ).transpose(1, 2)
        return w_uk, _placed_like(scale, w_uk)

    @property
    def w_uv(self) -> tuple[Tensor, Tensor | None]:
        """Decode V-projection. The gather already yields ``[H, N_blk, K_blk]``."""
        if not self._ragged_scales:
            return super().w_uv
        w_uv = self._kv_b_proj_weight[..., self.qk_nope_head_dim :].permute(
            [1, 2, 0]
        )
        scale = self._gather_per_head_scale(
            start_row_offset=self.qk_nope_head_dim, n_rows=self.v_head_dim
        )
        return w_uv, _placed_like(scale, w_uv)

    @property
    def w_k(self) -> tuple[Tensor, Tensor | None]:
        """Prefill K-projection, flattened to ``[H*Dn, R]``."""
        if not self._ragged_scales:
            return super().w_k
        w_k = (
            self._kv_b_proj_weight[..., : self.qk_nope_head_dim]
            .permute([1, 2, 0])
            .reshape((-1, self.kv_lora_rank))
        )
        scale = self._gather_per_head_scale(
            start_row_offset=0, n_rows=self.qk_nope_head_dim
        ).reshape((-1, self.kv_lora_rank // self._b_scale_granularity))
        return w_k, _placed_like(scale, w_k)

    def create_mla_prefill_metadata(
        self, input_row_offsets: Tensor, kv_collection: PagedCacheValues
    ) -> MLAPrefillMetadata:
        """Creates unused dense metadata fields for the sparse graph signature.

        Sparse kernels gather physical cache rows in both prefill and decode,
        so the dense planner, whose budget a valid batch's page-aligned
        aggregate context can exceed, never runs.
        """
        replicated = DeviceMapping.replicated(input_row_offsets.mesh)
        buffer_row_offsets, cache_offsets, buffer_lengths = (
            value.rebind_mapping(replicated)
            for value in _sparse_prefill_plan(input_row_offsets)
        )
        return MLAPrefillMetadata(
            buffer_row_offsets=buffer_row_offsets,
            cache_offsets=cache_offsets,
            buffer_lengths=buffer_lengths,
        )

    def _effective_graph_mode(self) -> str:
        """Graph mode after the sparse-prefill kernel's head-count gating."""
        mode = self.graph_mode
        if mode == "decode":
            return mode
        if not _ENABLE_SPARSE_MLA_PREFILL_KERNEL:
            _warn_prefill_kernel_disabled()
            return "decode"
        if not _sparse_prefill_head_count_supported(
            self.n_heads, self.kv_params.dtype
        ):
            if self.n_heads not in _WARNED_FALLBACK_HEADS:
                _WARNED_FALLBACK_HEADS.add(self.n_heads)
                logger.warning(
                    "Sparse MLA prefill does not support %d query heads "
                    "(supported: %s); falling back to the slower decode path "
                    "for prefill. This usually means tensor-parallel attention "
                    "sharded the head count below a supported value.",
                    self.n_heads,
                    _SPARSE_PREFILL_SUPPORTED_HEADS,
                )
            return "decode"
        return mode

    def _mla_impl(
        self,
        xq: Tensor,
        kv: Tensor,
        kv_collection: PagedCacheValues,
        layer_idx: Tensor,
        input_row_offsets: Tensor,
        freqs_cis: Tensor,
        kv_norm_gamma: Tensor,
        _mla_prefill_metadata: MLAPrefillMetadata | None = None,
        epsilon: float = 1e-6,
        *,
        sparse_indices: Tensor | None = None,
        sparse_indices_stride: int | None = None,
        selection_lag: int = 0,
    ) -> Tensor:
        if sparse_indices is None:
            raise ValueError("sparse attention requires physical sparse rows")
        attn_kwargs: dict[str, Any] = {
            "q": xq,
            "kv": kv,
            "input_row_offsets": input_row_offsets,
            "freqs_cis": freqs_cis,
            "kv_norm_gamma": kv_norm_gamma,
            "kv_params": self.kv_params,
            "kv_collection": kv_collection,
            "layer_idx": layer_idx,
            "epsilon": epsilon,
            "mask_variant": MHAMaskVariant.CAUSAL_MASK,
            "scale": self.scale,
            "v_head_dim": self.v_head_dim,
        }
        if self.quantized:
            attn_kwargs["quant_config"] = self.quant_config
            attn_kwargs["scale_granularity_override"] = (
                self._b_scale_granularity
            )

        w_k, w_k_scale = self.w_k
        w_uk, w_uk_scale = self.w_uk
        w_uv, w_uv_scale = self.w_uv

        effective_graph_mode = self._effective_graph_mode()

        if effective_graph_mode in ("prefill", "auto"):
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
                mla_prefill_metadata.buffer_lengths.to(CPU())
            )
            attn_kwargs["w_k"] = w_k
            attn_kwargs["w_uv"] = w_uv
            if self.quantized:
                attn_kwargs["w_k_scale"] = w_k_scale
                attn_kwargs["w_uv_scale"] = w_uv_scale

        # Unlike the dense base, the absorbed decode weights go in for every
        # graph mode: the sparse path runs the absorbed kernel for prefill too.
        attn_kwargs["w_uk"] = w_uk
        attn_kwargs["w_uv"] = w_uv
        if self.quantized:
            attn_kwargs["w_uk_scale"] = w_uk_scale
            attn_kwargs["w_uv_scale"] = w_uv_scale
        assert kv_collection.attention_dispatch_metadata is not None
        attn_kwargs["scalar_args"] = kv_collection.attention_dispatch_metadata
        assert kv_collection.mla_num_partitions is not None
        attn_kwargs["num_partitions_scalar"] = kv_collection.mla_num_partitions

        sparse_kw: dict[str, Any] = {
            "sparse_indices": sparse_indices,
            "physical_indices": True,
            "sparse_indices_stride": sparse_indices_stride,
            "selection_lag": selection_lag,
            # The op always passes a no-op sink; the AMD sparse arms
            # reject one unless it is declared inert.
            "sparse_attn_sink_inert": True,
        }

        if effective_graph_mode == "decode":
            result = mla_decode_graph(**attn_kwargs, **sparse_kw)
        else:
            # TODO(KERN-3198): "prefill" uses the combined op because
            # mla_prefill_graph doesn't support sparse args yet.
            result = mla_prefill_decode_graph(**attn_kwargs, **sparse_kw)
        result = result.rebind_mapping(xq.mapping)

        return result.reshape((result.shape[0], self.n_heads * self.v_head_dim))

    def forward(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        indexer_kv_collection: PagedCacheValues,
        freqs_cis: Tensor,
        layer_idx: Tensor,
        input_row_offsets: Tensor,
        mla_prefill_metadata: MLAPrefillMetadata | None = None,
        prev_topk_indices: Tensor | None = None,
        reuse_prev_topk: bool = False,
        indexer_layer_idx: Tensor | None = None,
        selection_lag: int = 0,
    ) -> tuple[Tensor, Tensor]:
        """Returns the attention output and the top-k selection it used.

        ``selection_lag`` is the number of positions between the token a
        reused selection was computed for and this call's token.
        """
        if selection_lag != 0 and not reuse_prev_topk:
            raise ValueError(
                f"selection_lag={selection_lag} describes a reused selection,"
                " but reuse_prev_topk is False."
            )
        q_lora_rank = self.q_lora_rank
        assert q_lora_rank is not None  # enforced in __init__
        qkv = quant_ops.matmul(x, self.wqkv)
        q_a_out, kv = qkv.split([q_lora_rank, self.cache_head_dim], axis=1)
        q_a_normed = self.q_a_layernorm(q_a_out)
        xq = quant_ops.matmul(q_a_normed, self.q_b_proj)

        # Explicit token dim: the shape rule cannot prove a ``-1`` leading
        # dim against a per-shard (data-parallel) token count.
        xq = xq.reshape((xq.shape[0], self.n_heads, self.qk_head_dim))

        freqs_cis = F.cast(freqs_cis, xq.dtype)

        if self.indexer is not None and not reuse_prev_topk:
            # ``full`` layer: run the lightning indexer and select top-k keys.
            topk_indices = self.indexer(
                x,
                q_a_normed,
                freqs_cis,
                input_row_offsets,
                indexer_kv_collection,
                # The indexer cache holds only ``full`` layers, so its slot
                # is this layer's rank among them, not ``layer_idx``.
                layer_idx if indexer_layer_idx is None else indexer_layer_idx,
                kv_collection,
                MHAMaskVariant.CAUSAL_MASK,
            )
        else:
            # ``shared`` layer: reuse the previous full layer's top-k selection
            # (cross-layer index sharing). Physical rows name a page and
            # offset, not a layer, so they are valid for this layer's own MLA
            # cache too.
            if prev_topk_indices is None:
                raise ValueError(
                    "Shared (skip_topk) sparse attention layers require top-k "
                    "indices from a previous full indexer layer."
                )
            topk_indices = prev_topk_indices

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
            sparse_indices_stride=sparse_selection_width(
                self.kv_params, self.index_topk
            ),
            selection_lag=selection_lag,
        )

        return self.o_proj(attn_out), topk_indices


def tensor_parallel_sparse_latent_attention_with_rope(
    layer: QuantizedSparseLatentAttentionWithRope,
    num_devices: int = 1,
) -> QuantizedSparseLatentAttentionWithRope:
    """Shards sparse latent attention along the TP axis.

    The dense weights follow the base layout; the indexer stays replicated, so
    every device computes the same top-k selection over its own key cache.

    Placements are assigned even on a single device (matching the dense base);
    ``num_devices`` only gates the changes that depend on the head axis really
    being split.
    """
    tensor_parallel_latent_attention_with_rope(layer)

    if num_devices > 1 and layer._ragged_scales:
        # The per-head scale gather indexes ``kv_b_proj``'s scale by global
        # head, and per-head rows straddle its blocks, so a row shard of the
        # scale is not a head shard. Keep the (few-dozen-row) scale whole on
        # every device; the gathered per-head scales are then sharded with
        # their weights.
        kv_b_proj = layer.kv_b_proj
        assert isinstance(kv_b_proj, FP8BlockTensor)
        kv_b_proj.weight_scale_inv = kv_b_proj.weight_scale_inv.to(
            DeviceMapping.replicated(kv_b_proj.mesh)
        )

    return layer
