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
from collections.abc import Mapping
from typing import Any

import numpy as np
from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.graph import TensorValue, ops
from max.nn.attention_mask import MHAMaskVariant
from max.nn.common_layers import quant_ops
from max.nn.common_layers.context_parallel import (
    combine_partial_attention,
    dcp_group_heads,
    dcp_ranks,
    dcp_view,
    gather_columns,
    split_selection,
    to_dcp_mesh,
)
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.common_layers.mesh_axis import DCP, TP
from max.nn.common_layers.multi_latent_attention import (
    MLAPrefillMetadata,
)
from max.nn.common_layers.quant_mla import (
    QuantizedLatentAttention,
    tensor_parallel_latent_attention_with_rope,
)
from max.nn.common_layers.quant_tensor import FP8BlockTensor
from max.nn.kernels import (
    hisparse_gather_group,
    mla_decode_graph,
    mla_decode_graph_cp,
    mla_prefill_decode_graph,
    store_mla_latent_ragged,
)
from max.nn.quant_config import QuantConfig
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping, NamedMapping, Sharded, Unknown

from ...deepseekV3_2.layers.sparse_mla import sparse_selection_width
from ..model_config import HiSparseInputs
from .indexer import Indexer
from .prefill_context_parallel import (
    PrefillContextParallel,
    PrefillContextParallelStep,
)

logger = logging.getLogger("max.pipelines")

# The latent RMSNorm epsilon, shared by the attention op and the standalone
# latent store so both write the same cache rows.
_LATENT_NORM_EPS = 1e-6


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
    replicated-to-sharded move is a local slice, not a collective. A weight
    on a DCP view (decode context parallelism smaller than TP) takes its
    scale onto the view first, which moves no data.
    """
    if weight.mesh.num_devices == 1:
        return scale
    return to_dcp_mesh(scale, weight.mesh).to(weight.mapping)


def _head_sharded(t: Tensor) -> Tensor:
    """Shards ``t``'s leading head axis over every attention TP rank.

    On a DCP view that is the ``tp`` axis of groups and each group's
    ``dcp`` axis; on a model mesh, its ``tp`` axis. A replicated-to-sharded
    move along an axis is a local slice, not a collective.
    """
    mesh = t.mesh
    names: tuple[str, ...] = (TP, DCP) if DCP in mesh.axis_names else (TP,)
    if not any(name in mesh.axis_names for name in names):
        names = mesh.axis_names
    return t.to(
        DeviceMapping(
            mesh,
            tuple(
                Sharded(0) if name in names else p
                for name, p in zip(mesh.axis_names, t.placements, strict=True)
            ),
        )
    )


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
        dcp_size: int = 1,
        dcp_interleave: int = 1,
        dcp_q_replicate: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.dcp_size = dcp_size
        self.dcp_interleave = dcp_interleave
        # Under DCP every rank attends all of its group's heads. With
        # ``dcp_q_replicate`` it computes them itself from replicated ``q_b``
        # and ``w_kc`` weights; otherwise each rank absorbs its own TP heads
        # and the group all-gathers the absorbed queries, as vLLM's
        # ``dcp_q_replicate=False`` does.
        self.dcp_q_replicate = dcp_q_replicate

        if self.q_lora_rank is None:
            raise ValueError(
                "q_lora_rank is required for sparse latent attention"
            )
        if dcp_size > 1 and not dcp_q_replicate and self.quantized:
            raise NotImplementedError(
                "decode context parallelism with a gathered query absorbs "
                "each rank's heads with BF16 MLA up-projections; FP8 ones "
                "need dcp_q_replicate=True"
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
                dcp_size=dcp_size,
                dcp_interleave=dcp_interleave,
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
        if mode == "decode" or self.dcp_size > 1:
            # Under decode context parallelism every step runs the decode
            # kernel, the one that returns the LSE the ranks merge by and
            # reads per-token selection lengths.
            return "decode"
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
        epsilon: float = _LATENT_NORM_EPS,
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

        if self.dcp_size > 1:
            # Under DCP every MTP draft step runs its own indexer
            # (``Glm5_1MTPProposer`` declares no reuse), so no caller lags a
            # selection here. A lagged reuse measurably lowered acceptance
            # under DCP for a reason not yet understood, so it stays refused.
            if selection_lag != 0:
                raise NotImplementedError(
                    "decode context parallelism takes no selection_lag: a "
                    "draft step must run its own indexer, because reusing a "
                    "lagged selection lowered MTP acceptance under DCP for an "
                    "unexplained reason"
                )
            return self._mla_dcp(attn_kwargs, sparse_indices, xq)

        # The split-K count only the non-DCP ops take; the DCP op sizes its
        # partitions from ``split_len_bound``.
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

    def _mla_dcp(
        self,
        attn_kwargs: Mapping[str, Any],
        selection: Tensor,
        xq: Tensor,
    ) -> Tensor:
        """Attends this rank's share of the keys and merges the ranks.

        ``selection`` packs each token's owned rows with their count (see
        ``context_parallel.merge_topk``); over an NVFP4 cache the rows are
        the value rows then their scale-pool rows. Every rank attends all of
        its group's heads over its own rows, stores only the positions it
        owns, and returns its partial output and LSE; the merge leaves each
        rank its TP heads for the head-sharded ``o_proj``.

        With BF16 weights the ranks merge the latent outputs and each applies
        its own heads' ``w_vc`` afterwards, as vLLM and SGLang do; FP8
        weights apply ``w_vc`` in the kernel before the merge.
        """
        width = sparse_selection_width(self.kv_params, self.index_topk)
        rows, lengths = split_selection(selection, width)
        latent = not self.quantized
        # The decode op runs on the model mesh with the caches. A replicated
        # query's weights shard by group on the DCP view, where each device's
        # shard is its group's heads; on the model mesh they are Unknown.
        kwargs = {
            name: to_dcp_mesh(value, xq.mesh)
            if isinstance(value, Tensor) and name.startswith("w_")
            else value
            for name, value in attn_kwargs.items()
        }
        if not self.dcp_q_replicate:
            kwargs["q"] = self._gathered_latent_q(xq, kwargs["w_uk"])
        out, lse = mla_decode_graph_cp(
            **kwargs,
            sparse_indices=rows,
            sparse_lengths=lengths,
            sparse_indices_stride=width,
            # Each rank holds about 1/dcp of the selected rows; the split-K
            # grid only needs a bound, correctness comes from the lengths.
            split_len_bound=-(-self.index_topk // self.dcp_size) + 16,
            dcp_size=self.dcp_size,
            dcp_interleave=self.dcp_interleave,
            dcp_rank=dcp_ranks(xq.mesh, self.dcp_size),
            q_absorbed=not self.dcp_q_replicate,
            latent_output=latent,
        )
        # The partials keep ``xq``'s token placement off the DCP axis, which
        # lets the merged heads keep it for ``o_proj`` on a ``(dp, tp)`` mesh.
        merged = combine_partial_attention(out, lse, self.dcp_size)
        if latent:
            merged = self._apply_w_vc(merged, attn_kwargs["w_uv"])
        return merged.reshape(
            (merged.shape[0], merged.shape[1] * self.v_head_dim)
        )

    def _gathered_latent_q(self, xq: Tensor, w_uk: Tensor) -> Tensor:
        """Absorbs this rank's query heads and all-gathers the group's.

        ``xq`` and ``w_uk`` are head-sharded as under tensor parallelism.
        Each rank projects its heads' ``q_nope`` through ``w_kc`` and keeps
        their unrotated rope part, ``[T, H / tp, kv_lora_rank + rope]``;
        the all-gather leaves every rank its DCP group's heads (all ``H``
        when the group spans TP), which the decode op rotates and attends
        without reading ``w_uk``.
        """
        nope = self.qk_nope_head_dim
        width = self.kv_lora_rank + self.qk_rope_head_dim
        # [H, T, nope] @ [H, nope, kv_lora_rank] -> [H, T, kv_lora_rank]; both
        # sides are head-sharded, so each rank absorbs only its own heads.
        latent = F.matmul(
            F.permute(xq[:, :, :nope], [1, 0, 2]), F.permute(w_uk, [0, 2, 1])
        )
        absorbed = F.concat(
            [F.permute(latent, [1, 0, 2]), xq[:, :, nope:]], axis=2
        )
        local = absorbed.reshape((absorbed.shape[0], absorbed.shape[1] * width))
        gathered = gather_columns(local, self.dcp_size)
        heads = dcp_group_heads(xq.mesh, self.n_heads, self.dcp_size)
        return gathered.reshape((gathered.shape[0], heads, width))

    def _apply_w_vc(self, merged: Tensor, w_uv: Tensor) -> Tensor:
        """Projects each rank's merged latent heads through their ``w_vc``.

        ``merged`` is ``[T, H / tp, kv_lora_rank]``, head-sharded over the
        ``tp`` axis. ``w_uv`` is ``[H, v_head_dim, kv_lora_rank]``:
        head-sharded like ``merged`` with a gathered query; with a
        replicated one, replicated within each DCP group on the mesh's DCP
        view. A ``w_uv`` that is not yet head-sharded is resharded by head
        first, which each rank does by slicing its own rows, with no
        communication.
        """
        w = to_dcp_mesh(_head_sharded(w_uv), merged.mesh)
        # [H, T, kv_lora_rank] @ [H, kv_lora_rank, v] -> [H, T, v]
        out = F.matmul(F.permute(merged, [1, 0, 2]), F.permute(w, [0, 2, 1]))
        return F.permute(out, [1, 0, 2])

    def forward(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        indexer_kv_collection: PagedCacheValues,
        freqs_cis: Tensor,
        layer_idx: Tensor,
        input_row_offsets: Tensor,
        mla_prefill_metadata: MLAPrefillMetadata | None = None,
        prev_topk_indices: Selection | None = None,
        reuse_prev_topk: bool = False,
        indexer_layer_idx: Tensor | None = None,
        selection_lag: int = 0,
        pcp: PrefillContextParallelStep | None = None,
        hisparse: HiSparseInputs | None = None,
        hisparse_group_size: Tensor | None = None,
    ) -> tuple[Tensor, Selection]:
        """Returns the attention output and the top-k selection it used.

        ``selection_lag`` is the number of positions between the token a
        reused selection was computed for and this call's token.

        With ``pcp`` (a predicated step, :meth:`PrefillContextParallelStep.
        predicated`) the layer runs both prefill context parallelism
        (:meth:`_attend_pcp`) and the DCP path (:meth:`_attend`), each on
        its share of the rows (see :meth:`_predicated`), and returns the
        selection split into its DCP and PCP rows' halves.

        ``hisparse_group_size`` is how many consecutive layers, starting at
        this one, read the selection this layer resolves: the gather that
        fills their hot rows runs here, once.
        """
        args: tuple[Any, ...] = (
            x,
            kv_collection,
            indexer_kv_collection,
            freqs_cis,
            layer_idx,
            input_row_offsets,
            mla_prefill_metadata,
            prev_topk_indices,
            reuse_prev_topk,
            indexer_layer_idx,
            selection_lag,
        )
        if pcp is None:
            if isinstance(prev_topk_indices, tuple):
                raise ValueError(
                    "a split selection comes only from a predicated layer"
                )
            attn_out, topk_indices = self._attend(
                *args,
                hisparse=hisparse,
                hisparse_group_size=hisparse_group_size,
            )
            return self.o_proj(attn_out), topk_indices
        if pcp.cut is None:
            raise ValueError(
                "prefill context parallelism runs predicated: build the step"
                " with PrefillContextParallelStep.predicated"
            )
        if selection_lag != 0:
            raise ValueError(
                "prefill context parallelism takes no lagged selection"
            )
        return self._predicated(args, pcp)

    def _predicated(
        self, args: tuple[Any, ...], step: PrefillContextParallelStep
    ) -> tuple[Tensor, Selection]:
        """Runs both PCP and DCP attention, each on its share of the rows.

        A PCP step gives every row to :meth:`_attend_pcp` and none to
        :meth:`_attend`; any other step the reverse. Every op returns before
        launching on zero rows, so the empty path does no GPU work. Both
        paths still take the whole batch's row offsets and caches, so each
        of their ops must size its launch from its row tensor, not from the
        offsets' last entry.
        """
        x, prev = args[0], args[7]
        width = sparse_selection_width(self.kv_params, self.index_topk)
        x_pcp, x_dcp = step.rows_split(x)
        prev_pcp = prev_dcp = None
        pcp = step.plan(args[5], x_pcp, args[1], args[2])
        own = None
        if isinstance(prev, tuple):
            # The last predicated layer's halves: its DCP rows, and each
            # rank's own PCP rows, which this rank's path reads as they are.
            prev_dcp, own = prev
        elif prev is not None:
            prev_pcp, prev_dcp = step.rows_split(prev)
            own = pcp.own_rows(_columns(prev_pcp, width))
        attn_pcp, topk_pcp = self._attend_pcp(
            x_pcp,
            args[1],
            args[2],
            args[3],
            args[4],
            args[5],
            args[6],
            own,
            args[8],
            args[9],
            pcp=pcp,
        )
        # Each rank's own rows and the selection's columns only: the next
        # layer's PCP path reads no others, so padding them to the step's
        # rows or a count column would copy them for nothing.
        topk_pcp = _columns(topk_pcp, width)
        # Each device's rows are its own, as the DCP path's selection is.
        topk_pcp = topk_pcp.rebind_mapping(
            DeviceMapping(topk_pcp.mesh, (Unknown(),) * topk_pcp.mesh.ndim)
        )
        attn_dcp, topk_dcp = self._attend(
            x_dcp,
            args[1],
            args[2],
            args[3],
            args[4],
            args[5],
            args[6],
            prev_dcp,
            args[8],
            args[9],
            args[10],
        )
        topk_pcp, topk_dcp = (
            t if t.dtype == DType.int32 else F.cast(t, DType.int32)
            for t in (topk_pcp, topk_dcp)
        )
        attn = step.rows_join(attn_pcp, attn_dcp, x)
        # The selection stays split: the next layer reads each half on its
        # own path, so joining it would only copy the DCP rows every step.
        return self.o_proj(attn), (topk_dcp, topk_pcp)

    def _attend(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        indexer_kv_collection: PagedCacheValues,
        freqs_cis: Tensor,
        layer_idx: Tensor,
        input_row_offsets: Tensor,
        mla_prefill_metadata: MLAPrefillMetadata | None,
        prev_topk_indices: Tensor | None,
        reuse_prev_topk: bool,
        indexer_layer_idx: Tensor | None,
        selection_lag: int,
        *,
        hisparse: HiSparseInputs | None = None,
        hisparse_group_size: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """The attention before ``o_proj``, and its selection."""
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
        # Under decode context parallelism with a replicated query, ``q_b``
        # shards by DCP group on the mesh's DCP view, and each device
        # computes its group's heads; back on the model mesh they are
        # ``Unknown`` unless one group spans TP.
        q_b_mesh = self.q_b_proj.mesh
        xq = to_dcp_mesh(
            quant_ops.matmul(to_dcp_mesh(q_a_normed, q_b_mesh), self.q_b_proj),
            q_a_normed.mesh,
        )
        n_heads = self.n_heads
        if self.dcp_size > 1 and self.dcp_q_replicate:
            n_heads = dcp_group_heads(xq.mesh, n_heads, self.dcp_size)

        # Explicit token dim: the shape rule cannot prove a ``-1`` leading
        # dim against a per-shard (data-parallel) token count.
        xq = xq.reshape((xq.shape[0], n_heads, self.qk_head_dim))

        freqs_cis = F.cast(freqs_cis, xq.dtype)

        runs_indexer = self.indexer is not None and not reuse_prev_topk
        if runs_indexer:
            # ``full`` layer: run the lightning indexer and select top-k keys.
            assert self.indexer is not None
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
                hisparse,
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

        if hisparse is not None and runs_indexer:
            # Copy this layer's bytes for the misses the indexer layer's
            # resolve queued, from the host mirror into the hot rows, before
            # the attention reads them. The same launch fills the consecutive
            # ``shared`` layers that reuse this selection, so they skip their
            # own gather.
            assert hisparse_group_size is not None
            hisparse_gather_group(
                kv_collection,
                hisparse.hot_table,
                hisparse.host_mirror,
                layer_idx,
                hisparse_group_size,
                hisparse.params,
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
            sparse_indices_stride=sparse_selection_width(
                self.kv_params, self.index_topk
            ),
            selection_lag=selection_lag,
        )

        return attn_out, topk_indices

    def _attend_pcp(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        indexer_kv_collection: PagedCacheValues,
        freqs_cis: Tensor,
        layer_idx: Tensor,
        input_row_offsets: Tensor,
        mla_prefill_metadata: MLAPrefillMetadata | None,
        prev_topk_indices: Tensor | None,
        reuse_prev_topk: bool,
        indexer_layer_idx: Tensor | None,
        *,
        pcp: PrefillContextParallel,
    ) -> tuple[Tensor, Tensor]:
        """Attends a prefill step split over each DCP group's ranks.

        Each rank projects its run of the batch's rows (``pcp.local_rows``),
        the group all-gathers the chunk's raw latent rows, every rank stores
        the positions it owns into its DCP cache and every position into the
        whole-context scratch, and the non-DCP sparse prefill op attends the
        rank's queries over the scratch with all of the group's heads. One
        all-to-all leaves every rank its tensor-parallel heads of every row,
        as the DCP merge does, for the head-sharded ``o_proj``. See
        :mod:`.prefill_context_parallel`.
        """
        if self.quantized or not self.dcp_q_replicate:
            raise NotImplementedError(
                "prefill context parallelism needs BF16 MLA weights and a "
                "replicated decode context parallel query"
            )
        q_lora_rank = self.q_lora_rank
        assert q_lora_rank is not None
        x_local = pcp.local_rows(x)
        qkv = quant_ops.matmul(x_local, self.wqkv)
        q_a_out, kv = qkv.split([q_lora_rank, self.cache_head_dim], axis=1)
        q_a_normed = self.q_a_layernorm(q_a_out)
        # ``q_b`` holds the rank's group's heads, replicated within the group
        # on the DCP view; each rank projects its own rows.
        assert isinstance(self.q_b_proj, Tensor)
        q_b = to_dcp_mesh(self.q_b_proj, x_local.mesh)
        # Per device: the matmul rule cannot place rows sharded along DCP.
        xq = F.call_on_mesh(
            lambda a, w: a @ w.T, x_local.mesh, out_specs=q_a_normed.mapping
        )(q_a_normed, q_b)
        n_heads = dcp_group_heads(xq.mesh, self.n_heads, self.dcp_size)
        if not _sparse_prefill_head_count_supported(
            n_heads, self.kv_params.dtype
        ):
            raise NotImplementedError(
                f"prefill context parallelism attends a DCP group's {n_heads} "
                "heads with the sparse prefill kernel, which takes "
                f"{_SPARSE_PREFILL_SUPPORTED_HEADS + _SPARSE_PREFILL_TP_SHARDED_HEADS}"
            )
        xq = xq.reshape((xq.shape[0], n_heads, self.qk_head_dim))
        freqs_cis = F.cast(freqs_cis, xq.dtype)

        # The whole chunk's raw latent rows, normed, roped and stored: by
        # ownership into the DCP cache, whole into the scratch.
        kv_all = pcp.gather_rows(kv, x)
        store_mla_latent_ragged(
            kv_all,
            pcp.batch_row_offsets,
            freqs_cis,
            self.kv_a_proj_layernorm,
            self.kv_params,
            kv_collection,
            layer_idx,
            _LATENT_NORM_EPS,
            dcp_size=self.dcp_size,
            dcp_interleave=self.dcp_interleave,
            dcp_rank=dcp_ranks(x.mesh, self.dcp_size),
        )
        scratch_slot = F.constant(0, DType.uint32, device=CPU())
        store_mla_latent_ragged(
            kv_all,
            pcp.batch_row_offsets,
            freqs_cis,
            self.kv_a_proj_layernorm,
            self.kv_params,
            pcp.mla_scratch,
            scratch_slot,
            _LATENT_NORM_EPS,
        )
        full_layer = self.indexer is not None and not reuse_prev_topk
        indexer_slot = (
            layer_idx if indexer_layer_idx is None else indexer_layer_idx
        )
        pcp.fill_prefix(
            [(kv_collection, pcp.mla_scratch, layer_idx)]
            + (
                [(indexer_kv_collection, pcp.indexer_scratch, indexer_slot)]
                if full_layer
                else []
            )
        )
        if full_layer:
            assert self.indexer is not None
            topk_indices = self.indexer.forward_pcp(
                x_local,
                q_a_normed,
                freqs_cis,
                indexer_kv_collection,
                indexer_slot,
                pcp,
                x,
            )
        else:
            if prev_topk_indices is None:
                raise ValueError(
                    "Shared (skip_topk) sparse attention layers require top-k "
                    "indices from a previous full indexer layer."
                )
            topk_indices = prev_topk_indices

        scratch = pcp.mla_view
        # The plain sparse op reads its group's heads' weights whole: on the
        # DCP view they are the group's shard of each.
        w_uk, _ = self.w_uk
        w_uv, _ = self.w_uv
        w_uk = to_dcp_mesh(w_uk, xq.mesh)
        # The sparse prefill reads only ``w_uk`` and ``w_uv``; the op still
        # types the dense prefill's ``[H * Dn, R]`` ``w_k``. A view of
        # ``w_uk`` fills it without a second resident copy of the heads.
        w_k = w_uk.reshape((-1, self.kv_lora_rank))
        if pcp.zero_prefill_offsets is not None:
            # The plan's, built once: the sparse op never reads them.
            buffer_row_offsets = cache_offsets = pcp.zero_prefill_offsets
        else:
            metadata = self.create_mla_prefill_metadata(
                pcp.row_offsets, scratch
            )
            buffer_row_offsets = metadata.buffer_row_offsets
            cache_offsets = metadata.cache_offsets
        assert scratch.attention_dispatch_metadata is not None
        assert scratch.mla_num_partitions is not None
        out = mla_prefill_decode_graph(
            q=xq,
            kv=kv,
            input_row_offsets=pcp.row_offsets,
            freqs_cis=freqs_cis,
            kv_norm_gamma=self.kv_a_proj_layernorm,
            buffer_row_offsets=buffer_row_offsets,
            cache_offsets=cache_offsets,
            # Unused by the sparse op; a host zero avoids a device read.
            buffer_length=F.constant([0], DType.int32, device=CPU()),
            w_k=w_k,
            w_uk=w_uk,
            w_uv=to_dcp_mesh(w_uv, xq.mesh),
            kv_params=self.kv_params,
            kv_collection=scratch,
            layer_idx=scratch_slot,
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=self.scale,
            epsilon=_LATENT_NORM_EPS,
            v_head_dim=self.v_head_dim,
            scalar_args=scratch.attention_dispatch_metadata,
            num_partitions_scalar=scratch.mla_num_partitions,
            sparse_indices=topk_indices,
            sparse_indices_stride=sparse_selection_width(
                self.kv_params, self.index_topk
            ),
            physical_indices=True,
            sparse_attn_sink_inert=True,
        )
        del input_row_offsets, mla_prefill_metadata
        merged = pcp.to_head_sharded(out, x)
        attn_out = merged.reshape(
            (merged.shape[0], merged.shape[1] * self.v_head_dim)
        )
        return attn_out, topk_indices


Selection = Tensor | tuple[Tensor, Tensor]
"""A layer's top-k selection: one tensor, or under predicated prefill
context parallelism its DCP rows and each rank's own PCP rows, which the
next layer reads without a join."""


def _columns(t: Tensor, n: int) -> Tensor:
    """Each device's ``t`` cut or ``-1``-padded to its first ``n`` columns."""

    def cut(v: Tensor) -> Tensor:
        have = int(v.shape[1])
        if have == n:
            return v
        if have > n:
            return v[:, :n]
        fill = F.full(
            [v.shape[0], n - have], -1, dtype=v.dtype, device=v.device
        )
        return F.concat([v, fill], axis=1)

    return F.call_on_mesh(cut, t.mesh, out_specs=t.mapping)(t)


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
    if layer.dcp_size > 1 and layer.dcp_q_replicate:
        # Decode context parallelism with a replicated query: every rank of a
        # DCP group computes all of the group's queries and absorbs them
        # itself. The Q and KV up-projections shard by head across the
        # groups and replicate within each, along the ``tp`` axis of the
        # mesh's DCP view, so with one group spanning TP they stay whole.
        # ``o_proj`` is head-sharded over every rank, and each rank applies
        # only its own heads' rows of ``w_vc`` after the merge: a group's
        # merge leaves each rank its tensor-parallel heads.
        mesh = layer.o_proj.weight.mesh
        view = dcp_view(mesh, layer.dcp_size)
        layer.q_b_proj = layer.q_b_proj.to(NamedMapping(view, (TP, None)))
        layer.kv_b_proj = layer.kv_b_proj.to(NamedMapping(view, (TP, None)))
        layer.o_proj.weight = layer.o_proj.weight.to(
            NamedMapping(mesh, (None, TP))
        )
    else:
        if layer.dcp_size > 1:
            # With a gathered query every weight shards as under tensor
            # parallelism; each DCP group all-gathers its absorbed query
            # heads.
            dcp_view(layer.o_proj.weight.mesh, layer.dcp_size)
        tensor_parallel_latent_attention_with_rope(layer)

    if num_devices > 1 and layer._ragged_scales:
        # The per-head scale gather indexes ``kv_b_proj``'s scale by global
        # head, and per-head rows straddle its blocks, so a row shard of the
        # scale is not a head shard. Keep the (few-dozen-row) scale whole on
        # every device of its weight's mesh; the gathered per-head scales
        # are then sharded with their weights.
        kv_b_proj = layer.kv_b_proj
        assert isinstance(kv_b_proj, FP8BlockTensor)
        kv_b_proj.weight_scale_inv = kv_b_proj.weight_scale_inv.to(
            DeviceMapping.replicated(kv_b_proj.weight.mesh)
        )

    return layer
