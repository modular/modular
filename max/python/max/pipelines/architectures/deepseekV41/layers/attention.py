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
"""Compressed sparse attention for DeepSeek-V4.1-Flash decoder layers.

Each layer attends over a sliding window of recent tokens plus a top-k
selection of compressed KV entries chosen by an indexer. Layers can share
work: a layer may read another layer's compressed KV and index keys, or
reuse the top-k selection an earlier layer computed in the same forward.

The window and compressed caches hold quantized records (MXFP8 per token,
NVFP4 per compression group) that the store ops rotate and quantize; the
attention names each record by its absolute slot in its cache leaf.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from max.driver import accelerator_architecture_name
from max.dtype import DType
from max.graph import (
    DeviceRef,
    Dim,
    ShardingStrategy,
    TensorValue,
    Weight,
    ops,
)
from max.nn.kernels import (
    block_diagonal_block_scaled_matmul,
    quantize_dynamic_block_scaled,
    rope_quantize_dynamic_block_scaled,
)
from max.nn.kv_cache import PagedCacheValues
from max.nn.layer import Module
from max.nn.linear import Linear
from max.nn.norm.rms_norm import RMSNorm
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from max.nn.quant_ops import _matmul_float8_mxfp8

from ...deepseekV4.layers.rope import apply_rope_tail
from ..layer_resolver import LayerInfo
from ..model_config import (
    DeepseekV41Config,
    compressor_bf16_enabled,
    index_list_fused_enabled,
    q_mxfp8_enabled,
    woa_mxfp8_enabled,
)
from .compressor import DeepseekV41Compressor, ring_save_slots, ring_slots
from .csa2 import (
    HEAD_DIM,
    MAX_COMPRESSED,
    PREFILL_ROWS,
    WINDOW_ENTRIES,
    csa2_attention,
    index_list,
    index_list_ops,
    leaf_slots,
    store_compressed_kv,
    store_window_kv,
)
from .indexer import (
    DeepseekV41Indexer,
    candidate_keep_blocks,
    index_key_slots,
    key_positions,
    quantize_queries,
    rope_cos_sin_table,
    select_index_keys,
    select_index_keys_block_topk,
    select_index_keys_masked,
    store_index_keys,
)
from .mxfp8_matmul import mxfp8_linear
from .query import DeepseekV41Query
from .ragged import V41RaggedRows as RaggedRows
from .reuse import ForwardSelection, bind_cache_sources, selection_for_layer
from .rope import rope_for_layer

# MXFP8 block size: one E8M0 scale per 32 elements.
_MXFP8_BLOCK_SIZE = 32
# A forward whose requests each bring at most this many query tokens runs
# the decode attention, which reads the records in place; a longer one
# dequantizes the records it names once, first.
_DECODE_MAX_QUERY_TOKENS = 8


def prefill_rows_bound(
    tokens: TensorValue,
    batch: TensorValue,
    longest: TensorValue,
    cached: TensorValue,
    ratio: int | None,
) -> TensorValue:
    """Returns the most distinct slots a forward's CSA2 lists can name.

    ``T + 128 * B`` window slots (each request's tokens and the 127 before
    them), plus each request's compressed slots: at most 512 a token and
    every group of its context. The longest request's ``L`` tokens name at
    most ``min(512 * L, G)`` of them and the other ``T - L`` tokens at most
    ``min(512 * (T - L), (B - 1) * G)``, with ``G`` the groups of the
    longest context.

    Args:
        tokens: Host int64 scalar ``T``, the forward's query tokens.
        batch: Host int64 scalar ``B``, its requests.
        longest: Host int64 scalar ``L``, the most query tokens a request
            brings.
        cached: Host int64 scalar, the most tokens a request has cached.
        ratio: The layer's compression ratio, ``None`` without a
            compressed cache.
    """
    named = tokens + batch * WINDOW_ENTRIES
    if ratio is None:
        return named
    groups = ops.floor_div(longest + cached + (ratio - 1), ratio)
    return (
        named
        + ops.min(longest * MAX_COMPRESSED, groups)
        + ops.min((tokens - longest) * MAX_COMPRESSED, (batch - 1) * groups)
    )


def _output_projection_quant_config() -> QuantConfig:
    """Returns the MXFP8 ``QuantConfig`` for the attention projections.

    ``block_size=(1, 32)`` declares ``weight_scale`` as
    ``[out_dim, in_dim / 32]``: the checkpoint's 32x32 block scale expanded
    to one scale row per output row.
    """
    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.DYNAMIC,
            dtype=DType.float32,
            block_size=(1, _MXFP8_BLOCK_SIZE),
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float8_e8m0fnu,
            block_size=(1, _MXFP8_BLOCK_SIZE),
        ),
        mlp_quantized_layers=set(),
        attn_quantized_layers=set(),
        format=QuantFormat.MXFP8,
    )


@dataclass(frozen=True)
class CSA2Cache:
    """Cache handles shared by every :class:`CSA2Attention` layer.

    The per-layer maps are keyed by the owning (KV-source) layer id, not the
    consuming layer's id.
    """

    swa_collection: PagedCacheValues
    """Sliding-window record leaf shared by every layer, indexed by
    ``layer_id``: an MXFP8 record per token."""
    main_kv_by_layer: Mapping[int, PagedCacheValues]
    """Owning layer id -> compressed record leaf: an NVFP4 record per
    compression group."""
    index_k_by_layer: Mapping[int, PagedCacheValues]
    """Owning layer id -> indexer key cache.

    Must be a different object from the matching ``main_kv_by_layer`` cache,
    since "Full" layers write to both.
    """
    comp_layer_index_by_layer_id: Mapping[int, int]
    """Owning layer id -> its slot index in the main KV and index key caches."""
    state_by_layer: Mapping[int, PagedCacheValues] = field(default_factory=dict)
    """Owning layer id -> pair-carry ring (``c2a_state``) for ratio two.

    A request's scratch block of raw float32 ``wkv`` row then ``wgate`` row
    by ``position % ring_rows``, holding its last forward's rows, so a
    forward starting at an odd position can pool against row ``start - 1``.
    Required by every ratio-2 KV-source layer.
    """


@dataclass
class _ForwardSelectionBox:
    """Holds the current forward's ``ForwardSelection`` while tracing.

    Kept out of ``CSA2Cache`` because the selection is not persistent state.
    Reset ``.current = None`` before retracing the same layers for a new
    forward, or the last layer's selection leaks into the next trace.
    """

    current: ForwardSelection[TensorValue, CandidateBlocks] | None = None


def wo_a_linear(
    group_width: int, out_dim: int, device: DeviceRef, *, mxfp8: bool
) -> Linear:
    """Declares ``attn.wo_a`` with a ``[out_dim, group_width]`` weight.

    With ``mxfp8`` the weight is ``float8_e4m3fn`` plus a
    ``[out_dim, group_width / 32]`` ``float8_e8m0fnu`` ``weight_scale``,
    tile-interleaved at load (see :func:`mxfp8_grouped_projection`);
    otherwise it is bf16 and must be dequantized on the host at load.
    """
    if mxfp8:
        return Linear(
            in_dim=group_width,
            out_dim=out_dim,
            dtype=DType.float8_e4m3fn,
            device=device,
            quant_config=_output_projection_quant_config(),
        )
    return Linear(group_width, out_dim, DType.bfloat16, device)


def bf16_grouped_projection(
    x: TensorValue, weight: TensorValue, groups: int
) -> TensorValue:
    """Block-diagonal bf16 projection; see :func:`mxfp8_grouped_projection`.

    The group axis is the only batched dim (``[g, t, k] @ [g, k, r]``);
    broadcasting ``weight`` over tokens would materialize a per-layer
    constant large enough to exhaust device memory.

    Args:
        x: ``[t, groups * k]``.
        weight: ``[groups * r, k]``.
        groups: The number of (local) groups.

    Returns:
        ``[t, groups * r]``.
    """
    t = x.shape[0]
    k = int(weight.shape[1])
    r = int(weight.shape[0]) // groups
    grouped = ops.transpose(ops.reshape(x, [t, groups, k]), 0, 1)
    projected = ops.matmul(
        grouped,
        ops.transpose(ops.reshape(weight, [groups, r, k]), -1, -2),
    )
    return ops.reshape(ops.transpose(projected, 0, 1), [t, groups * r])


# Rows per tile of the interleaved block-scale layout.
_SCALE_TILE_ROWS = 128


def _grouped_mxfp8_supported() -> bool:
    """Whether the build target runs the block-diagonal MXFP8 matmul (SM100)."""
    try:
        return accelerator_architecture_name().startswith("sm_10")
    except Exception:
        return False


def _deinterleave_scales(scale: TensorValue) -> TensorValue:
    """Undoes the 128 x 4 tile interleave of ``[rows, cols]`` block scales."""
    rows, cols = int(scale.shape[0]), int(scale.shape[1])
    tiled = ops.reshape(
        scale,
        [rows // _SCALE_TILE_ROWS, cols // 4, 32, _SCALE_TILE_ROWS // 32, 4],
    )
    return ops.reshape(ops.permute(tiled, [0, 3, 2, 1, 4]), [rows, cols])


def _mxfp8_projection_per_group(
    x: TensorValue,
    weight: TensorValue,
    weight_scale: TensorValue,
    groups: int,
) -> TensorValue:
    """:func:`mxfp8_grouped_projection` as one dense MXFP8 matmul per group.

    Takes row-major ``[groups * r, k / 32]`` scales.
    """
    k = int(weight.shape[1])
    r = int(weight.shape[0]) // groups
    x = ops.cast(x, DType.bfloat16)
    outs = [
        _matmul_float8_mxfp8(
            x[:, g * k : (g + 1) * k],
            weight[g * r : (g + 1) * r, :],
            weight_scale[g * r : (g + 1) * r, :],
        )
        for g in range(groups)
    ]
    return outs[0] if groups == 1 else ops.concat(outs, axis=1)


def undo_rope_tail(
    x: TensorValue, freqs_cis: TensorValue, head_dim: int
) -> TensorValue:
    """Rotates each head's trailing dims back by the conjugate RoPE.

    Args:
        x: ``[t, heads * head_dim]``.
        freqs_cis: ``[t, rope_dim // 2, 2]`` cos/sin at each row's position.
        head_dim: Columns per head.

    Returns:
        ``x`` with each head's last ``rope_dim`` dims rotated back.
    """
    t, heads = x.shape[0], int(x.shape[1]) // head_dim
    return ops.reshape(
        apply_rope_tail(
            ops.reshape(x, [1, t, heads, head_dim]),
            freqs_cis,
            int(freqs_cis.shape[-2]) * 2,
            inverse=True,
        ),
        [t, heads * head_dim],
    )


def mxfp8_grouped_projection(
    x: TensorValue,
    weight: TensorValue,
    weight_scale: TensorValue,
    groups: int,
    rope: tuple[TensorValue, int] | None = None,
) -> TensorValue:
    """Block-diagonal MXFP8 projection (the grouped einsum ``bgk,grk->bgr``).

    Group ``g``'s ``k`` input columns multiply ``weight`` rows
    ``[g * r, (g + 1) * r)``, with ``r = weight.shape[0] // groups``. On
    SM100 the activations are quantized once, token-major, and one
    block-diagonal block-scaled matmul reads each group's columns and
    writes each group's output columns in place. Elsewhere each group takes
    a dense MXFP8 matmul.

    Args:
        x: ``[t, groups * k]`` activations; cast to bf16 for the quantizer.
        weight: ``[groups * r, k]`` ``float8_e4m3fn``.
        weight_scale: ``[groups * r, k / 32]`` ``float8_e8m0fnu``, the
            checkpoint's 32x32 block scale row-expanded, then
            tile-interleaved: row ``n``, column ``c`` is stored at
            ``[n // 128, c // 4, n % 32, (n % 128) // 32, c % 4]`` of
            ``[groups * r / 128, k / 128, 32, 4, 4]``. ``r`` and ``k`` must
            be multiples of 128.
        groups: The number of (local) groups.
        rope: Optional ``(freqs_cis, head_dim)``. Each ``head_dim`` head of
            ``x`` first has its trailing dims rotated back by the conjugate
            of its row of ``freqs_cis`` (``[t, rope_dim // 2, 2]``). On
            SM100 the quantizer applies the rotation.

    Returns:
        ``[t, groups * r]`` bf16.
    """
    # ``Linear`` declares its weight scale on the host; move it beside ``x``.
    weight_scale = weight_scale.to(x.device)
    if not _grouped_mxfp8_supported():
        if rope is not None:
            x = undo_rope_tail(x, *rope)
        return _mxfp8_projection_per_group(
            x, weight, _deinterleave_scales(weight_scale), groups
        )
    x = ops.cast(x, DType.bfloat16)
    if rope is None:
        x_q, x_scales = quantize_dynamic_block_scaled(
            x,
            sf_vector_size=32,
            scales_type=DType.float8_e8m0fnu,
            out_type=DType.float8_e4m3fn,
        )
    else:
        freqs_cis, head_dim = rope
        x_q, x_scales = rope_quantize_dynamic_block_scaled(
            x,
            freqs_cis,
            head_dim,
            inverse=True,
            sf_vector_size=32,
            scales_type=DType.float8_e8m0fnu,
            out_type=DType.float8_e4m3fn,
        )
    rows, k = (int(d) for d in weight.shape)
    return block_diagonal_block_scaled_matmul(
        x_q,
        weight,
        x_scales,
        ops.reshape(
            weight_scale, [rows // _SCALE_TILE_ROWS, k // 128, 32, 4, 4]
        ),
        groups,
    )


@dataclass(frozen=True)
class CompressedKV:
    """:meth:`CSA2Attention.compress` output, one row per raw token."""

    latent: TensorValue
    """``[T, head_dim]`` bf16 pooled, normalized latent, before RoPE.

    Zero on rows that do not close a group. Input to the compressed record
    and the index keys, whose stores apply RoPE.
    """
    valid: TensorValue
    """``[T]`` bool, whether the row closes a group (publishes an entry)."""


@dataclass
class PendingCSA2Attention:
    """State passed from :meth:`CSA2Attention.begin` to ``end``.

    ``index_q`` and ``index_leaf`` are set only on "Full" and "Reindex"
    layers.
    """

    q: TensorValue
    out_shape: list[Dim]
    freqs_cis: TensorValue
    swa_collection: PagedCacheValues
    comp_collection: PagedCacheValues | None
    comp_index: int | None
    """This layer's index among the compressed leaf's layers."""
    layer_comp: TensorValue | None
    """``comp_index`` as a host uint32 scalar, for the indexer ops."""
    index_q: tuple[TensorValue, TensorValue, TensorValue] | None = None
    """MXFP4 index queries, scale words and weights ("Full"/"Reindex")."""
    index_leaf: PagedCacheValues | None = None
    """The index-key leaf the selection reads ("Full"/"Reindex")."""


@dataclass(frozen=True)
class CandidateBlocks:
    """Blocks of KV positions kept by a candidate-source layer.

    Later candidate-consumer layers restrict their scores to these blocks
    before their own per-token top-k.
    """

    kept: TensorValue | None
    """``[T, candidate_topk_blocks]`` int32 kept block ids, ``-1``-padded.

    ``None`` when block selection is off or the score width holds at most
    ``candidate_topk_blocks`` blocks, so no block can be dropped.
    """
    compress_ratio: int
    """The source layer's compression ratio, which consumers must share.

    Block ``b`` spans ``block_size * ratio`` raw columns.
    """


class CSA2Attention(Module):
    """One decoder layer's compressed sparse attention.

    Behavior depends on ``layer_info.attention_mode``:

    - "SWA-only": sliding-window attention only.
    - "Full": compresses this layer's KV (ratio two pools adjacent token
      pairs, carrying an unfinished pair across forwards), stores the
      compressed and index-key rows, then scores its indexer against the
      stored index-key history.
    - "Reindex": scores its own indexer against its KV source layer's index
      keys.
    - "Reuse": reuses the selection its index source layer produced in this
      forward.

    Compressed caches hold one row per raw token; a group's entry sits on
    the row closing it and other rows are masked out. Each request's
    index-key history is gathered into a ``[max_seq_len]`` position-indexed
    table, so scoring is ``O(max_seq_len)`` per query row; ``max_seq_len``
    here is the model's position capacity, which includes the speculative
    slack. A candidate-source layer also keeps its best-scoring blocks of
    positions, and later candidate-consumer layers restrict their scores to
    those blocks.
    Selections pass between layers as graph values, so a "Reuse" layer
    depends on its source in the graph.

    Built per layer by :func:`make_csa2_attn_factory`; called as
    ``attn(h, rows, cache)`` with a :class:`CSA2Cache`.
    """

    def __init__(
        self,
        layer_info: LayerInfo,
        device: DeviceRef,
        selection_box: _ForwardSelectionBox,
        config: DeepseekV41Config,
        *,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        window: int,
        num_comp: int,
        max_seq_len: int,
        trace: (
            list[
                tuple[
                    int, ForwardSelection[TensorValue, CandidateBlocks] | None
                ]
            ]
            | None
        ) = None,
    ) -> None:
        super().__init__()
        self.layer_info = layer_info
        self.selection_box = selection_box
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.window = window
        if window != WINDOW_ENTRIES or head_dim != HEAD_DIM:
            raise ValueError(
                f"CSA2 attends {WINDOW_ENTRIES}-token windows of "
                f"{HEAD_DIM}-wide records, got window={window}, "
                f"head_dim={head_dim}"
            )
        if num_comp > MAX_COMPRESSED:
            raise ValueError(
                f"CSA2 attends at most {MAX_COMPRESSED} compressed entries, "
                f"got num_comp={num_comp}"
            )
        self.num_comp = num_comp
        self.scale = head_dim**-0.5
        self.device = device
        self._selection_trace = trace
        self.candidate_topk_blocks = config.candidate_topk_blocks
        self.candidate_block_size = config.candidate_block_size
        self.candidate_blocks = candidate_keep_blocks(
            config.max_seq_len,
            max_seq_len,
            layer_info.compress_ratio,
            block_size=config.candidate_block_size,
            topk_blocks=config.candidate_topk_blocks,
        )
        # A consumer row keeps its newest block, which may hold one key, and
        # ``candidate_topk_blocks - 1`` full blocks; the masked selection
        # needs at least ``index_topk`` of them.
        if (
            config.candidate_topk_blocks > 0
            and config.candidate_block_size > 0
            and config.index_topk is not None
            and (config.candidate_topk_blocks - 1) * config.candidate_block_size
            + 1
            < config.index_topk
        ):
            raise ValueError(
                f"candidate_topk_blocks={config.candidate_topk_blocks} blocks "
                f"of candidate_block_size={config.candidate_block_size} keys "
                f"can keep fewer than index_topk={config.index_topk} keys"
            )
        self.rope = rope_for_layer(
            config, layer_info.layer_id, max_seq_len, device
        )
        # MXFP8, or bf16 host-dequantized weights with `MAX_DSV41_Q_MXFP8=0`.
        q_quant_config = (
            _output_projection_quant_config() if q_mxfp8_enabled() else None
        )
        self.query_proj = DeepseekV41Query(
            hidden_size,
            config.q_lora_rank,
            head_dim,
            num_heads,
            config.rms_norm_eps,
            DType.bfloat16,
            device,
            quant_config=q_quant_config,
        )
        # Top-level so its checkpoint path is `attn.kv_norm.weight`.
        self.kv_norm = RMSNorm(head_dim, DType.float32, config.rms_norm_eps)
        # "Full" and "Reindex" both publish a selection; only "Full" produces
        # its own index keys (`owns_k`).
        self.indexer = (
            DeepseekV41Indexer(
                hidden_size,
                config.q_lora_rank,
                head_dim,
                config.index_n_heads,
                config.index_head_dim,
                config.qk_rope_head_dim,
                config.rms_norm_eps,
                DType.bfloat16,
                device,
                owns_k=layer_info.attention_mode == "Full",
                wq_b_quant_config=q_quant_config,
            )
            if layer_info.has_indexer
            else None
        )
        # Only KV-source layers have compressor weights; Reindex/Reuse layers
        # read their source's caches.
        is_kv_source = (
            layer_info.compress_ratio > 0
            and layer_info.kv_source_layer_id == layer_info.layer_id
        )
        self.compressor = (
            DeepseekV41Compressor(
                layer_info.compress_ratio,
                config.rms_norm_eps,
                hidden_size=hidden_size,
                device=device,
                head_dim=head_dim,
                weight_dtype=(
                    DType.bfloat16
                    if compressor_bf16_enabled()
                    else DType.float32
                ),
            )
            if is_kv_source
            else None
        )
        self.attn_sink = Weight(
            name="attn_sink",
            dtype=DType.float32,
            shape=(num_heads,),
            device=device,
        )
        self.o_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        # `wo_a`'s weight is consumed raw by a grouped matmul, never called
        # as a Linear: MXFP8 by default, or host-dequantized bf16 with
        # `MAX_DSV41_WOA_MXFP8=0`.
        self.woa_mxfp8 = woa_mxfp8_enabled()
        self.wo_a = wo_a_linear(
            num_heads * head_dim // config.o_groups,
            config.o_groups * config.o_lora_rank,
            device,
            mxfp8=self.woa_mxfp8,
        )
        # MXFP8 weight with a dynamically quantized activation.
        self.wo_b = Linear(
            in_dim=config.o_groups * config.o_lora_rank,
            out_dim=hidden_size,
            dtype=DType.float8_e4m3fn,
            device=device,
            quant_config=_output_projection_quant_config(),
        )

    def shard_heads(self, num_devices: int) -> None:
        """Declares a head-parallel split of this layer's weights.

        ``wq_b`` and ``attn_sink`` split per head, ``wo_a`` along its groups
        and ``wo_b`` along its input, so each device's output is a partial sum
        that must be all-reduced. The shared latent side stays whole.

        Args:
            num_devices: Number of tensor-parallel devices.

        Raises:
            ValueError: If heads or groups do not divide evenly.
        """
        if self.num_heads % num_devices or self.o_groups % num_devices:
            raise ValueError(
                f"{self.num_heads} heads in {self.o_groups} groups do not "
                f"split across {num_devices} devices"
            )
        self.query_proj.wq_b.sharding_strategy = ShardingStrategy.rowwise(
            num_devices
        )
        self.attn_sink.sharding_strategy = ShardingStrategy.axiswise(
            0, num_devices
        )
        self.wo_a.sharding_strategy = ShardingStrategy.rowwise(num_devices)
        self.wo_b.sharding_strategy = ShardingStrategy.columnwise(num_devices)
        if self.indexer is not None:
            self.indexer.shard_heads(num_devices)

    def keep_local_heads(self, num_devices: int) -> None:
        """Keeps only this device's share of the :meth:`shard_heads` split."""
        self.num_heads //= num_devices
        self.o_groups //= num_devices
        self.query_proj.local_num_heads //= num_devices

    def _output_projection(
        self, out: TensorValue, freqs_cis: TensorValue
    ) -> TensorValue:
        """Projects ``[t, num_heads, head_dim]`` to ``[t, hidden_size]``.

        Undoes RoPE on each head's trailing dims at the query position, then
        applies the grouped low-rank ``wo_a`` and ``wo_b``.

        Args:
            out: ``[t, num_heads, head_dim]`` attention output.
            freqs_cis: The queries' gathered cos/sin table.

        Returns:
            ``[t, hidden_size]``.
        """
        out = ops.reshape(out, [out.shape[0], self.num_heads * self.head_dim])
        if self.woa_mxfp8:
            assert self.wo_a.weight_scale is not None
            return mxfp8_linear(
                self.wo_b,
                mxfp8_grouped_projection(
                    out,
                    self.wo_a.weight,
                    self.wo_a.weight_scale,
                    self.o_groups,
                    rope=(freqs_cis, self.head_dim),
                ),
            )
        return mxfp8_linear(
            self.wo_b,
            bf16_grouped_projection(
                undo_rope_tail(out, freqs_cis, self.head_dim),
                self.wo_a.weight,
                self.o_groups,
            ),
        )

    def __call__(
        self,
        h: TensorValue,
        rows: RaggedRows,
        cache: object | None,
        *,
        swa_valid_from: TensorValue | None = None,
    ) -> TensorValue:
        return self.end(
            self.begin(h, rows, cache), rows, swa_valid_from=swa_valid_from
        )

    def begin(
        self, h: TensorValue, rows: RaggedRows, cache: object | None
    ) -> PendingCSA2Attention:
        """Runs :meth:`__call__` up to the indexer's top-k."""
        if not isinstance(cache, CSA2Cache):
            raise TypeError(
                f"CSA2Attention requires a CSA2Cache, got {type(cache)!r}"
            )
        layer = self.layer_info
        b, s, hidden = h.shape[0], h.shape[1], h.shape[2]
        # DeepseekV41Query expects a leading batch-of-1 axis.
        h_flat = ops.reshape(h, [1, -1, hidden])
        freqs_cis = ops.gather(
            self.rope.freqs_cis_base(), rows.positions, axis=0
        )
        qr_scales = None
        indexer_mxfp8 = (
            self.indexer is None or self.indexer.wq_b.quant_config is not None
        )
        if self.query_proj.can_fuse_norm_quant() and indexer_mxfp8:
            qr, qr_scales, q, kv = self.query_proj.call_fused(
                h_flat, freqs_cis, self.kv_norm
            )
        else:
            qr, q, kv_raw = self.query_proj(h_flat, freqs_cis)
            kv = ops.cast(self.kv_norm(kv_raw), DType.bfloat16)
        q = ops.reshape(q, [-1, self.num_heads, self.head_dim])
        state = PendingCSA2Attention(
            q=q,
            out_shape=[b, s, hidden],
            freqs_cis=freqs_cis,
            swa_collection=cache.swa_collection,
            comp_collection=None,
            comp_index=None,
            layer_comp=None,
        )
        rope_table = rope_cos_sin_table(self.rope.freqs_cis_base())
        # The window record: full-width RMSNorm of the KV latent; the store
        # applies RoPE to its trailing dims and quantizes it to MXFP8.
        # Every layer, including "SWA-only", owns its sliding-window slot.
        swa = cache.swa_collection
        store_window_kv(
            swa.kv_blocks,
            ops.reshape(kv, [-1, self.head_dim]),
            rows.positions,
            rope_table,
            leaf_slots(swa, rows.bid, rows.positions, layer.layer_id),
            block_stride=swa.page_stride,
        )
        if layer.attention_mode == "SWA-only":
            return state

        binding = bind_cache_sources(
            layer, cache.main_kv_by_layer, cache.index_k_by_layer
        )
        assert binding is not None
        state.comp_collection = binding.main_kv_leaf
        comp_index = cache.comp_layer_index_by_layer_id[
            binding.kv_source_layer_id
        ]
        layer_comp = ops.constant(comp_index, DType.uint32, DeviceRef.CPU())
        state.comp_index = comp_index
        state.layer_comp = layer_comp

        if layer.attention_mode in ("Full", "Reindex"):
            indexer = self.indexer
            assert indexer is not None
            index_queries, index_weights = indexer(qr, h_flat, qr_scales)
            state.index_q = quantize_queries(
                index_queries,
                index_weights,
                rows.positions,
                rope_table,
                n_heads=indexer.n_heads,
                weight_scale=indexer.softmax_scale,
            )
            state.index_leaf = binding.index_k_leaf
            if indexer.owns_k:
                # A "Full" layer is its own KV source: the pre-RoPE pooled
                # latent feeds both the compressed record and the index key,
                # whose stores rotate it.
                assert self.compressor is not None
                raw = ops.reshape(
                    self.compressor.project(h_flat),
                    [-1, self.head_dim * layer.compress_ratio],
                )
                state_leaf = None
                state_layer = None
                if layer.compress_ratio == 2:
                    state_leaf = self._state_leaf(
                        cache, binding.kv_source_layer_id
                    )
                    state_layer = layer_comp
                compressed = self.compress(raw, rows, state_leaf, state_layer)
                # Group `p // ratio` closes on row `p`; other rows skip.
                comp = binding.main_kv_leaf
                group = ops.floor_div(
                    ops.cast(rows.positions, DType.int32), layer.compress_ratio
                )
                store_compressed_kv(
                    comp.kv_blocks,
                    compressed.latent,
                    rows.positions,
                    rope_table,
                    ops.where(
                        compressed.valid,
                        leaf_slots(comp, rows.bid, group, comp_index),
                        ops.constant(-1, DType.int32, group.device),
                    ),
                    ratio=layer.compress_ratio,
                    block_stride=comp.page_stride,
                )
                # Stored before selection so this forward's keys are in the
                # history read below and by dependent Reindex layers.
                store_index_keys(
                    binding.index_k_leaf,
                    ops.reshape(
                        indexer.wk(compressed.latent), [-1, indexer.head_dim]
                    ),
                    rows.positions,
                    rope_table,
                    indexer.k_norm.weight,
                    index_key_slots(
                        binding.index_k_leaf,
                        rows,
                        ratio=layer.compress_ratio,
                    ),
                    layer_comp,
                    ratio=layer.compress_ratio,
                    epsilon=indexer.rms_norm_eps,
                )
        return state

    def compress(
        self,
        raw: TensorValue,
        rows: RaggedRows,
        state: PagedCacheValues | None = None,
        layer_idx: TensorValue | None = None,
    ) -> CompressedKV:
        """Pools ``raw`` into one latent per closed compression group.

        Args:
            raw: ``[T, head_dim * ratio]`` float32, this layer's
                ``compressor.project`` output.
            rows: The forward's ragged rows.
            state: Ratio two only: the layer's ``c2a_state`` ring. The last
                rows of ``raw`` are saved there, and a request starting at an
                odd position pools against the row saved before it.
            layer_idx: Ratio two only: host uint32 index of the layer in
                ``state``.

        Returns:
            The pooled latent and whether its row closes a group.
        """
        assert self.compressor is not None
        positions = rows.positions
        slots = None if state is None else ring_slots(state, rows)
        save_slots = (
            None
            if state is None or slots is None
            else ring_save_slots(state, rows, slots)
        )
        latent = self.compressor(
            raw,
            positions,
            rows.offsets,
            state,
            slots,
            save_slots,
            layer_idx,
        )
        # Group ``p // ratio`` closes on the row where ``(p + 1) % ratio`` is
        # zero; only that row publishes an entry.
        ratio = self.layer_info.compress_ratio
        valid = ops.equal(
            ops.cast(positions + 1, DType.int64) % ratio,
            ops.constant(0, DType.int64, positions.device),
        )
        return CompressedKV(
            latent=ops.cast(latent, DType.bfloat16), valid=valid
        )

    @staticmethod
    def _state_leaf(cache: CSA2Cache, source_layer_id: int) -> PagedCacheValues:
        """Returns a ratio-2 KV-source layer's ``c2a_state`` leaf.

        Raises:
            ValueError: If the source layer has no ``c2a_state`` cache.
        """
        state_leaf = cache.state_by_layer.get(source_layer_id)
        if state_leaf is None:
            raise ValueError(
                f"ratio-2 KV-source layer {source_layer_id} needs its "
                "c2a_state carry leaf in CSA2Cache.state_by_layer: without "
                "it a forward starting at an odd position cannot pool its "
                "first pair"
            )
        return state_leaf

    def _index_list(
        self,
        state: PendingCSA2Attention,
        rows: RaggedRows,
        selected: TensorValue | None,
        swa_valid_from: TensorValue | None = None,
    ) -> TensorValue:
        """Builds each row's ``[T, 640]`` absolute-slot index list.

        The window slots of positions ``p, p - 1, ..., p - 127`` come first
        (``-1`` below position 0), then the compressed slots of the
        ``selected`` groups' records, ``-1``-padded to 512. On a GPU one
        kernel builds it (:func:`~.csa2.index_list`) unless
        :func:`~max.pipelines.architectures.deepseekV41.model_config.index_list_fused_enabled`
        is off; otherwise graph ops do (:func:`~.csa2.index_list_ops`).

        Args:
            state: :meth:`begin`'s state.
            rows: The forward's bookkeeping.
            selected: ``[T, num_comp]`` int32 raw positions of the selected
                groups' closing tokens, ``-1`` for none; ``None`` on a
                window-only layer.
            swa_valid_from: Per-request SWA validity boundary; see :meth:`end`.
        """
        if selected is not None:
            assert state.comp_collection is not None
            assert state.comp_index is not None
        build = (
            index_list
            if state.q.device.is_gpu() and index_list_fused_enabled()
            else index_list_ops
        )
        return build(
            rows.positions,
            rows.bid,
            state.swa_collection,
            self.layer_info.layer_id,
            valid_from=swa_valid_from,
            selected=selected,
            comp=state.comp_collection,
            comp_layer=state.comp_index or 0,
            ratio=self.layer_info.compress_ratio,
        )

    def _attend(
        self,
        state: PendingCSA2Attention,
        rows: RaggedRows,
        idx: TensorValue,
    ) -> TensorValue:
        """Runs the CSA2 attention over ``idx``'s slots.

        A forward where some request brings more than
        ``_DECODE_MAX_QUERY_TOKENS`` query tokens runs the prefill attention
        if its lists cannot name more than ``PREFILL_ROWS`` distinct slots
        (:func:`prefill_rows_bound`). Every other forward runs the decode
        attention: a prefill naming more slots than its workspace holds
        would abort. The op makes the choice from a host flag rather than a
        graph branch: a decode attention inside an ``ops.cond`` branch read
        the window cache before this forward's window stores landed.
        """
        swa = state.swa_collection
        comp = state.comp_collection
        q = state.q
        t = q.shape[0]
        sink = ops.cast(self.attn_sink, DType.float32)
        host = DeviceRef.CPU()
        tokens, batch = (
            ops.reshape(ops.shape_to_tensor([dim]), [])
            for dim in (t, rows.batch)
        )
        longest = ops.cast(ops.reshape(swa.max_prompt_length, []), DType.int64)
        named = prefill_rows_bound(
            tokens,
            batch,
            longest,
            ops.cast(ops.reshape(swa.max_cache_length, []), DType.int64),
            None if comp is None else self.layer_info.compress_ratio,
        )
        use_prefill = (
            longest > ops.constant(_DECODE_MAX_QUERY_TOKENS, DType.int64, host)
        ) & (named <= ops.constant(PREFILL_ROWS, DType.int64, host))

        return csa2_attention(
            q,
            idx,
            sink,
            swa.kv_blocks,
            None if comp is None else comp.kv_blocks,
            scale=self.scale,
            window_stride=swa.page_stride,
            comp_stride=None if comp is None else comp.page_stride,
            compressed_rows=self.num_comp,
            prefill=use_prefill,
            max_distinct_rows=PREFILL_ROWS,
        )

    def _select(
        self, state: PendingCSA2Attention, rows: RaggedRows
    ) -> tuple[TensorValue, CandidateBlocks | None]:
        """Selects this layer's ``[T, num_comp]`` compressed positions.

        A candidate source also keeps its best blocks for later consumers;
        a consumer selects only among the blocks its source kept. Its own
        selection is never restricted. Positions are unordered and
        ``-1``-padded.
        """
        assert state.index_q is not None and state.index_leaf is not None
        assert state.layer_comp is not None
        layer = self.layer_info
        ratio = layer.compress_ratio
        block_size = self.candidate_block_size
        top_k = self.num_comp
        leaf, q, offsets = state.index_leaf, state.index_q, rows.offsets
        pool: CandidateBlocks | None = None
        if layer.candidate_role == "source" and self.candidate_blocks:
            ids, kept = select_index_keys_block_topk(
                leaf,
                q,
                offsets,
                rows.positions,
                state.layer_comp,
                top_k=top_k,
                ratio=ratio,
                block_size=block_size,
                topk_blocks=self.candidate_topk_blocks,
            )
            pool = CandidateBlocks(kept=kept, compress_ratio=ratio)
            return key_positions(ids, ratio), pool
        if layer.candidate_role == "source":
            pool = CandidateBlocks(kept=None, compress_ratio=ratio)
        kept_blocks = (
            self._candidate_kept()
            if layer.candidate_role == "consumer"
            else None
        )
        if kept_blocks is None:
            ids = select_index_keys(
                leaf, q, offsets, state.layer_comp, top_k=top_k, ratio=ratio
            )
        else:
            ids = select_index_keys_masked(
                leaf,
                q,
                offsets,
                kept_blocks,
                state.layer_comp,
                top_k=top_k,
                ratio=ratio,
                block_size=block_size,
            )
        return key_positions(ids, ratio), pool

    def _candidate_kept(self) -> TensorValue | None:
        """Returns the candidate source's kept block ids, if it can drop one."""
        current = self.selection_box.current
        pool = None if current is None else current.candidate_pool
        # No pool only when the traced layers start past the source layer;
        # selection then stays unrestricted.
        if pool is None:
            return None
        assert isinstance(pool, CandidateBlocks)
        if pool.kept is None:
            return None
        if pool.compress_ratio != self.layer_info.compress_ratio:
            raise ValueError(
                f"candidate consumer layer {self.layer_info.layer_id} "
                f"(ratio {self.layer_info.compress_ratio}) cannot read blocks "
                f"of a ratio-{pool.compress_ratio} source"
            )
        return pool.kept

    def end(
        self,
        state: PendingCSA2Attention,
        rows: RaggedRows,
        *,
        swa_valid_from: TensorValue | None = None,
    ) -> TensorValue:
        """Runs :meth:`__call__` from the indexer's top-k on.

        Takes no ``cache``: ``state`` carries the caches :meth:`begin`
        resolved.

        Args:
            state: :meth:`begin`'s intermediate state.
            rows: The query rows, possibly gathered after :meth:`begin`.
            swa_valid_from: Optional ``[batch]`` integer absolute start of
                each request's valid SWA suffix, on the query device. Pass
                it only for layers whose SWA writes omitted rows. A cut
                layer that wrote all rows in :meth:`begin` must leave this
                unset even when its query rows were gathered. Earlier
                chunks and prefix hits require the caller's cache-validity
                boundary, not simply the current chunk's start. Compressed
                KV selection is unaffected.
        """
        layer = self.layer_info
        if layer.attention_mode == "SWA-only":
            selection_for_layer(layer, self.selection_box.current)
            self.selection_box.current = None
            out = self._attend(
                state, rows, self._index_list(state, rows, None, swa_valid_from)
            )
            if self._selection_trace is not None:
                self._selection_trace.append((layer.layer_id, None))
            return ops.reshape(
                self._output_projection(out, state.freqs_cis),
                state.out_shape,
            )

        if layer.attention_mode in ("Full", "Reindex"):
            produced, pool = self._select(state, rows)
            selection = selection_for_layer(
                layer,
                self.selection_box.current,
                produced=produced,
                candidate_pool=pool,
            )
        elif layer.attention_mode == "Reuse":
            selection = selection_for_layer(layer, self.selection_box.current)
            assert selection is not None
            produced = selection.indices
        else:
            raise ValueError(
                f"unsupported attention mode {layer.attention_mode!r}"
            )

        self.selection_box.current = selection
        out = self._attend(
            state, rows, self._index_list(state, rows, produced, swa_valid_from)
        )
        if self._selection_trace is not None:
            self._selection_trace.append((layer.layer_id, selection))
        return ops.reshape(
            self._output_projection(out, state.freqs_cis),
            state.out_shape,
        )


def make_csa2_attn_factory(
    *,
    num_heads: int,
    head_dim: int,
    window: int,
    num_comp: int = 1,
    trace: (
        list[tuple[int, ForwardSelection[TensorValue, CandidateBlocks] | None]]
        | None
    ) = None,
) -> tuple[
    Callable[[DeepseekV41Config, LayerInfo, DeviceRef, int], Module],
    _ForwardSelectionBox,
]:
    """Builds an ``attn_factory`` and the selection box its layers share.

    Reset ``box.current = None`` before retracing the same layers for a new,
    independent forward.

    Returns:
        The factory and its forward-scoped :class:`_ForwardSelectionBox`.
    """
    box = _ForwardSelectionBox()

    def factory(
        config: DeepseekV41Config,
        layer_info: LayerInfo,
        device: DeviceRef,
        max_seq_len: int,
    ) -> Module:
        return CSA2Attention(
            layer_info,
            device,
            box,
            config,
            hidden_size=config.hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            window=window,
            num_comp=num_comp,
            max_seq_len=max_seq_len,
            trace=trace,
        )

    return factory, box
