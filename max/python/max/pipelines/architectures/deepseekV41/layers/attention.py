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
    TensorType,
    TensorValue,
    Weight,
    ops,
)
from max.nn.kernels import (
    KEY_CACHE_INDEX,
    VALUE_CACHE_INDEX,
    ceildiv,
    grouped_matmul_block_scaled,
    grouped_quantize_dynamic_block_scaled,
    kv_cache_gather_rows_ragged,
    store_k_cache_ragged,
    store_v_cache_ragged,
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

from ...deepseekV4.layers.quantization import fp8_qat_quantize
from ...deepseekV4.layers.ragged import arange_to, host_scalar
from ...deepseekV4.layers.rope import apply_rope_tail
from ..layer_resolver import LayerInfo
from ..model_config import DeepseekV41Config, woa_mxfp8_enabled
from .compressor import (
    DeepseekV41Compressor,
    compressed_rope_positions,
)
from .indexer import (
    DeepseekV41Indexer,
    candidate_blocks_active,
    index_key_slots,
    key_positions,
    quantize_queries,
    rope_cos_sin_table,
    select_index_keys,
    select_index_keys_block_topk,
    select_index_keys_masked,
    store_index_keys,
)
from .quantization import nvfp4_qat_quantize
from .query import DeepseekV41Query
from .ragged import V41RaggedRows as RaggedRows
from .reuse import ForwardSelection, bind_cache_sources, selection_for_layer
from .rope import rope_for_layer
from .sparse_attention import sparse_attention

# MXFP8 block size: one E8M0 scale per 32 elements.
_MXFP8_BLOCK_SIZE = 32


def _output_projection_quant_config() -> QuantConfig:
    """Returns the MXFP8 ``QuantConfig`` for the attention output projections.

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
    """Sliding-window cache shared by every layer, indexed by ``layer_id``."""
    main_kv_by_layer: Mapping[int, PagedCacheValues]
    """Owning layer id -> compressed KV cache."""
    index_k_by_layer: Mapping[int, PagedCacheValues]
    """Owning layer id -> indexer key cache.

    Must be a different object from the matching ``main_kv_by_layer`` cache,
    since "Full" layers write to both.
    """
    comp_layer_index_by_layer_id: Mapping[int, int]
    """Owning layer id -> its slot index in the main KV and index key caches."""
    state_by_layer: Mapping[int, PagedCacheValues] = field(default_factory=dict)
    """Owning layer id -> pair-carry cache (``c2a_state``) for ratio two.

    Holds each token's raw float32 ``wkv`` row (key plane) and ``wgate`` row
    (value plane) by position, so a forward starting at an odd position can
    pool against row ``start - 1``. Required by every ratio-2 KV-source
    layer.
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


# Rows per scale tile of the block-scaled grouped matmul; each group's
# activation scales start on a fresh tile.
_SCALE_TILE_ROWS = 128


def _grouped_mxfp8_supported() -> bool:
    """Whether the build target runs the grouped MXFP8 matmul (SM100)."""
    try:
        return accelerator_architecture_name().startswith("sm_10")
    except Exception:
        return False


def _multiples(step: Dim, count: int, device: DeviceRef) -> TensorValue:
    """``[0, step, ..., (count - 1) * step]`` int32, valid for ``step == 0``.

    ``ops.range`` rejects a zero step, which an empty batch would give, so
    this offsets a ``step + 1`` range by ``[0, count)``.
    """
    shifted = ops.range(
        0,
        host_scalar((step + 1) * count),
        host_scalar(step + 1),
        out_dim=count,
        device=device,
        dtype=DType.int32,
    )
    return shifted - ops.range(
        0, count, 1, out_dim=count, device=device, dtype=DType.int32
    )


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


def mxfp8_grouped_projection(
    x: TensorValue,
    weight: TensorValue,
    weight_scale: TensorValue,
    groups: int,
) -> TensorValue:
    """Block-diagonal MXFP8 projection (the grouped einsum ``bgk,grk->bgr``).

    Group ``g``'s ``k`` input columns multiply ``weight`` rows
    ``[g * r, (g + 1) * r)``, with ``r = weight.shape[0] // groups``. Runs as
    one grouped block-scaled matmul in which each group is an expert over
    all ``t`` tokens.

    The activations are quantized group-major: row ``g * t + i`` of the
    ``[groups * t, k]`` ``float8_e4m3fn`` operand is token ``i``'s group
    ``g`` slice, with one ``float8_e8m0fnu`` scale per 32 columns. The
    scales are tile-interleaved like ``weight_scale``, as
    ``[tiles, k / 128, 32, 4, 4]``, and group ``g``'s start at tile
    ``g * ceil(t / 128)``. The product comes back group-major and is
    transposed to token-major. Off SM100, where the grouped matmul does not
    run, each group takes a dense MXFP8 matmul instead. An empty batch
    returns ``[0, groups * r]`` without launching the grouped kernels.

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

    Returns:
        ``[t, groups * r]`` bf16.
    """
    t = x.shape[0]
    r = int(weight.shape[0]) // groups
    # ``Linear`` declares its weight scale on the host; move it beside ``x``.
    weight_scale = weight_scale.to(x.device)
    if not _grouped_mxfp8_supported():
        return _mxfp8_projection_per_group(
            x, weight, _deinterleave_scales(weight_scale), groups
        )
    # The grouped quantizer cannot launch on zero rows, which an idle data
    # parallel replica passes every step.
    out_type = TensorType(DType.bfloat16, [t, groups * r], x.device)
    return ops.cond(
        ops.equal(ops.shape_to_tensor([t])[0], 0),
        [out_type],
        lambda: ops.broadcast_to(
            ops.constant(0.0, DType.bfloat16, device=x.device),
            [t, groups * r],
        ),
        lambda: _mxfp8_grouped_gemm(x, weight, weight_scale, groups),
    )[0]


def _mxfp8_grouped_gemm(
    x: TensorValue,
    weight: TensorValue,
    weight_scale: TensorValue,
    groups: int,
) -> TensorValue:
    """The grouped matmul body of :func:`mxfp8_grouped_projection`."""
    t = x.shape[0]
    k = int(weight.shape[1])
    r = int(weight.shape[0]) // groups
    device = x.device
    rows = t * groups
    # Group-major row g * t + i reads row i * groups + g of the token-major
    # ``[t * groups, k]`` view, so the quantizer gathers the transpose.
    gather = ops.reshape(
        ops.transpose(
            ops.reshape(arange_to(rows, rows, device), [t, groups]), 0, 1
        ),
        [rows],
    )
    starts = _multiples(t, groups + 1, device)
    # A group's scale tile is ``starts[g] // 128 + offsets[g]``; its padded
    # start is tile ``g * ceil(t / 128)``. Float floor is exact below 2**24.
    padded = _multiples(ceildiv(t, Dim(_SCALE_TILE_ROWS)), groups, device)
    unpadded = ops.cast(
        ops.floor(
            ops.cast(starts[:groups], DType.float32) / float(_SCALE_TILE_ROWS)
        ),
        DType.int32,
    )
    offsets = ops.cast(padded - unpadded, DType.uint32)
    starts = ops.cast(starts, DType.uint32)
    expert_ids = ops.range(
        0, groups, 1, out_dim=groups, device=device, dtype=DType.int32
    )
    ones = ops.broadcast_to(
        ops.constant(1.0, DType.float32, device=device), [groups]
    )
    x_q, x_scales = grouped_quantize_dynamic_block_scaled(
        ops.reshape(ops.cast(x, DType.bfloat16), [rows, k]),
        row_offsets=starts,
        scales_offsets=offsets,
        expert_ids=expert_ids,
        sf_tensor=ones,
        sf_vector_size=32,
        scales_type=DType.float8_e8m0fnu,
        out_type=DType.float8_e4m3fn,
        indices=gather,
    )
    usage_stats = ops.concat(
        [
            ops.reshape(ops.cast(host_scalar(t), DType.uint32), [1]),
            ops.constant([groups], DType.uint32, DeviceRef.CPU()),
        ]
    )
    out = grouped_matmul_block_scaled(
        x_q,
        ops.reshape(weight, [groups, r, k]),
        x_scales,
        ops.reshape(
            weight_scale,
            [groups, r // _SCALE_TILE_ROWS, k // 128, 32, 4, 4],
        ),
        starts,
        offsets,
        expert_ids,
        ones,
        usage_stats,
    )
    return ops.reshape(
        ops.transpose(ops.reshape(out, [groups, t, r]), 0, 1),
        [t, groups * r],
    )


@dataclass(frozen=True)
class CompressedKV:
    """:meth:`CSA2Attention.compress` output, one row per raw token."""

    latent: TensorValue
    """``[T, head_dim]`` bf16 pooled, normalized latent, before RoPE.

    Zero on rows that do not close a group. Input to the index keys.
    """
    latent_rope: TensorValue
    """bf16 ``latent`` with RoPE applied at the group's first position."""
    latent_rope_f32: TensorValue
    """``latent_rope`` before bf16 rounding.

    Rounded onto the NVFP4 grid before it is stored in the compressed cache.
    """
    freqs_cis: TensorValue
    """``[T, rope_head_dim // 2, 2]`` cos/sin at each compressed position.

    Shared with the index keys.
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
    layer_swa: TensorValue
    comp_collection: PagedCacheValues | None
    layer_comp: TensorValue | None
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
    table, so scoring is ``O(max_seq_len)`` per query row. A candidate-source
    layer also keeps its best-scoring blocks of positions, and later
    candidate-consumer layers restrict their scores to those blocks.
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
        self.num_comp = num_comp
        self.scale = head_dim**-0.5
        self.device = device
        self._selection_trace = trace
        self.max_seq_len = max_seq_len
        self.candidate_topk_blocks = config.candidate_topk_blocks
        self.candidate_block_size = config.candidate_block_size
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
        # bf16, not `config.dtype` (the packed quantized dtype): these
        # Linears load host-dequantized weights.
        self.query_proj = DeepseekV41Query(
            hidden_size,
            config.q_lora_rank,
            head_dim,
            num_heads,
            config.rms_norm_eps,
            DType.bfloat16,
            device,
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
        t = out.shape[0]
        out = ops.reshape(
            apply_rope_tail(
                ops.reshape(out, [1, t, self.num_heads, self.head_dim]),
                freqs_cis,
                int(freqs_cis.shape[-2]) * 2,
                inverse=True,
            ),
            [t, self.num_heads, self.head_dim],
        )
        if self.woa_mxfp8:
            assert self.wo_a.weight_scale is not None
            return self.wo_b(
                mxfp8_grouped_projection(
                    ops.reshape(out, [t, self.num_heads * self.head_dim]),
                    self.wo_a.weight,
                    self.wo_a.weight_scale,
                    self.o_groups,
                )
            )
        return self.wo_b(
            bf16_grouped_projection(
                ops.reshape(out, [t, self.num_heads * self.head_dim]),
                self.wo_a.weight,
                self.o_groups,
            )
        )

    def __call__(
        self, h: TensorValue, rows: RaggedRows, cache: object | None
    ) -> TensorValue:
        return self.end(self.begin(h, rows, cache), rows)

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
        qr, q, kv_raw = self.query_proj(h_flat, freqs_cis)
        q = ops.reshape(q, [-1, self.num_heads, self.head_dim])
        state = PendingCSA2Attention(
            q=q,
            out_shape=[b, s, hidden],
            freqs_cis=freqs_cis,
            swa_collection=cache.swa_collection,
            layer_swa=ops.constant(
                layer.layer_id, DType.uint32, DeviceRef.CPU()
            ),
            comp_collection=None,
            layer_comp=None,
        )
        # The SWA record: full-width RMSNorm of the KV latent, then RoPE on
        # its trailing `qk_rope_head_dim` dims.
        kv_latent = apply_rope_tail(
            self.kv_norm(kv_raw), freqs_cis, int(freqs_cis.shape[-2]) * 2
        )
        # Round the bf16 row onto the MXFP8 grid over all dims.
        kv_latent = fp8_qat_quantize(
            ops.cast(kv_latent, DType.bfloat16),
            block_size=_MXFP8_BLOCK_SIZE,
        )
        # Every layer, including "SWA-only", owns its sliding-window slot.
        store_k_cache_ragged(
            cache.swa_collection,
            ops.reshape(kv_latent, [-1, 1, self.head_dim]),
            rows.offsets,
            state.layer_swa,
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
        state.layer_comp = layer_comp

        if layer.attention_mode in ("Full", "Reindex"):
            indexer = self.indexer
            assert indexer is not None
            rope_table = rope_cos_sin_table(self.rope.freqs_cis_base())
            index_queries, index_weights = indexer(qr, h_flat)
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
                # latent feeds the index key; the RoPE'd copy is stored in
                # the compressed cache.
                assert self.compressor is not None
                raw = ops.reshape(
                    self.compressor.project(h_flat),
                    [-1, self.head_dim * layer.compress_ratio],
                )
                prior_row = prior_valid = None
                if layer.compress_ratio == 2:
                    prior_row, prior_valid = self._pair_carry(
                        cache, binding.kv_source_layer_id, rows, raw, layer_comp
                    )
                compressed = self.compress(
                    raw, rows.positions, rows.offsets, prior_row, prior_valid
                )
                # One row per raw token, so the stores reuse `rows.offsets`;
                # non-closing rows hold zeros masked by visibility.
                # Round the float32 rotated row onto the NVFP4 grid. Exact
                # in bf16: E2M1 x E4M3 has at most 6 significant bits.
                stored = ops.cast(
                    nvfp4_qat_quantize(compressed.latent_rope_f32),
                    DType.bfloat16,
                )
                store_k_cache_ragged(
                    binding.main_kv_leaf,
                    ops.unsqueeze(stored, 1),
                    rows.offsets,
                    layer_comp,
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
        positions: TensorValue,
        offsets: TensorValue,
        prior_row: TensorValue | None = None,
        prior_valid: TensorValue | None = None,
    ) -> CompressedKV:
        """Pools ``raw`` and RoPEs the result at its compressed positions.

        Uses no cache kernels, so it runs on any device.

        Args:
            raw: ``[T, head_dim * ratio]`` float32, this layer's
                ``compressor.project`` output.
            positions: ``[T]`` int32 absolute token positions.
            offsets: ``[B + 1]`` uint32 request row offsets.
            prior_row: Ratio two only: ``[B, 2 * head_dim]`` float32, each
                request's raw row at position ``start - 1``.
            prior_valid: Ratio two only: ``[B]`` bool, whether that row
                exists.

        Returns:
            The pooled latent, its RoPE'd copies, cos/sin and validity.
        """
        assert self.compressor is not None
        active = ops.broadcast_to(
            ops.constant(True, DType.bool, raw.device), [raw.shape[0]]
        )
        latent, valid = self.compressor(
            raw,
            ops.cast(positions, DType.int64),
            offsets,
            active,
            prior_row,
            prior_valid,
        )
        freqs_cis = ops.gather(
            self.rope.freqs_cis_base(),
            compressed_rope_positions(
                positions, self.layer_info.compress_ratio
            ),
            axis=0,
        )
        # RoPE in float32, then round to bf16. `apply_rope_tail` needs a
        # sequence axis at 1.
        latent_rope = ops.reshape(
            apply_rope_tail(
                ops.unsqueeze(ops.cast(latent, DType.float32), 0),
                freqs_cis,
                int(freqs_cis.shape[-2]) * 2,
            ),
            [-1, self.head_dim],
        )
        return CompressedKV(
            latent=latent,
            latent_rope=ops.cast(latent_rope, DType.bfloat16),
            latent_rope_f32=latent_rope,
            freqs_cis=freqs_cis,
            valid=valid,
        )

    def _pair_carry(
        self,
        cache: CSA2Cache,
        source_layer_id: int,
        rows: RaggedRows,
        raw: TensorValue,
        layer_comp: TensorValue,
    ) -> tuple[TensorValue, TensorValue]:
        """Reads each request's predecessor raw row, then stores this step's.

        Row ``start - 1`` of the ``c2a_state`` cache holds the raw
        ``wkv``/``wgate`` row written by the previous forward; ``start > 0``
        is assumed to imply it is present, which holds for append-only
        forwards. A prefix-cache hit ending at an odd length is not handled.

        Returns:
            ``[B, 2 * head_dim]`` float32 predecessor rows and ``[B]`` bool
            validity.

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
        width = self.head_dim
        device = raw.device
        prior_valid = rows.starts > 0
        predecessor = ops.max(
            rows.starts - 1, ops.constant(0, DType.int32, device)
        )
        # One gathered row per request.
        per_request = ops.cast(
            arange_to(rows.batch + 1, rows.batch + 1, device), DType.uint32
        )
        prior_kv, prior_score = (
            ops.squeeze(
                kv_cache_gather_rows_ragged(
                    state_leaf,
                    ops.unsqueeze(predecessor, -1),
                    per_request,
                    layer_comp,
                    key_or_value=plane,
                ),
                1,
            )
            for plane in (KEY_CACHE_INDEX, VALUE_CACHE_INDEX)
        )
        store_k_cache_ragged(
            state_leaf,
            ops.unsqueeze(raw[:, :width], 1),
            rows.offsets,
            layer_comp,
        )
        store_v_cache_ragged(
            state_leaf,
            ops.unsqueeze(raw[:, width:], 1),
            rows.offsets,
            layer_comp,
        )
        return ops.concat([prior_kv, prior_score], axis=-1), prior_valid

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
        active = candidate_blocks_active(
            self.max_seq_len,
            ratio,
            block_size=block_size,
            topk_blocks=self.candidate_topk_blocks,
        )
        leaf, q, offsets = state.index_leaf, state.index_q, rows.offsets
        pool: CandidateBlocks | None = None
        if layer.candidate_role == "source" and active:
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

    def end(self, state: PendingCSA2Attention, rows: RaggedRows) -> TensorValue:
        """Runs :meth:`__call__` from the indexer's top-k on.

        Takes no ``cache``: ``state`` carries the caches :meth:`begin`
        resolved.
        """
        layer = self.layer_info
        if layer.attention_mode == "SWA-only":
            selection_for_layer(layer, self.selection_box.current)
            self.selection_box.current = None
            out = sparse_attention(
                state.q,
                rows.offsets,
                None,
                self.attn_sink,
                state.swa_collection,
                None,
                state.layer_swa,
                None,
                scale=self.scale,
                window=self.window,
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
        out = sparse_attention(
            state.q,
            rows.offsets,
            produced,
            self.attn_sink,
            state.swa_collection,
            state.comp_collection,
            state.layer_swa,
            state.layer_comp,
            scale=self.scale,
            window=self.window,
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
