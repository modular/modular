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
"""Muse Spark's attention prologue and epilogue norms as graph ops.

`kv_cache.qk_rms_norm_rope_split_store.ragged.paged`: of the `[q | k | v | ...]`
projection, a weightless per-head qk-norm, a Q pre-scale, RoPE and the KV
store in one launch:

    q = rope(rms_norm(q) * q_scale)     -> output
    k = rope(rms_norm(k))               -> key cache
    v                                   -> value cache

`norm.head_rms_norm_sigmoid_gate`: `rms_norm(x) * sigmoid(gate)` per head,
with the gate the projection's last columns.

Both run `nn.head_rms_norm_rope` (the prologue below SM-count tokens runs a
`rowwise` kernel) and match the unfused ops bit for bit.
"""

from std.math import rsqrt
from std.utils.coord import ComptimeInt, Coord, DynamicCoord
from std.utils.numerics import get_accum_type

import extensibility
from extensibility import InputTensor, OutputTensor
from extensibility import _MutableInputTensor as MutableInputTensor
from layout import UNKNOWN_VALUE
from internal_utils.fp8_utils import cast_saturating
from max.gpu.host import DeviceContext
from max.gpu.host.info import is_gpu

from algorithm import rowwise
from algorithm.rowwise_types import at_axis
from kv_cache.types import KVCacheT
from nn._ragged_utils import get_batch_from_row_offsets
from nn.fused_qk_rope import rope_value
from nn.head_rms_norm_rope import (
    HEAD_DIM,
    GateArgs,
    In,
    Out,
    QkArgs,
    head_norm_gate,
    qk_norm_rope_store,
)
from nn.kv_cache import generic_get_paged_cache

from .monoids import ReduceSum


@always_inline
def _in[T: DType, U: DType](t: InputTensor[dtype=T, ...]) -> In[U]:
    return (
        t.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmUntrackedOrigin]()
        .unsafe_bitcast[Scalar[U]]()
    )


@always_inline
def _out[T: DType](t: OutputTensor[dtype=T, ...]) -> Out[DType.bfloat16]:
    return (
        t.unsafe_ptr()
        .unsafe_origin_cast[MutUntrackedOrigin]()
        .unsafe_bitcast[BFloat16]()
    )


@extensibility.register("kv_cache.qk_rms_norm_rope_split_store.ragged.paged")
struct QKRMSNormRopeSplitStore:
    """Registers the `kv_cache.qk_rms_norm_rope_split_store.ragged.paged` graph op.
    """

    @staticmethod
    def execute[
        dtype: DType,
        freq_dtype: DType,
        cache_dtype: DType,
        target: StaticString,
        //,
        n_q_heads: Int,
    ](
        output: OutputTensor[dtype=dtype, rank=2, ...],
        qkv: InputTensor[dtype=dtype, rank=2, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        freqs_cis: InputTensor[dtype=freq_dtype, rank=2, ...],
        kv_blocks: MutableInputTensor[dtype=cache_dtype, rank=6, ...],
        page_stride: InputTensor[dtype=.int64, rank=1, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        layer_idx: UInt32,
        epsilon: Float32,
        q_scale: Float32,
        ctx: DeviceContext,
    ) raises:
        """Writes the roped Q and stores the roped K and the V to the cache.

        Parameters:
            dtype: Element type of the projection and the Q output.
            freq_dtype: Element type of `freqs_cis`.
            cache_dtype: Element type of the KV cache.
            target: Compilation target.
            n_q_heads: Query heads in `qkv` (this device's).

        Args:
            output: The roped Q `[tokens, n_q_heads * head_dim]`.
            qkv: The projection `[tokens, q_dim + 2 * kv_dim + ...]`.
            input_row_offsets: Ragged offsets `[batch + 1]`.
            freqs_cis: Interleaved (cos, sin) pairs `[positions, head_dim]`,
                applied to rotate-half (safetensors) heads.
            kv_blocks: The paged KV cache blocks.
            page_stride: The paged cache's page stride.
            cache_lengths: Each sequence's cached length.
            kv_lookup_table: The paged cache's page table.
            max_prompt_length: The batch's longest prompt.
            max_cache_length: The batch's longest cache.
            layer_idx: The cache layer to store into.
            epsilon: The norm's epsilon.
            q_scale: The factor on the normed Q, rounded to `dtype` first.
            ctx: Device context.

        Raises:
            Error: If the launch fails.
        """
        comptime assert is_gpu[
            target
        ](), "mo.qk_rms_norm_rope_split_store is GPU only"
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            page_stride,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )
        var k_cache = kv_collection.get_key_cache(Int(layer_idx))
        var v_cache = kv_collection.get_value_cache(Int(layer_idx))
        comptime n_kv = type_of(k_cache).kv_params.num_heads
        comptime width = Int(qkv.static_spec.shape_tuple[1])
        comptime assert width != UNKNOWN_VALUE, "the qkv width must be static"
        comptime assert (
            width >= (n_q_heads + 2 * n_kv) * HEAD_DIM
        ), "qkv is not [q | k | v | ...]"
        comptime assert (
            type_of(k_cache).kv_params.head_size == HEAD_DIM
        ), "64-channel heads only"
        comptime assert dtype == DType.bfloat16
        if qkv.dim_size[0]() < ctx.default_device_info.sm_count:
            _rowwise_prologue[target, n_q_heads](
                output,
                qkv,
                input_row_offsets,
                freqs_cis,
                k_cache,
                v_cache,
                epsilon,
                q_scale,
                ctx,
            )
            return
        qk_norm_rope_store[n_q_heads, n_kv, type_of(k_cache).page_size](
            QkArgs[cache_dtype, freq_dtype](
                _in[dtype, DType.bfloat16](qkv),
                _out(output),
                _in[freq_dtype, freq_dtype](freqs_cis),
                _in[DType.uint32, DType.uint32](input_row_offsets),
                _in[DType.uint32, DType.uint32](cache_lengths),
                _in[DType.uint32, DType.uint32](kv_lookup_table),
                k_cache.blocks.ptr.unsafe_origin_cast[MutUntrackedOrigin](),
                v_cache.blocks.ptr.unsafe_origin_cast[MutUntrackedOrigin](),
                Int64(qkv.stride_length[0]()),
                Int64(qkv.dim_size[0]()),
                Int64(input_row_offsets.dim_size[0]() - 1),
                Int64(kv_lookup_table.dim_size[1]()),
                Int64(k_cache._stride()),
                epsilon,
                q_scale.cast[dtype]().cast[DType.float32](),
            ),
            ctx,
        )


@extensibility.register("norm.head_rms_norm_sigmoid_gate")
struct HeadRMSNormSigmoidGate:
    """Registers the `norm.head_rms_norm_sigmoid_gate` graph op."""

    @staticmethod
    def execute[
        dtype: DType,
        //,
        n_heads: Int,
    ](
        output: OutputTensor[dtype=dtype, rank=2, ...],
        x: InputTensor[dtype=dtype, rank=2, ...],
        qkvg: InputTensor[dtype=dtype, rank=2, ...],
        epsilon: Float32,
        ctx: DeviceContext,
    ) raises:
        """Writes `rms_norm(x) * sigmoid(gate)` per 64-channel head, with the
        gate the last `n_heads * 64` columns of `qkvg`.

        Parameters:
            dtype: Element type.
            n_heads: Heads in `x` (this device's).

        Args:
            output: `[tokens, n_heads * 64]`.
            x: The attention output `[tokens, n_heads * 64]`.
            qkvg: The `[q | k | v | gate]` projection.
            epsilon: The norm's epsilon.
            ctx: Device context.

        Raises:
            Error: If the launch fails.
        """
        comptime assert (
            ctx.target.is_gpu()
        ), "norm.head_rms_norm_sigmoid_gate is GPU only"
        comptime width = n_heads * HEAD_DIM
        comptime qkvg_width = Int(qkvg.static_spec.shape_tuple[1])
        comptime assert qkvg_width >= width, "qkvg has no gate"
        comptime assert dtype == DType.bfloat16
        head_norm_gate[n_heads](
            GateArgs(
                _in[dtype, DType.bfloat16](x),
                _in[dtype, DType.bfloat16](qkvg).unsafe_offset(
                    qkvg_width - width
                ),
                _out(output),
                Int64(qkvg.stride_length[0]()),
                Int64(x.dim_size[0]()),
                epsilon,
            ),
            ctx,
        )


def _rowwise_prologue[
    dtype: DType,
    freq_dtype: DType,
    cache_t: KVCacheT,
    //,
    target: StaticString,
    n_q_heads: Int,
](
    output: OutputTensor[dtype=dtype, rank=2, ...],
    qkv: InputTensor[dtype=dtype, rank=2, ...],
    input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
    freqs_cis: InputTensor[dtype=freq_dtype, rank=2, ...],
    k_cache: cache_t,
    v_cache: cache_t,
    epsilon: Float32,
    q_scale: Float32,
    ctx: DeviceContext,
) raises:
    """The prologue as one `rowwise` kernel, a warp per (token, head) row:
    below SM-count tokens it beats `nn.head_rms_norm_rope`'s lane kernel."""
    comptime cache_dtype = cache_t.dtype
    comptime head_dim = cache_t.kv_params.head_size
    comptime n_kv_heads = cache_t.kv_params.num_heads
    comptime half = head_dim // 2
    comptime heads = n_q_heads + 2 * n_kv_heads
    comptime accum = get_accum_type[dtype]()
    comptime simd_width = rowwise.pick_simd_width[
        ReduceSum[accum, 1], target, 64, dtype, accum
    ]()
    # A tile must sit wholly in one half of the head to find its partner.
    comptime assert half % simd_width == 0, "head_dim / 2 % simd width"

    var axis_size_accum = Scalar[accum](head_dim)
    var scale = q_scale.cast[dtype]().cast[accum]()
    var offsets = input_row_offsets.to_tile_tensor[.int64]()

    @inline(.always)
    def body[
        params: rowwise.ContextParams
    ](row_coords: Coord, mut ctx_p: rowwise.Context[params]) {
        var qkv,
        var output,
        var freqs_cis,
        var k_cache,
        var v_cache,
        var offsets,
        var axis_size_accum,
        var epsilon,
        var scale,
    }:
        comptime row_rank = row_coords.rank
        var token = Int(row_coords[0])
        var head = Int(row_coords[1])
        var bi = get_batch_from_row_offsets(offsets, token)
        var pos = k_cache.cache_length(bi) + Int(
            UInt32(token) - rebind[UInt32](offsets.load((bi,)))
        )

        @inline(.always)
        def load[
            width: Int, alignment: Int
        ](idx: DynamicCoord[DType.int, row_rank]) {var qkv} -> SIMD[
            dtype, width
        ]:
            return qkv.load[width, element_alignment=alignment](
                Coord(
                    Int(idx[0]),
                    Int(idx[1]) * head_dim + Int(idx[2]),
                )
            )

        var row = rowwise.Row[
            params, accum, dtype, 2, row_rank, is_cached=True
        ](row_coords, ComptimeInt[head_dim](), ctx_p, load)

        @inline(.always)
        def square[
            width: Int
        ](
            tile: SIMD[dtype, width], idx: DynamicCoord[DType.int, row_rank]
        ) {} -> SIMD[accum, width]:
            var t = tile.cast[accum]()
            return t * t

        # V rows reduce too, so every row's participants run the same
        # collectives; their result is unused.
        var ssq = row.reduce[ReduceSum[accum, params.simd_width]](
            square, load
        ).acc
        var inv_rms = rsqrt(ssq / axis_size_accum + epsilon.cast[accum]())

        @inline(.always)
        def normed_tile[
            width: Int
        ](tile: SIMD[dtype, width], idx: DynamicCoord[DType.int, row_rank]) {
            var inv_rms, var scale, var head
        } -> SIMD[dtype, width]:
            if head >= n_q_heads + n_kv_heads:
                return tile
            var n = (tile.cast[accum]() * inv_rms.slice[width]()).cast[dtype]()
            if head < n_q_heads:
                n = (n.cast[accum]() * scale).cast[dtype]()
            return n

        var normed = row.cache[dtype, shared=True](normed_tile)

        @inline(.always)
        def write[
            width: Int
        ](nc: SIMD[dtype, width], idx: DynamicCoord[DType.int, row_rank]) {
            var normed,
            var load,
            var normed_tile,
            var freqs_cis,
            var output,
            var k_cache,
            var v_cache,
            var head,
            var bi,
            var pos,
            var ctx_p,
        }:
            var d = Int(idx[2])
            if head >= n_q_heads + n_kv_heads:
                v_cache.store(
                    bi,
                    head - n_q_heads - n_kv_heads,
                    pos,
                    d,
                    cast_saturating[cache_dtype](nc),
                )
                return
            # Rope the tile's pairs with their rotate-half partners, as
            # `rope_split_store` does, and keep this tile's half.
            var lo = d < half
            var partner = normed.load[width](
                at_axis[2](idx, d + half if lo else d - half),
                load,
                normed_tile,
            )
            var re = nc if lo else partner
            var im = partner if lo else nc
            var freq = freqs_cis.load[width * 2](Coord(pos, 2 * (d % half)))
            var r = rope_value(re.interleave(im), freq)
            var r_re, r_im = r.deinterleave()
            var roped = rebind[SIMD[dtype, width]](r_re if lo else r_im)
            if head < n_q_heads:
                comptime alignment = ctx_p.element_alignment[dtype, width]()
                output.store[width, element_alignment=alignment](
                    Coord(Int(idx[0]), head * head_dim + d), roped
                )
            else:
                k_cache.store(
                    bi,
                    head - n_q_heads,
                    pos,
                    d,
                    cast_saturating[cache_dtype](roped),
                )

        row.elementwise(normed, write, load, normed_tile)

    rowwise.launch[
        axis=2,
        simd_width=simd_width,
        target=target,
        num_phases=2,
        computationally_expensive=True,
        name="qk_rms_norm_rope_split_store",
    ](
        body,
        Coord(
            Int(qkv.shape()[0]),
            ComptimeInt[heads](),
            ComptimeInt[head_dim](),
        ),
        ctx,
    )
