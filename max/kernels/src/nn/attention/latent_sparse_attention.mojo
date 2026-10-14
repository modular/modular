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
"""Sparse attention over a shared K=V latent held in two paged leaves.

DeepSeek-V4's compressed sparse attention attends from ``q[t, h, :]`` to one
shared latent row per key -- the same row serves as key and value for every
head -- and the keys a query sees come from two caches:

* the last ``window`` token positions, kept in a sliding-window leaf paged by
  token position, and
* a per-query list of compressed entries (the indexer's top-k, or every
  closed window on ratio-128 layers), kept in a leaf whose page holds
  ``slots_per_page`` entries and is addressed by entry index. Because the
  leaf's ``page_size`` parameter is read off the block shape, ``load`` on that
  leaf already pages by entry, so the entry index is passed straight through
  as the token index.

``attn_sink[h]`` enters the softmax denominator only: the running max is
taken over the gathered scores alone, then ``den += exp(sink - max)``. There
is no value row for the sink.

The batch is ragged: query row ``t`` belongs to the sequence ``b`` with
``input_row_offsets[b] <= t < input_row_offsets[b + 1]`` and sits at absolute
position ``pos = cache_lengths[b] + t - input_row_offsets[b]`` in the window
leaf. Its window keys are positions ``max(0, pos - window + 1) .. pos``, all
of which must already be stored -- the store ops run before this op in the
same graph, as they do for every other paged attention. Compressed keys are
``comp_indices[t, k] >= 0``; ``-1`` is skipped.

On SM100 with a BF16 512-wide latent, the op runs on the sparse MLA decode
kernel (tensor cores, split-K): the compressed entries are its main sparse
list, the window its extra always-attend list, and every query row is its own
one-token batch so each row carries its own list lengths. A plan kernel turns
the window range and the compressed list (compacted past its ``-1`` slots)
into flat row numbers into both leaves, which the kernel reads through
page-agnostic views of the two leaves.

Elsewhere, one block per (query row, group of heads); each warp owns
``heads_per_warp`` heads and reads every key row once as a lane-strided
vector, so a key costs one vector load per warp plus a warp reduction per
head. This is the portable form of the kernel; it does not use tensor cores.
"""

from std.math import ceildiv, exp
from std.memory import Layout as AllocLayout, alloc, dealloc
from std.utils.numerics import min_or_neg_inf

from max.gpu import WARP_SIZE, block_idx, lane_id, warp_id
from max.gpu.host import DeviceContext
from max.gpu.host.info import _is_sm10x_gpu, is_cpu
import max.gpu.primitives.warp as warp

from kv_cache.types import (
    KVCacheStaticParams,
    KVCacheT,
    PagedKVCache,
    PagedKVCacheCollection,
)
from layout import (
    Idx,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from nn.attention.gpu.mla import flare_mla_decoding
from nn.attention.mha_mask import NullMask
from nn.attention.mha_utils import MHAConfig, NonNullPointer

# Page size of the flat row views the SM100 path reads both leaves through. A
# multiple of 64 keeps split-K boundaries on whole tiles.
comptime _VIEW_PAGE = 128


@inline(.always)
def _batch_of_row(
    row: Int,
    row_offsets: ImmPointer[UInt32, ImmutAnyOrigin],
    num_batches: Int,
) -> Int:
    var r = UInt32(row)
    for b in range(num_batches):
        if (
            r >= row_offsets[unsafe_offset=b]
            and r < row_offsets[unsafe_offset=b + 1]
        ):
            return b
    return 0


@inline(.always)
def _attend_key[
    lane_width: Int,
    heads_per_warp: Int,
    round_p: Bool,
](
    k: SIMD[DType.float32, lane_width],
    q: Array[SIMD[DType.float32, lane_width], heads_per_warp],
    mut m: Array[Float32, heads_per_warp],
    mut l: Array[Float32, heads_per_warp],
    mut acc: Array[SIMD[DType.float32, lane_width], heads_per_warp],
):
    """One online-softmax step for every head this warp owns.

    Every lane holds a ``lane_width`` slice of ``k`` and of each head's
    scaled query; the warp reduction leaves the full score in all lanes, so
    the running max and denominator stay lane-uniform without a broadcast.
    `round_p` rounds the weight to bf16 for the value product only, as the
    SM100 decode feeds P to its tensor-core PV product (diagnostic).
    """
    comptime for i in range(heads_per_warp):
        var s = warp.sum((q[i] * k).reduce_add())
        if s > m[i]:
            var corr = exp(m[i] - s)
            l[i] = l[i] * corr
            acc[i] = acc[i] * corr
            m[i] = s
        var p = exp(s - m[i])
        l[i] = l[i] + p
        comptime if round_p:
            p = p.cast[DType.bfloat16]().cast[DType.float32]()
        acc[i] = acc[i] + p * k


def _latent_sparse_attention_gpu_kernel[
    swa_t: KVCacheT,
    comp_t: KVCacheT,
    q_type: DType,
    out_type: DType,
    //,
    head_dim: Int,
    heads_per_warp: Int,
    warps_per_block: Int,
    window: Int,
    round_p: Bool = False,
](
    out_ptr: Pointer[Scalar[out_type], MutAnyOrigin],
    q_ptr: ImmPointer[Scalar[q_type], ImmutAnyOrigin],
    row_offsets: ImmPointer[UInt32, ImmutAnyOrigin],
    comp_indices: ImmPointer[Int32, ImmutAnyOrigin],
    attn_sink: ImmPointer[Float32, ImmutAnyOrigin],
    swa_cache: swa_t,
    comp_cache: comp_t,
    num_heads: Int32,
    num_batches: Int32,
    num_comp: Int32,
    q_stride0: Int32,
    q_stride1: Int32,
    out_stride0: Int32,
    out_stride1: Int32,
    idx_stride0: Int32,
    idx_stride1: Int32,
    scale: Float32,
):
    comptime lane_width = head_dim // WARP_SIZE
    var t = block_idx.x
    var lane = Int(lane_id())
    var h0 = (block_idx.y * warps_per_block + Int(warp_id())) * heads_per_warp
    var nh = Int(num_heads)
    if h0 >= nh:
        return

    var b = _batch_of_row(t, row_offsets, Int(num_batches))
    var pos = swa_cache.cache_length(b) + t - Int(row_offsets[unsafe_offset=b])
    var d0 = lane * lane_width

    var q = Array[SIMD[DType.float32, lane_width], heads_per_warp](
        fill=SIMD[DType.float32, lane_width](0)
    )
    var m = Array[Float32, heads_per_warp](fill=min_or_neg_inf[DType.float32]())
    var l = Array[Float32, heads_per_warp](fill=Float32(0))
    var acc = Array[SIMD[DType.float32, lane_width], heads_per_warp](
        fill=SIMD[DType.float32, lane_width](0)
    )
    comptime for i in range(heads_per_warp):
        var h = h0 + i
        if h < nh:
            q[i] = (
                q_ptr.unsafe_load[width=lane_width](
                    t * Int(q_stride0) + h * Int(q_stride1) + d0
                ).cast[DType.float32]()
                * scale
            )

    var start = max(pos - window + 1, 0)
    for p in range(start, pos + 1):
        var k = swa_cache.load[width=lane_width, output_dtype=DType.float32](
            b, 0, p, d0
        )
        _attend_key[lane_width, heads_per_warp, round_p](k, q, m, l, acc)

    for j in range(Int(num_comp)):
        var e = Int(
            comp_indices[
                unsafe_offset=t * Int(idx_stride0) + j * Int(idx_stride1)
            ]
        )
        if e < 0:
            continue
        var k = comp_cache.load[width=lane_width, output_dtype=DType.float32](
            b, 0, e, d0
        )
        _attend_key[lane_width, heads_per_warp, round_p](k, q, m, l, acc)

    comptime for i in range(heads_per_warp):
        var h = h0 + i
        if h < nh:
            var den = l[i] + exp(attn_sink[unsafe_offset=h] - m[i])
            out_ptr.unsafe_store(
                t * Int(out_stride0) + h * Int(out_stride1) + d0,
                (acc[i] / den).cast[out_type](),
            )


@inline(.always)
def _cpu_attend[
    cache_t: KVCacheT
](
    cache: cache_t,
    b: Int,
    key: Int,
    head_dim: Int,
    qh: Pointer[Float32, MutAnyOrigin],
    acc: Pointer[Float32, MutAnyOrigin],
    mut m: Float32,
    mut l: Float32,
):
    var s = Float32(0)
    for d in range(head_dim):
        s += (
            qh[unsafe_offset=d]
            * cache.load[width=1, output_dtype=DType.float32](b, 0, key, d)[0]
        )
    if s > m:
        var corr = exp(m - s)
        l *= corr
        for d in range(head_dim):
            acc[unsafe_offset=d] *= corr
        m = s
    var p = exp(s - m)
    l += p
    for d in range(head_dim):
        acc[unsafe_offset=d] += (
            p * cache.load[width=1, output_dtype=DType.float32](b, 0, key, d)[0]
        )


def _latent_sparse_attention_cpu[
    swa_t: KVCacheT,
    comp_t: KVCacheT,
    q_type: DType,
    out_type: DType,
    //,
    head_dim: Int,
    window: Int,
](
    out_ptr: Pointer[Scalar[out_type], MutAnyOrigin],
    q_ptr: ImmPointer[Scalar[q_type], ImmutAnyOrigin],
    row_offsets: ImmPointer[UInt32, ImmutAnyOrigin],
    comp_indices: ImmPointer[Int32, ImmutAnyOrigin],
    attn_sink: ImmPointer[Float32, ImmutAnyOrigin],
    swa_cache: swa_t,
    comp_cache: comp_t,
    num_rows: Int,
    num_heads: Int,
    num_batches: Int,
    num_comp: Int,
    q_stride0: Int,
    q_stride1: Int,
    out_stride0: Int,
    out_stride1: Int,
    idx_stride0: Int,
    idx_stride1: Int,
    scale: Float32,
):
    """Scalar reference path; the same math as the GPU kernel, one key at a time.
    """
    var acc_alloc = alloc(AllocLayout[Float32](count=head_dim))
    var qh_alloc = alloc(AllocLayout[Float32](count=head_dim))
    var acc = rebind[Pointer[Float32, MutAnyOrigin]](acc_alloc.unsafe_ptr())
    var qh = rebind[Pointer[Float32, MutAnyOrigin]](qh_alloc.unsafe_ptr())
    for t in range(num_rows):
        var b = _batch_of_row(t, row_offsets, num_batches)
        var pos = (
            swa_cache.cache_length(b) + t - Int(row_offsets[unsafe_offset=b])
        )
        var start = max(pos - window + 1, 0)
        for h in range(num_heads):
            for d in range(head_dim):
                qh[unsafe_offset=d] = (
                    q_ptr[unsafe_offset=t * q_stride0 + h * q_stride1 + d].cast[
                        DType.float32
                    ]()
                    * scale
                )
                acc[unsafe_offset=d] = 0
            var m = min_or_neg_inf[DType.float32]()
            var l = Float32(0)
            for p in range(start, pos + 1):
                _cpu_attend(swa_cache, b, p, head_dim, qh, acc, m, l)
            for j in range(num_comp):
                var e = Int(
                    comp_indices[
                        unsafe_offset=t * idx_stride0 + j * idx_stride1
                    ]
                )
                if e >= 0:
                    _cpu_attend(comp_cache, b, e, head_dim, qh, acc, m, l)
            var den = l + exp(attn_sink[unsafe_offset=h] - m)
            for d in range(head_dim):
                out_ptr[unsafe_offset=t * out_stride0 + h * out_stride1 + d] = (
                    acc[unsafe_offset=d] / den
                ).cast[out_type]()
    dealloc(acc_alloc^)
    dealloc(qh_alloc^)


def _sm100_decode_plan_kernel[
    swa_t: KVCacheT,
    comp_t: KVCacheT,
    //,
    window: Int,
](
    comp_rows: Pointer[Int32, MutAnyOrigin],
    comp_len: Pointer[Int32, MutAnyOrigin],
    win_rows: Pointer[Int32, MutAnyOrigin],
    win_len: Pointer[Int32, MutAnyOrigin],
    row_batches: Pointer[UInt32, MutAnyOrigin],
    row_lengths: Pointer[UInt32, MutAnyOrigin],
    scalar_args: Pointer[Int64, MutAnyOrigin],
    row_offsets: ImmPointer[UInt32, ImmutAnyOrigin],
    comp_indices: ImmPointer[Int32, ImmutAnyOrigin],
    swa_cache: swa_t,
    comp_cache: comp_t,
    num_rows: Int32,
    num_batches: Int32,
    num_comp: Int32,
    comp_stride: Int32,
    idx_stride0: Int32,
    idx_stride1: Int32,
):
    """Writes one query row's key lists as flat leaf rows, one warp per row.

    The compressed list is compacted past its `-1` slots so its length alone
    bounds it. `row_batches` / `row_lengths` make every query row its own
    batch, and `scalar_args` is the decode dispatch buffer for that batching.

    The decode clamps a row's main list to its batch's cache length plus
    one, a token-count bound that means nothing for compressed entries, so
    `row_lengths` holds the list length itself and the clamp never cuts it.
    """
    var t = Int(block_idx.x)
    var lane = Int(lane_id())
    var b = _batch_of_row(t, row_offsets, Int(num_batches))
    var pos = swa_cache.cache_length(b) + t - Int(row_offsets[unsafe_offset=b])
    var nc = Int(num_comp)
    var out_base = t * Int(comp_stride)

    var count = 0
    for c0 in range(0, nc, WARP_SIZE):
        var j = c0 + lane
        var e = Int32(-1)
        if j < nc:
            e = comp_indices[
                unsafe_offset=t * Int(idx_stride0) + j * Int(idx_stride1)
            ]
        var valid = Int32(1) if e >= 0 else Int32(0)
        var slot = count + Int(warp.prefix_sum[exclusive=True](valid))
        if e >= 0:
            comp_rows[unsafe_offset=out_base + slot] = Int32(
                comp_cache.row_idx(UInt32(b), UInt32(e))
            )
        count += Int(warp.sum(valid))
    for j in range(count + lane, Int(comp_stride), WARP_SIZE):
        comp_rows[unsafe_offset=out_base + j] = -1

    var start = max(pos - window + 1, 0)
    for i in range(lane, window, WARP_SIZE):
        var p = start + i
        var r = Int32(-1)
        if p <= pos:
            r = Int32(swa_cache.row_idx(UInt32(b), UInt32(p)))
        win_rows[unsafe_offset=t * window + i] = r

    if lane == 0:
        comp_len[unsafe_offset=t] = Int32(count)
        win_len[unsafe_offset=t] = Int32(pos - start + 1)
        row_batches[unsafe_offset=t] = UInt32(t)
        row_lengths[unsafe_offset=t] = UInt32(count)
        if t == 0:
            row_batches[unsafe_offset=Int(num_rows)] = UInt32(num_rows)
            scalar_args[unsafe_offset=0] = Int64(num_rows)
            scalar_args[unsafe_offset=1] = 1
            scalar_args[unsafe_offset=2] = 0


def _flat_row_view[
    dtype: DType,
    kv_params: KVCacheStaticParams,
    //,
](
    cache: PagedKVCache[dtype, kv_params, ...],
    row_lengths: Pointer[UInt32, MutAnyOrigin],
    num_rows: Int,
) -> PagedKVCacheCollection[
    dtype,
    kv_params,
    _VIEW_PAGE,
    MutAnyOrigin,
    ImmutAnyOrigin,
    ImmutAnyOrigin,
    MutUntrackedOrigin,
]:
    """Views one layer of a leaf as a single-layer cache whose rows are flat.

    The view's block stride equals its page size, so the encoded sparse index
    `block * page_size + slot` it decodes is the flat row `cache.row_idx`
    names, whatever the leaf's own page size and layer count. The view's
    cache lengths are `row_lengths`, one per query row. It keeps the leaf's
    lookup table only to fill the field: the sparse decode addresses rows by
    index and never reads it.
    """
    var num_pages = ceildiv(cache.num_kv_rows(), _VIEW_PAGE)
    return PagedKVCacheCollection[
        dtype,
        kv_params,
        _VIEW_PAGE,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutUntrackedOrigin,
    ](
        TileTensor(
            rebind[Pointer[Scalar[dtype], MutAnyOrigin]](cache.blocks.ptr),
            row_major(
                Int64(num_pages),
                Idx[1],
                Idx[1],
                Idx[_VIEW_PAGE],
                Idx[kv_params.num_heads],
                Idx[kv_params.head_size],
            ),
        ),
        TileTensor(
            rebind[Pointer[UInt32, ImmutAnyOrigin]](row_lengths),
            row_major(Int64(num_rows)),
        ),
        TileTensor(
            rebind[Pointer[UInt32, ImmutAnyOrigin]](cache.lookup_table.ptr),
            row_major(
                Int64(cache.lookup_table.dim[0]()),
                Int64(cache.lookup_table.dim[1]()),
            ),
        ),
        UInt32(1),
        cache.max_context_length(),
    )


def _latent_sparse_attention_sm100[
    q_type: DType,
    out_type: DType,
    //,
    num_heads: Int,
    head_dim: Int,
    window: Int,
    split_k: Bool,
](
    out_ptr: Pointer[Scalar[out_type], MutAnyOrigin],
    q_ptr: ImmPointer[Scalar[q_type], ImmutAnyOrigin],
    row_offsets: ImmPointer[UInt32, ImmutAnyOrigin],
    comp_indices: ImmPointer[Int32, ImmutAnyOrigin],
    attn_sink: ImmPointer[Float32, ImmutAnyOrigin],
    swa_cache: PagedKVCache[...],
    comp_cache: PagedKVCache[...],
    num_rows: Int,
    num_batches: Int,
    num_comp: Int,
    idx_stride0: Int,
    idx_stride1: Int,
    scale: Float32,
    ctx: DeviceContext,
) raises:
    # Both leaves' views must be one cache type for `extra_k`; equal dtype
    # and kv_params make the view types equal, which the rebind below relies
    # on.
    comptime assert (
        type_of(swa_cache).dtype == type_of(comp_cache).dtype
        and type_of(swa_cache).kv_params == type_of(comp_cache).kv_params
    ), "the window and compressed leaves must share dtype and kv_params"
    # A zero-width list still needs a real buffer behind its pointer.
    var comp_stride = max(num_comp, 1)
    var comp_rows = ctx.enqueue_create_buffer[DType.int32](
        num_rows * comp_stride
    )
    var comp_len = ctx.enqueue_create_buffer[DType.int32](num_rows)
    var win_rows = ctx.enqueue_create_buffer[DType.int32](num_rows * window)
    var win_len = ctx.enqueue_create_buffer[DType.int32](num_rows)
    var row_batches = ctx.enqueue_create_buffer[DType.uint32](num_rows + 1)
    var row_lengths = ctx.enqueue_create_buffer[DType.uint32](num_rows)
    var scalar_args = ctx.enqueue_create_buffer[DType.int64](3)

    var comp_rows_ptr = rebind[Pointer[Int32, MutAnyOrigin]](
        comp_rows.unsafe_ptr()
    )
    var comp_len_ptr = rebind[Pointer[Int32, MutAnyOrigin]](
        comp_len.unsafe_ptr()
    )
    var win_rows_ptr = rebind[Pointer[Int32, MutAnyOrigin]](
        win_rows.unsafe_ptr()
    )
    var win_len_ptr = rebind[Pointer[Int32, MutAnyOrigin]](win_len.unsafe_ptr())
    var row_batches_ptr = rebind[Pointer[UInt32, MutAnyOrigin]](
        row_batches.unsafe_ptr()
    )
    var row_lengths_ptr = rebind[Pointer[UInt32, MutAnyOrigin]](
        row_lengths.unsafe_ptr()
    )
    var scalar_args_ptr = rebind[Pointer[Int64, MutAnyOrigin]](
        scalar_args.unsafe_ptr()
    )

    comptime plan = _sm100_decode_plan_kernel[
        swa_t=type_of(swa_cache),
        comp_t=type_of(comp_cache),
        window=window,
    ]
    ctx.enqueue_function[plan](
        comp_rows_ptr,
        comp_len_ptr,
        win_rows_ptr,
        win_len_ptr,
        row_batches_ptr,
        row_lengths_ptr,
        scalar_args_ptr,
        row_offsets,
        comp_indices,
        swa_cache,
        comp_cache,
        Int32(num_rows),
        Int32(num_batches),
        Int32(num_comp),
        Int32(comp_stride),
        Int32(idx_stride0),
        Int32(idx_stride1),
        grid_dim=num_rows,
        block_dim=WARP_SIZE,
    )

    var comp_view = _flat_row_view(
        comp_cache, row_lengths_ptr, num_rows
    ).get_key_cache(0)
    var swa_view = _flat_row_view(
        swa_cache, row_lengths_ptr, num_rows
    ).get_key_cache(0)

    flare_mla_decoding[
        rank=3,
        config=MHAConfig[q_type](num_heads, head_dim),
        ragged=True,
        sparse=True,
        has_extra_k=True,
    ](
        TileTensor(out_ptr, row_major(num_rows, Idx[num_heads], Idx[head_dim])),
        TileTensor(q_ptr, row_major(num_rows, Idx[num_heads], Idx[head_dim])),
        comp_view,
        NullMask(),
        TileTensor(row_batches_ptr, row_major(num_rows + 1)),
        scale,
        ctx,
        TileTensor(scalar_args_ptr, row_major((Idx[3],))),
        q_max_seq_len=1,
        d_indices=comp_rows_ptr,
        indices_stride=comp_stride,
        topk_lengths=NonNullPointer[DType.int32](comp_len),
        attn_sink_ptr=NonNullPointer[DType.float32](attn_sink),
        # Equal by the assert above; the compiler compares the two views'
        # parameters symbolically and can't see it.
        extra_k=rebind[type_of(comp_view)](swa_view),
        extra_d_indices=win_rows_ptr,
        extra_indices_stride=window,
        extra_topk_lengths=NonNullPointer[DType.int32](win_len),
        num_partitions_in=Optional[Int](None) if split_k else Optional[Int](1),
    )
    _ = comp_rows^
    _ = comp_len^
    _ = win_rows^
    _ = win_len^
    _ = row_batches^
    _ = row_lengths^
    _ = scalar_args^


def latent_sparse_attention_ragged_paged[
    q_type: DType,
    out_type: DType,
    //,
    target: StaticString,
    window: Int,
    split_k: Bool = True,
    portable_p_bf16: Bool = False,
](
    output: TileTensor[mut=True, out_type, address_space=.GENERIC, ...],
    q: TileTensor[mut=False, q_type, address_space=.GENERIC, ...],
    input_row_offsets: TileTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    comp_indices: TileTensor[mut=False, .int32, address_space=.GENERIC, ...],
    attn_sink: TileTensor[mut=False, .float32, address_space=.GENERIC, ...],
    swa_cache: PagedKVCache[...],
    comp_cache: PagedKVCache[...],
    scale: Float32,
    ctx: DeviceContext,
) raises:
    """Attends every query row to its window keys and its listed compressed entries.

    Parameters:
        q_type: Query element type (inferred).
        out_type: Output element type (inferred).
        target: Compilation target string, selects the CPU or GPU path.
        window: Sliding window length in tokens; a query at ``pos`` sees
            positions ``max(0, pos - window + 1) .. pos``.
        split_k: Whether the SM100 route may split a row's keys across
            CTAs. Split partial outputs are rounded to the output dtype
            before the combine; `False` runs one CTA per row and rounds the
            output once, as the portable kernel does.
        portable_p_bf16: Diagnostic. Runs the portable GPU kernel even on
            SM100, with each attention weight rounded to bf16 before it
            multiplies its value row, the one rounding the SM100 decode
            makes and the portable kernel does not.

    Args:
        output: ``[num_rows, num_heads, head_dim]``.
        q: ``[num_rows, num_heads, head_dim]``, the last axis contiguous.
        input_row_offsets: ``[batch + 1]`` ragged row offsets of ``q``.
        comp_indices: ``[num_rows, num_comp]`` entry indices into the
            compressed leaf; ``-1`` marks an unused slot.
        attn_sink: ``[num_heads]`` per-head sink logits.
        swa_cache: This layer's sliding-window leaf.
        comp_cache: This layer's compressed leaf.
        scale: Softmax scale applied to the scores.
        ctx: Device context used to enqueue the GPU kernel.
    """
    comptime swa_t = type_of(swa_cache)
    comptime comp_t = type_of(comp_cache)
    comptime head_dim = swa_t.kv_params.head_size
    comptime assert (
        comp_t.kv_params.head_size == head_dim
    ), "window and compressed leaves must share the latent width"
    comptime assert (
        swa_t.kv_params.num_heads == 1 and comp_t.kv_params.num_heads == 1
    ), "the latent leaves hold a single shared head"
    comptime assert (
        head_dim % WARP_SIZE == 0
    ), "head_dim must be a multiple of the warp size"
    comptime assert output.flat_rank == 3 and q.flat_rank == 3
    comptime assert comp_indices.flat_rank == 2
    comptime assert (
        input_row_offsets.flat_rank == 1 and attn_sink.flat_rank == 1
    )

    var num_rows = Int(q.dim[0]())
    var num_heads = Int(q.dim[1]())
    var num_batches = Int(input_row_offsets.dim[0]()) - 1
    var num_comp = Int(comp_indices.dim[1]())
    var q_stride0 = Int(q.dynamic_stride(0))
    var q_stride1 = Int(q.dynamic_stride(1))
    var out_stride0 = Int(output.dynamic_stride(0))
    var out_stride1 = Int(output.dynamic_stride(1))
    var idx_stride0 = Int(comp_indices.dynamic_stride(0))
    var idx_stride1 = Int(comp_indices.dynamic_stride(1))
    debug_assert(
        Int(q.dynamic_stride(2)) == 1 and Int(output.dynamic_stride(2)) == 1,
        "q and output must be contiguous along head_dim",
    )
    if num_rows == 0:
        return

    var out_ptr = output.unsafe_ptr().as_unsafe_any_origin()
    var q_ptr = q.unsafe_ptr().as_unsafe_any_origin()
    var offs_ptr = input_row_offsets.unsafe_ptr().as_unsafe_any_origin()
    var idx_ptr = comp_indices.unsafe_ptr().as_unsafe_any_origin()
    var sink_ptr = attn_sink.unsafe_ptr().as_unsafe_any_origin()

    comptime q_heads = q.static_shape[1]
    comptime use_sm100 = (
        not is_cpu[target]()
        and _is_sm10x_gpu(ctx.default_device_info)
        and q_type == .bfloat16
        and out_type == .bfloat16
        and swa_t.dtype == .bfloat16
        and swa_t.kv_params == comp_t.kv_params
        and comp_t.dtype == swa_t.dtype
        and not swa_t.quantization_enabled
        and not comp_t.quantization_enabled
        and head_dim == 512
        and q_heads != UNKNOWN_VALUE
        and not portable_p_bf16
    )
    comptime if use_sm100:
        # The decode grid is rows * splits along z; bound it by the largest
        # split count so a long prefill chunk stays on the portable kernel.
        comptime max_splits = (
            ctx.default_device_info.sm_count // 2 if split_k else 1
        )
        var dense = (
            q_stride1 == head_dim
            and q_stride0 == num_heads * head_dim
            and out_stride1 == head_dim
            and out_stride0 == num_heads * head_dim
        )
        # TMA descriptors need 16-byte-aligned bases.
        var aligned = (
            Int(q_ptr) % 16 == 0
            and Int(out_ptr) % 16 == 0
            and Int(swa_cache.blocks.ptr) % 16 == 0
            and Int(comp_cache.blocks.ptr) % 16 == 0
        )
        if dense and aligned and num_rows * max_splits <= 65535:
            _latent_sparse_attention_sm100[
                num_heads=q_heads,
                head_dim=head_dim,
                window=window,
                split_k=split_k,
            ](
                out_ptr,
                q_ptr,
                offs_ptr,
                idx_ptr,
                sink_ptr,
                swa_cache,
                comp_cache,
                num_rows,
                num_batches,
                num_comp,
                idx_stride0,
                idx_stride1,
                scale,
                ctx,
            )
            return

    comptime if is_cpu[target]():
        _latent_sparse_attention_cpu[head_dim=head_dim, window=window](
            out_ptr,
            q_ptr,
            offs_ptr,
            idx_ptr,
            sink_ptr,
            swa_cache,
            comp_cache,
            num_rows,
            num_heads,
            num_batches,
            num_comp,
            q_stride0,
            q_stride1,
            out_stride0,
            out_stride1,
            idx_stride0,
            idx_stride1,
            scale,
        )
    else:
        comptime heads_per_warp = 4
        comptime warps_per_block = 8
        comptime heads_per_block = heads_per_warp * warps_per_block
        comptime kernel = _latent_sparse_attention_gpu_kernel[
            swa_t=swa_t,
            comp_t=comp_t,
            q_type=q_type,
            out_type=out_type,
            head_dim=head_dim,
            heads_per_warp=heads_per_warp,
            warps_per_block=warps_per_block,
            window=window,
            round_p=portable_p_bf16,
        ]
        ctx.enqueue_function[kernel](
            out_ptr,
            q_ptr,
            offs_ptr,
            idx_ptr,
            sink_ptr,
            swa_cache,
            comp_cache,
            Int32(num_heads),
            Int32(num_batches),
            Int32(num_comp),
            Int32(q_stride0),
            Int32(q_stride1),
            Int32(out_stride0),
            Int32(out_stride1),
            Int32(idx_stride0),
            Int32(idx_stride1),
            scale,
            grid_dim=(num_rows, ceildiv(num_heads, heads_per_block)),
            block_dim=WARP_SIZE * warps_per_block,
        )
