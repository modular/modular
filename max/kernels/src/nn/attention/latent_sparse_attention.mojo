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

One block per (query row, group of heads); each warp owns ``heads_per_warp``
heads and reads every key row once as a lane-strided vector, so a key costs
one vector load per warp plus a warp reduction per head. This is the
portable form of the kernel; it does not use tensor cores.
"""

from std.math import ceildiv, exp
from std.memory import Layout as AllocLayout, alloc, dealloc
from std.utils.numerics import min_or_neg_inf

from max.gpu import WARP_SIZE, block_idx, lane_id, warp_id
from max.gpu.host import DeviceContext
from max.gpu.host.info import is_cpu
import max.gpu.primitives.warp as warp

from kv_cache.types import KVCacheT
from layout import LayoutTensor


@inline(.always)
def _batch_of_row(
    row: Int,
    row_offsets: Pointer[UInt32, MutAnyOrigin],
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
](
    out_ptr: Pointer[Scalar[out_type], MutAnyOrigin],
    q_ptr: Pointer[Scalar[q_type], MutAnyOrigin],
    row_offsets: Pointer[UInt32, MutAnyOrigin],
    comp_indices: Pointer[Int32, MutAnyOrigin],
    attn_sink: Pointer[Float32, MutAnyOrigin],
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
    var t = Int(block_idx.x)
    var lane = Int(lane_id())
    var h0 = (
        Int(block_idx.y) * warps_per_block + Int(warp_id())
    ) * heads_per_warp
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
        _attend_key(k, q, m, l, acc)

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
        _attend_key(k, q, m, l, acc)

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
    q_ptr: Pointer[Scalar[q_type], MutAnyOrigin],
    row_offsets: Pointer[UInt32, MutAnyOrigin],
    comp_indices: Pointer[Int32, MutAnyOrigin],
    attn_sink: Pointer[Float32, MutAnyOrigin],
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


def latent_sparse_attention_ragged_paged[
    swa_t: KVCacheT,
    comp_t: KVCacheT,
    q_type: DType,
    out_type: DType,
    //,
    target: StaticString,
    window: Int,
](
    output: LayoutTensor[mut=True, out_type, address_space=.GENERIC, ...],
    q: LayoutTensor[mut=False, q_type, address_space=.GENERIC, ...],
    input_row_offsets: LayoutTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    comp_indices: LayoutTensor[mut=False, .int32, address_space=.GENERIC, ...],
    attn_sink: LayoutTensor[mut=False, .float32, address_space=.GENERIC, ...],
    swa_cache: swa_t,
    comp_cache: comp_t,
    scale: Float32,
    ctx: DeviceContext,
) raises:
    """Attends every query row to its window keys and its listed compressed entries.

    Parameters:
        swa_t: The sliding-window leaf's cache type (inferred); paged by
            token position, one latent head.
        comp_t: The compressed leaf's cache type (inferred); paged by entry,
            one latent head with the same head size.
        q_type: Query element type (inferred).
        out_type: Output element type (inferred).
        target: Compilation target string, selects the CPU or GPU path.
        window: Sliding window length in tokens; a query at ``pos`` sees
            positions ``max(0, pos - window + 1) .. pos``.

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
    comptime assert output.layout.rank() == 3 and q.layout.rank() == 3
    comptime assert comp_indices.layout.rank() == 2
    comptime assert (
        input_row_offsets.layout.rank() == 1 and attn_sink.layout.rank() == 1
    )

    var num_rows = q.dim(0)
    var num_heads = q.dim(1)
    var num_batches = input_row_offsets.dim(0) - 1
    var num_comp = comp_indices.dim(1)
    var q_stride0 = Int(q.runtime_layout.stride.value[0])
    var q_stride1 = Int(q.runtime_layout.stride.value[1])
    var out_stride0 = Int(output.runtime_layout.stride.value[0])
    var out_stride1 = Int(output.runtime_layout.stride.value[1])
    var idx_stride0 = Int(comp_indices.runtime_layout.stride.value[0])
    var idx_stride1 = Int(comp_indices.runtime_layout.stride.value[1])
    debug_assert(
        Int(q.runtime_layout.stride.value[2]) == 1
        and Int(output.runtime_layout.stride.value[2]) == 1,
        "q and output must be contiguous along head_dim",
    )
    if num_rows == 0:
        return

    var out_ptr = rebind[Pointer[Scalar[out_type], MutAnyOrigin]](output.ptr)
    var q_ptr = rebind[Pointer[Scalar[q_type], MutAnyOrigin]](q.ptr)
    var offs_ptr = rebind[Pointer[UInt32, MutAnyOrigin]](input_row_offsets.ptr)
    var idx_ptr = rebind[Pointer[Int32, MutAnyOrigin]](comp_indices.ptr)
    var sink_ptr = rebind[Pointer[Float32, MutAnyOrigin]](attn_sink.ptr)

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
