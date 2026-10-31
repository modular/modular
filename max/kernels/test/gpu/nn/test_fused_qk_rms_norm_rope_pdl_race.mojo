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
"""PDL race test for `fused_qk_rms_norm_rope_ragged_paged`.

The kernel releases its dependent grids right after its own
`wait_on_dependent_grids()`. Both orderings around that release are exercised
by a real PDL chain with no host sync between launches:

1. `slow_producer` releases its dependents at entry, sleeps, then publishes the
   Q projection and the K cache (the data the fused kernel reads). Both start
   as zeros, so a fused kernel that read them before its wait would produce
   outputs that differ from the reference.
2. The fused kernel launches next at its production PDL level.
3. A PDL `consumer` then waits and copies out the Q output and the K cache. A
   dependent that observed the fused kernel's stores before they were visible
   would also show up as a mismatch.

The reference is the same op run with plain stream ordering on the final
inputs, so the comparison is bit exact.
"""

from std.math import ceildiv
from std.sys import get_defined_int
from std.sys.intrinsics import llvm_intrinsic

from max.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Idx, TileTensor, row_major
from layout._fillers import random
from nn.kv_cache import fused_qk_rms_norm_rope_ragged_paged

comptime dtype = DType.bfloat16
comptime head_size = 128
comptime rope_dim = 128
comptime num_q_heads = 8
comptime num_k_heads = 2
comptime page_size = 128
comptime max_seq_len = 512
comptime batch_size = 4
comptime epsilon = Float32(1e-6)

comptime kv_params = KVCacheStaticParams(
    num_heads=num_k_heads, head_size=head_size
)
comptime Collection = PagedKVCacheCollection[
    dtype,
    kv_params,
    page_size,
    MutAnyOrigin,
    ImmutAnyOrigin,
    ImmutAnyOrigin,
    MutAnyOrigin,
]


def slow_producer(
    q_dst: MutPointer[Scalar[dtype], MutAnyOrigin],
    q_src: MutPointer[Scalar[dtype], MutAnyOrigin],
    q_len: Int32,
    kv_dst: MutPointer[Scalar[dtype], MutAnyOrigin],
    kv_src: MutPointer[Scalar[dtype], MutAnyOrigin],
    kv_len: Int32,
    delay_us: Int32,
):
    """Releases the dependent grid at entry, spins, then publishes the inputs.

    The spin runs after the release, so the fused kernel is resident and past
    any pre-wait work before the writes land.
    """
    launch_dependent_grids()
    # nanosleep takes nanoseconds and caps at ~1ms per call.
    var spins = Int(delay_us)
    while spins > 0:
        llvm_intrinsic["llvm.nvvm.nanosleep", NoneType](Int32(1_000_000))
        spins -= 1000
    var tid = thread_idx.x + block_idx.x * block_dim.x
    var stride = grid_dim.x * block_dim.x
    for i in range(tid, Int(q_len), stride):
        q_dst[i] = q_src[i]
    for i in range(tid, Int(kv_len), stride):
        kv_dst[i] = kv_src[i]


def consumer(
    q_src: MutPointer[Scalar[dtype], MutAnyOrigin],
    q_dst: MutPointer[Scalar[dtype], MutAnyOrigin],
    q_len: Int32,
    kv_src: MutPointer[Scalar[dtype], MutAnyOrigin],
    kv_dst: MutPointer[Scalar[dtype], MutAnyOrigin],
    kv_len: Int32,
):
    """Waits for the fused kernel, then copies its outputs."""
    wait_on_dependent_grids()
    var tid = thread_idx.x + block_idx.x * block_dim.x
    var stride = grid_dim.x * block_dim.x
    for i in range(tid, Int(q_len), stride):
        q_dst[i] = q_src[i]
    for i in range(tid, Int(kv_len), stride):
        kv_dst[i] = kv_src[i]


def main() raises:
    with DeviceContext() as ctx:
        run(ctx)


def run(ctx: DeviceContext) raises:
    var iters = get_defined_int["ITERS", 20]()
    var delay_us = get_defined_int["DELAY_US", 3000]()

    var cache_lens = [5, 130, 33, 257]
    var pages_per_seq = 3
    var lut_cols = ((pages_per_seq + 7) // 8) * 8 + 16
    var num_blocks = batch_size * pages_per_seq
    var q_elems = batch_size * num_q_heads * head_size
    var kv_elems = num_blocks * 2 * 1 * page_size * num_k_heads * head_size

    var q_real = ctx.enqueue_create_buffer[dtype](q_elems)
    var q_in = ctx.enqueue_create_buffer[dtype](q_elems)
    var q_ref_out = ctx.enqueue_create_buffer[dtype](q_elems)
    var q_out = ctx.enqueue_create_buffer[dtype](q_elems)
    var q_copy = ctx.enqueue_create_buffer[dtype](q_elems)
    var kv_real = ctx.enqueue_create_buffer[dtype](kv_elems)
    var kv_ref = ctx.enqueue_create_buffer[dtype](kv_elems)
    var kv_work = ctx.enqueue_create_buffer[dtype](kv_elems)
    var kv_copy = ctx.enqueue_create_buffer[dtype](kv_elems)
    var gamma_q = ctx.enqueue_create_buffer[dtype](head_size)
    var gamma_k = ctx.enqueue_create_buffer[dtype](head_size)
    var freqs = ctx.enqueue_create_buffer[dtype](max_seq_len * rope_dim)
    var row_offsets = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    var cache_lengths = ctx.enqueue_create_buffer[.uint32](batch_size)
    var lut = ctx.enqueue_create_buffer[.uint32](batch_size * lut_cols)

    with q_real.map_to_host() as h:
        random(
            TileTensor(
                h, row_major(batch_size, Idx[num_q_heads], Idx[head_size])
            )
        )
    with kv_real.map_to_host() as h:
        random(TileTensor(h, row_major(kv_elems)))
    with gamma_q.map_to_host() as h:
        random(TileTensor(h, row_major(Idx[head_size])))
    with gamma_k.map_to_host() as h:
        random(TileTensor(h, row_major(Idx[head_size])))
    with freqs.map_to_host() as h:
        random(TileTensor(h, row_major(Idx[max_seq_len], Idx[rope_dim])))
    with row_offsets.map_to_host() as h:
        for i in range(batch_size + 1):
            h[i] = UInt32(i)
    with cache_lengths.map_to_host() as h:
        for i in range(batch_size):
            h[i] = UInt32(cache_lens[i])
    with lut.map_to_host() as h:
        for i in range(batch_size * lut_cols):
            h[i] = 0
        for bs in range(batch_size):
            for p in range(pages_per_seq):
                h[bs * lut_cols + p] = UInt32(bs * pages_per_seq + p)
    ctx.enqueue_copy(kv_ref, kv_real)
    ctx.synchronize()

    var kv_layout = row_major(
        Int64(num_blocks),
        Idx[2],
        Int64(1),
        Idx[page_size],
        Idx[num_k_heads],
        Idx[head_size],
    )
    var cache_lengths_tt = TileTensor(
        cache_lengths, row_major(Int64(batch_size))
    ).as_imm()
    var lut_tt = TileTensor(
        lut, row_major(Int64(batch_size), Int64(lut_cols))
    ).as_imm()
    var ref_collection = Collection(
        rebind[Collection.blocks_tt_type](
            TileTensor(
                kv_ref, row_major(kv_layout.shape_coord())
            ).as_unsafe_any_origin()
        ),
        cache_lengths_tt.as_unsafe_any_origin(),
        lut_tt.as_unsafe_any_origin(),
        UInt32(1),
        UInt32(257),
    )
    var work_collection = Collection(
        rebind[Collection.blocks_tt_type](
            TileTensor(
                kv_work, row_major(kv_layout.shape_coord())
            ).as_unsafe_any_origin()
        ),
        cache_lengths_tt.as_unsafe_any_origin(),
        lut_tt.as_unsafe_any_origin(),
        UInt32(1),
        UInt32(257),
    )

    var q_real_tt = TileTensor(
        q_real, row_major(batch_size, Idx[num_q_heads], Idx[head_size])
    )
    var q_in_tt = TileTensor(
        q_in, row_major(batch_size, Idx[num_q_heads], Idx[head_size])
    )
    var q_ref_out_tt = TileTensor(
        q_ref_out, row_major(batch_size, Idx[num_q_heads], Idx[head_size])
    )
    var q_out_tt = TileTensor(
        q_out, row_major(batch_size, Idx[num_q_heads], Idx[head_size])
    )
    var gamma_q_tt = TileTensor(gamma_q, row_major[head_size]())
    var gamma_k_tt = TileTensor(gamma_k, row_major[head_size]())
    var freqs_tt = TileTensor(freqs, row_major(Idx[max_seq_len], Idx[rope_dim]))
    var row_offsets_tt = TileTensor(
        row_offsets, row_major(batch_size + 1)
    ).as_imm()

    @inline(.always)
    def ref_q_fn[
        width: Int, alignment: Int
    ](token: Int, head: Int, col: Int) {var q_real_tt} -> SIMD[dtype, width]:
        return q_real_tt.load[width=width]((token, head, col))

    @inline(.always)
    def work_q_fn[
        width: Int, alignment: Int
    ](token: Int, head: Int, col: Int) {var q_in_tt} -> SIMD[dtype, width]:
        return q_in_tt.load[width=width]((token, head, col))

    # Reference: plain stream order on the final inputs.
    fused_qk_rms_norm_rope_ragged_paged[
        target="gpu", multiply_before_cast=True, interleaved=False
    ](
        ref_collection,
        ref_q_fn,
        gamma_q_tt,
        gamma_k_tt,
        freqs_tt,
        epsilon,
        Scalar[dtype](0.0),
        UInt32(0),
        row_offsets_tt,
        q_ref_out_tt,
        ctx,
    )
    ctx.synchronize()

    var q_ref_host = ctx.enqueue_create_host_buffer[dtype](q_elems)
    var kv_ref_host = ctx.enqueue_create_host_buffer[dtype](kv_elems)
    var q_copy_host = ctx.enqueue_create_host_buffer[dtype](q_elems)
    var kv_copy_host = ctx.enqueue_create_host_buffer[dtype](kv_elems)
    ctx.enqueue_copy(q_ref_host, q_ref_out)
    ctx.enqueue_copy(kv_ref_host, kv_ref)
    ctx.synchronize()

    # The test only detects a premature read if the outputs depend on the
    # inputs, so reject a degenerate reference.
    var nonzero = 0
    for i in range(q_elems):
        if q_ref_host[i] != 0:
            nonzero += 1
    if nonzero == 0:
        raise Error("reference Q output is all zero")

    for it in range(iters):
        # Stale inputs: zeros the producer overwrites after its delay.
        ctx.enqueue_memset[dtype](q_in, 0)
        ctx.enqueue_memset[dtype](kv_work, 0)
        ctx.enqueue_memset[dtype](q_copy, 0)
        ctx.enqueue_memset[dtype](kv_copy, 0)

        ctx.enqueue_function[slow_producer](
            q_in,
            q_real,
            Int32(q_elems),
            kv_work,
            kv_real,
            Int32(kv_elems),
            Int32(delay_us),
            grid_dim=64,
            block_dim=256,
            attributes=pdl_launch_attributes(PDLLevel.OVERLAP_AT_END),
        )
        fused_qk_rms_norm_rope_ragged_paged[
            target="gpu", multiply_before_cast=True, interleaved=False
        ](
            work_collection,
            work_q_fn,
            gamma_q_tt,
            gamma_k_tt,
            freqs_tt,
            epsilon,
            Scalar[dtype](0.0),
            UInt32(0),
            row_offsets_tt,
            q_out_tt,
            ctx,
        )
        ctx.enqueue_function[consumer](
            q_out,
            q_copy,
            Int32(q_elems),
            kv_work,
            kv_copy,
            Int32(kv_elems),
            grid_dim=64,
            block_dim=256,
            attributes=pdl_launch_attributes(PDLLevel.OVERLAP_AT_END),
        )
        ctx.enqueue_copy(q_copy_host, q_copy)
        ctx.enqueue_copy(kv_copy_host, kv_copy)
        ctx.synchronize()

        var q_bad = 0
        for i in range(q_elems):
            if q_copy_host[i] != q_ref_host[i]:
                q_bad += 1
        var kv_bad = 0
        for i in range(kv_elems):
            if kv_copy_host[i] != kv_ref_host[i]:
                kv_bad += 1
        if q_bad != 0 or kv_bad != 0:
            raise Error(
                String(
                    "PDL race: iter ",
                    it,
                    ": ",
                    q_bad,
                    "/",
                    q_elems,
                    " Q and ",
                    kv_bad,
                    "/",
                    kv_elems,
                    " K/V elements differ from the stream-ordered reference",
                )
            )

    print("PASS: fused_qk_rms_norm_rope PDL chain x", iters, " iters")

    _ = q_real^
    _ = q_in^
    _ = q_ref_out^
    _ = q_out^
    _ = q_copy^
    _ = kv_real^
    _ = kv_ref^
    _ = kv_work^
    _ = kv_copy^
    _ = gamma_q^
    _ = gamma_k^
    _ = freqs^
    _ = row_offsets^
    _ = cache_lengths^
    _ = lut^
