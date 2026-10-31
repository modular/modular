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
"""PDL race test for the split-K GEMV's weight prefetch.

The split-K GEMV hoists iteration 0's weight loads above its PDL wait. The
activation -- the predecessor's output -- must still be consumed only after
`wait_on_dependent_grids()`, and any dependent must not observe this kernel's
stores early. Both orderings are exercised by a real PDL chain with no host
sync between launches:

1. `slow_producer` sleeps, then writes the activation and signals completion.
   It launches first with the PDL attribute; a premature GEMV would read the
   *stale* activation (pre-filled with a sentinel that produces a wrong
   result).
2. The GEMV launches next, PDL-enabled at the production level
   (`OVERLAP_AT_BEGINNING`), with static N inside the prefetch gate.
3. A PDL `consumer` then reads the GEMV output and copies it out; a missing
   release or a store visible after the dependent's wait shows up as a
   mismatch.

Covers the peel's guarded paths: `unroll_factor == 1` and `> 1`, and a K
small enough that some threads own no K columns.

At `OVERLAP_AT_BEGINNING` the GEMV releases its dependent grid on entry, before
its own wait. `run_chain_case` checks that a GEMV launched that early by another
GEMV (`producer -> GEMV -> GEMV -> consumer`) still reads the first GEMV's
finished output. With `require_overlap` it also proves the release happens on
entry: a flag-setting stub launched after the second GEMV must go resident
while the producer is still running, which cannot happen if either GEMV
releases its dependent at exit.
"""

from max.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceContext
from std.sys.intrinsics import llvm_intrinsic
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from layout import Idx, TileTensor, row_major
from linalg.gemv import gemv_split_k, gemv_split_k_pdl_prefetch_enabled
from std.math import ceildiv
from std.sys import get_defined_int
from std.time import global_perf_counter_ns


def slow_producer[
    dtype: DType,
](
    act: MutPointer[Scalar[dtype], MutAnyOrigin],
    length_dev: Int32,
    delay_dev: Int32,
):
    """Releases the dependent grid at entry, spins, then writes `act`.

    The spin runs for `delay_dev` microseconds *after* the release, so a
    PDL-launched dependent goes resident and runs its pre-wait prologue
    (the GEMV's weight prefetch) while this kernel still occupies its SM.
    The activation write lands at the very end: any dependent that reads
    it before its own PDL wait observes the stale sentinel and fails.
    """
    var length = Int(length_dev)
    launch_dependent_grids()
    # nanosleep takes NANOSECONDS and caps at ~1ms per call; `delay_dev`
    # is microseconds, so sleep 1ms chunks and decrement 1000us per call.
    var spins = Int(delay_dev)
    while spins > 0:
        llvm_intrinsic["llvm.nvvm.nanosleep", NoneType](Int32(1_000_000))
        spins -= 1000
    # Full grid stride: exactly one writer per element, no write/write
    # race in the fixture itself.
    var tid = thread_idx.x + block_idx.x * block_dim.x
    var stride = grid_dim.x * block_dim.x
    for i in range(tid, length, stride):
        act[i] = Scalar[dtype](2.0)


def handshake_producer[
    dtype: DType,
](
    act: MutPointer[Scalar[dtype], MutAnyOrigin],
    length_dev: Int32,
    flag: MutPointer[Int32, MutAnyOrigin],
    seen: MutPointer[Int32, MutAnyOrigin],
    timeout_us: Int32,
):
    """Releases the dependent grid at entry, then holds `act` back until a
    kernel launched behind the dependent chain proves it is resident.

    `flag_stub` sets `flag` before its own PDL wait, so `flag` can only
    become non-zero while this kernel is still running if every grid in
    between released its dependent on entry. The wait is bounded; `seen`
    records whether the flag arrived in time.
    """
    var length = Int(length_dev)
    launch_dependent_grids()
    var start = global_perf_counter_ns()
    var arrived = False
    while global_perf_counter_ns() - start < UInt64(timeout_us) * 1000:
        if flag.unsafe_load[volatile=True]() != 0:
            arrived = True
            break
        llvm_intrinsic["llvm.nvvm.nanosleep", NoneType](Int32(1000))
    if thread_idx.x == 0 and block_idx.x == 0:
        seen.unsafe_store[volatile=True](Int32(1) if arrived else Int32(0))
    var tid = thread_idx.x + block_idx.x * block_dim.x
    var stride = grid_dim.x * block_dim.x
    for i in range(tid, length, stride):
        act[i] = Scalar[dtype](2.0)


def flag_stub(flag: MutPointer[Int32, MutAnyOrigin]):
    """Marks itself resident before the PDL wait, then waits for its
    predecessor like any dependent."""
    if thread_idx.x == 0 and block_idx.x == 0:
        flag.unsafe_store[volatile=True](Int32(1))
    wait_on_dependent_grids()


def consumer[
    dtype: DType,
](
    src: MutPointer[Scalar[dtype], MutAnyOrigin],
    dst: MutPointer[Scalar[dtype], MutAnyOrigin],
    length_dev: Int32,
):
    """Waits for the GEMV grid, then copies its output backwards."""
    var length = Int(length_dev)
    wait_on_dependent_grids()
    var tid = thread_idx.x + block_idx.x * block_dim.x
    var stride = Int(grid_dim.x * block_dim.x)
    var inverse_tid = stride - tid
    for i in range(length - inverse_tid, -1, -stride):
        dst[i] = src[i]


def run_case[
    dtype: DType,
    M: Int,
    N: Int,
    K: Int,
    tile_m: Int,
    tile_n: Int,
    num_threads: Int,
    unroll_factor: Int,
    simd_width: Int,
](ctx: DeviceContext, label: String, num_iters: Int, delay_cycles: Int,) raises:
    """One (shape, config) case: slow producer -> PDL GEMV -> PDL consumer."""
    comptime check_bounds_m = tile_m > 1
    comptime check_bounds_n = N % tile_n != 0

    var a_host = ctx.enqueue_create_host_buffer[dtype](M * K)
    var w_host = ctx.enqueue_create_host_buffer[dtype](N * K)
    var stale_host = ctx.enqueue_create_host_buffer[dtype](M * K)
    var c_host = ctx.enqueue_create_host_buffer[dtype](M * N)
    var result_host = ctx.enqueue_create_host_buffer[dtype](M * N)

    # The real activation is all 2.0; the stale sentinel is all 1.0. The
    # weight is all 1.0, so the correct output is 2K everywhere and a
    # hoisted activation read yields 1K (or garbage) -- off by 2x.
    for i in range(M * K):
        a_host[i] = Scalar[dtype](2.0)
        stale_host[i] = Scalar[dtype](1.0)
    for i in range(N * K):
        w_host[i] = Scalar[dtype](1.0)

    var a_device = ctx.enqueue_create_buffer[dtype](M * K)
    var w_device = ctx.enqueue_create_buffer[dtype](N * K)
    var c_device = ctx.enqueue_create_buffer[dtype](M * N)
    var result_device = ctx.enqueue_create_buffer[dtype](M * N)
    ctx.enqueue_copy(w_device, w_host)

    # Static (Idx) N and K are REQUIRED: `row_major(M, N)` with plain Ints
    # leaves `static_shape[1] == -1`, the prefetch gate turns off, and the
    # peel is never compiled -- the exact regression the review caught. M is
    # dynamic, as in graph launches, which the wide-N gate requires.
    var a_tensor = TileTensor(a_device, row_major(Int64(M), Idx[K])).as_imm()
    var w_tensor = TileTensor(w_device, row_major(Idx[N], Idx[K])).as_imm()
    var c_tensor = TileTensor(c_device, row_major(Int64(M), Idx[N]))

    comptime assert (
        type_of(c_tensor).static_shape[1] == N
    ), "output N must be static for the prefetch gate"
    # The gated cases must actually compile the peel; the gate-off case
    # must not. Keeps this test honest if the gate expression changes.
    comptime gated = gemv_split_k_pdl_prefetch_enabled[
        PDLLevel.OVERLAP_AT_BEGINNING,
        -1,
        N,
        K,
        dtype,
        dtype,
        target_is_nvidia=True,
    ]()
    comptime assert gated == (
        N <= 4096 or (dtype == .bfloat16 and N >= 7168 and K >= 7168)
    ), "prefetch gate disagrees with this test's case table"

    var expected = Scalar[dtype](2 * K)

    for iter in range(num_iters):
        # Reset: activation holds the stale sentinel.
        ctx.enqueue_copy(a_device, stale_host)
        ctx.enqueue_memset[dtype](c_device, 0)
        ctx.enqueue_memset[dtype](result_device, 0)

        # 1. Producer: sleeps, then publishes the real activation.
        comptime producer = slow_producer[dtype]
        ctx.enqueue_function[producer](
            a_device,
            Int32(M * K),
            Int32(delay_cycles),
            grid_dim=4,
            block_dim=64,
            attributes=pdl_launch_attributes(PDLLevel.OVERLAP_AT_END),
        )

        # 2. GEMV, PDL-on, static shapes: the prefetch path is compiled in.
        comptime kernel = gemv_split_k[
            dtype,
            dtype,
            dtype,
            type_of(c_tensor).LayoutType,
            type_of(a_tensor).LayoutType,
            type_of(w_tensor).LayoutType,
            type_of(c_tensor).Engine,
            type_of(a_tensor).Engine,
            type_of(w_tensor).Engine,
            simd_width=simd_width,
            tile_m=tile_m,
            tile_n=tile_n,
            num_threads=num_threads,
            unroll_factor=unroll_factor,
            check_bounds_m=check_bounds_m,
            check_bounds_n=check_bounds_n,
            pdl_level=PDLLevel.OVERLAP_AT_BEGINNING,
        ]
        ctx.enqueue_function[kernel](
            c_tensor,
            a_tensor,
            w_tensor,
            Int32(M),
            Int32(N),
            Int32(K),
            grid_dim=(ceildiv(M, tile_m), ceildiv(N, tile_n)),
            block_dim=num_threads,
            attributes=pdl_launch_attributes(PDLLevel.OVERLAP_AT_BEGINNING),
        )

        # 3. Consumer: PDL-waiting dependent of the GEMV.
        comptime consumer_kernel = consumer[dtype]
        var cta = 256
        ctx.enqueue_function[consumer_kernel](
            c_device,
            result_device,
            Int32(M * N),
            grid_dim=ceildiv(M * N, cta),
            block_dim=cta,
            attributes=pdl_launch_attributes(PDLLevel.OVERLAP_AT_END),
        )

        ctx.synchronize()
        ctx.enqueue_copy(result_host, result_device)
        ctx.synchronize()

        var errors = 0
        var first_idx = -1
        var first_val = Scalar[dtype](0)
        var tolerance = expected * Scalar[dtype](0.02)
        for i in range(M * N):
            var diff = result_host[i] - expected
            if diff < 0:
                diff = -diff
            # `not (diff <= tolerance)` also rejects NaN.
            if not (diff <= tolerance):
                if first_idx < 0:
                    first_idx = i
                    first_val = result_host[i]
                errors += 1
        if errors > 0:
            print(
                t"FAIL:  {label}  iter  {iter} :  {errors} / {M * N} "
                t" mismatches; first at  {first_idx}  got  {first_val} "
                t" expected  {expected}"
            )
            raise Error("GEMV PDL race detected in " + label)

    print(t"PASS:  {label}  x {num_iters}  iters")


def run_chain_case[
    dtype: DType,
    N1: Int,
    K1: Int,
    N2: Int,
    tile_n1: Int,
    tile_n2: Int,
    num_threads: Int,
    simd_width: Int,
    require_overlap: Bool = False,
](ctx: DeviceContext, label: String, num_iters: Int, delay_us: Int) raises:
    """Slow producer -> GEMV1 -> GEMV2 -> consumer, all PDL-chained.

    GEMV1 releases GEMV2 on entry (`OVERLAP_AT_BEGINNING`), so GEMV2 is
    resident and past its weight prefetch while GEMV1 and the producer are
    still running. GEMV2 consumes GEMV1's output (zeros until GEMV1 finishes),
    so a read before its own wait yields a wrong sum.

    The result check alone also passes if the GEMVs run back to back. With
    `require_overlap` the producer is `handshake_producer` and a `flag_stub`
    sits between GEMV2 and the consumer: the stub can only go resident while
    the producer runs if GEMV1 and GEMV2 both released on entry, and the
    producer reports whether it saw the stub within `delay_us`. Every block of
    GEMV1 must be resident for its release to fire, so use shapes that fit in
    one wave.
    """
    comptime M = 1
    comptime K2 = N1

    var a_host = ctx.enqueue_create_host_buffer[dtype](M * K1)
    var stale_host = ctx.enqueue_create_host_buffer[dtype](M * K1)
    var w1_host = ctx.enqueue_create_host_buffer[dtype](N1 * K1)
    var w2_host = ctx.enqueue_create_host_buffer[dtype](N2 * K2)
    var result_host = ctx.enqueue_create_host_buffer[dtype](M * N2)
    for i in range(M * K1):
        a_host[i] = Scalar[dtype](2.0)
        stale_host[i] = Scalar[dtype](1.0)
    for i in range(N1 * K1):
        w1_host[i] = Scalar[dtype](1.0)
    for i in range(N2 * K2):
        w2_host[i] = Scalar[dtype](1.0)

    var a_device = ctx.enqueue_create_buffer[dtype](M * K1)
    var w1_device = ctx.enqueue_create_buffer[dtype](N1 * K1)
    var w2_device = ctx.enqueue_create_buffer[dtype](N2 * K2)
    var c1_device = ctx.enqueue_create_buffer[dtype](M * N1)
    var c2_device = ctx.enqueue_create_buffer[dtype](M * N2)
    var result_device = ctx.enqueue_create_buffer[dtype](M * N2)
    var flag_device = ctx.enqueue_create_buffer[DType.int32](1)
    var seen_device = ctx.enqueue_create_buffer[DType.int32](1)
    var seen_host = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(w1_device, w1_host)
    ctx.enqueue_copy(w2_device, w2_host)

    var a_tensor = TileTensor(a_device, row_major(Int64(M), Idx[K1])).as_imm()
    var w1_tensor = TileTensor(w1_device, row_major(Idx[N1], Idx[K1])).as_imm()
    var c1_tensor = TileTensor(c1_device, row_major(Int64(M), Idx[N1]))
    var c1_in = TileTensor(c1_device, row_major(Int64(M), Idx[K2])).as_imm()
    var w2_tensor = TileTensor(w2_device, row_major(Idx[N2], Idx[K2])).as_imm()
    var c2_tensor = TileTensor(c2_device, row_major(Int64(M), Idx[N2]))

    # Both GEMVs take the peeled prefetch path at these static shapes.
    comptime assert (
        gemv_split_k_pdl_prefetch_enabled[
            PDLLevel.OVERLAP_AT_BEGINNING,
            -1,
            N1,
            K1,
            dtype,
            dtype,
            target_is_nvidia=True,
        ]()
        and gemv_split_k_pdl_prefetch_enabled[
            PDLLevel.OVERLAP_AT_BEGINNING,
            -1,
            N2,
            K2,
            dtype,
            dtype,
            target_is_nvidia=True,
        ]()
    ), "chain case must exercise the prefetch peel on both GEMVs"

    # c1[j] = sum_K1(2 * 1) = 2 * K1; c2[j] = sum_N1(c1) = 2 * K1 * N1.
    var expected = Scalar[dtype](2 * K1 * N1)

    var overlap_hits = 0
    for iter in range(num_iters):
        ctx.enqueue_copy(a_device, stale_host)
        ctx.enqueue_memset[dtype](c1_device, 0)
        ctx.enqueue_memset[dtype](c2_device, 0)
        ctx.enqueue_memset[dtype](result_device, 0)
        ctx.enqueue_memset[DType.int32](flag_device, 0)
        ctx.enqueue_memset[DType.int32](seen_device, 0)

        comptime if require_overlap:
            comptime hs_producer = handshake_producer[dtype]
            ctx.enqueue_function[hs_producer](
                a_device,
                Int32(M * K1),
                flag_device,
                seen_device,
                Int32(delay_us),
                grid_dim=4,
                block_dim=64,
                attributes=pdl_launch_attributes(PDLLevel.OVERLAP_AT_END),
            )
        else:
            comptime producer = slow_producer[dtype]
            ctx.enqueue_function[producer](
                a_device,
                Int32(M * K1),
                Int32(delay_us),
                grid_dim=4,
                block_dim=64,
                attributes=pdl_launch_attributes(PDLLevel.OVERLAP_AT_END),
            )

        comptime kernel1 = gemv_split_k[
            dtype,
            dtype,
            dtype,
            type_of(c1_tensor).LayoutType,
            type_of(a_tensor).LayoutType,
            type_of(w1_tensor).LayoutType,
            type_of(c1_tensor).Engine,
            type_of(a_tensor).Engine,
            type_of(w1_tensor).Engine,
            simd_width=simd_width,
            tile_m=1,
            tile_n=tile_n1,
            num_threads=num_threads,
            unroll_factor=1,
            check_bounds_m=False,
            check_bounds_n=N1 % tile_n1 != 0,
            pdl_level=PDLLevel.OVERLAP_AT_BEGINNING,
        ]
        ctx.enqueue_function[kernel1](
            c1_tensor,
            a_tensor,
            w1_tensor,
            Int32(M),
            Int32(N1),
            Int32(K1),
            grid_dim=(M, ceildiv(N1, tile_n1)),
            block_dim=num_threads,
            attributes=pdl_launch_attributes(PDLLevel.OVERLAP_AT_BEGINNING),
        )

        comptime kernel2 = gemv_split_k[
            dtype,
            dtype,
            dtype,
            type_of(c2_tensor).LayoutType,
            type_of(c1_in).LayoutType,
            type_of(w2_tensor).LayoutType,
            type_of(c2_tensor).Engine,
            type_of(c1_in).Engine,
            type_of(w2_tensor).Engine,
            simd_width=simd_width,
            tile_m=1,
            tile_n=tile_n2,
            num_threads=num_threads,
            unroll_factor=1,
            check_bounds_m=False,
            check_bounds_n=N2 % tile_n2 != 0,
            pdl_level=PDLLevel.OVERLAP_AT_BEGINNING,
        ]
        ctx.enqueue_function[kernel2](
            c2_tensor,
            c1_in,
            w2_tensor,
            Int32(M),
            Int32(N2),
            Int32(K2),
            grid_dim=(M, ceildiv(N2, tile_n2)),
            block_dim=num_threads,
            attributes=pdl_launch_attributes(PDLLevel.OVERLAP_AT_BEGINNING),
        )

        comptime if require_overlap:
            ctx.enqueue_function[flag_stub](
                flag_device,
                grid_dim=1,
                block_dim=32,
                attributes=pdl_launch_attributes(PDLLevel.OVERLAP_AT_END),
            )

        comptime consumer_kernel = consumer[dtype]
        var cta = 256
        ctx.enqueue_function[consumer_kernel](
            c2_device,
            result_device,
            Int32(M * N2),
            grid_dim=ceildiv(M * N2, cta),
            block_dim=cta,
            attributes=pdl_launch_attributes(PDLLevel.OVERLAP_AT_END),
        )

        ctx.synchronize()
        ctx.enqueue_copy(result_host, result_device)
        ctx.enqueue_copy(seen_host, seen_device)
        ctx.synchronize()

        # Iteration 0 absorbs the lazy load of the stub and both GEMV
        # instantiations, which can stall the host past any timeout.
        if iter > 0 and seen_host[0] == 1:
            overlap_hits += 1

        var errors = 0
        var tolerance = expected * Scalar[dtype](0.02)
        for i in range(M * N2):
            var diff = result_host[i] - expected
            if diff < 0:
                diff = -diff
            # `not (diff <= tolerance)` also rejects NaN.
            if not (diff <= tolerance):
                errors += 1
        if errors > 0:
            raise Error(
                "GEMV->GEMV PDL race detected in "
                + label
                + " iter "
                + String(iter)
                + ": "
                + String(errors)
                + "/"
                + String(M * N2)
                + " mismatches, first value "
                + String(result_host[0])
                + " expected "
                + String(expected)
            )

    # A release at exit can never produce a hit, so a regression is 0 hits;
    # a single miss is only a producer that was time-sliced out of its
    # bounded wait.
    comptime if require_overlap:
        if num_iters > 1 and overlap_hits == 0:
            raise Error(
                "GEMV->GEMV entry release did not overlap in "
                + label
                + ": the stub behind both GEMVs never went resident"
                " while the producer was running in any of "
                + String(num_iters - 1)
                + " counted iters"
            )

    print("PASS: ", label, " x", num_iters, " iters")


def main() raises:
    # The wide-N peel is gated on a dynamic M: a static M regresses it.
    comptime assert gemv_split_k_pdl_prefetch_enabled[
        PDLLevel.OVERLAP_AT_BEGINNING,
        -1,
        8192,
        7168,
        .bfloat16,
        .bfloat16,
        target_is_nvidia=True,
    ]()
    comptime assert not gemv_split_k_pdl_prefetch_enabled[
        PDLLevel.OVERLAP_AT_BEGINNING,
        1,
        8192,
        7168,
        .bfloat16,
        .bfloat16,
        target_is_nvidia=True,
    ]()

    var iters = get_defined_int["ITERS", 30]()
    # The producer spins for DELAY_US microseconds *after* releasing the
    # dependent grid: the GEMV is resident and past its pre-wait prefetch
    # long before the activation write lands, so only a correct PDL wait
    # can produce the expected value. 5ms dwarfs every launch/residency
    # latency without making the test slow (30 iters x ~5ms = ~150ms/case).
    var delay_cycles = get_defined_int["DELAY_US", 5000]()

    with DeviceContext() as ctx:
        # Gated static N; production decode dispatch shapes.
        # unroll_factor == 1: simple-loop peel path.
        run_case[
            .bfloat16,
            M=1,
            N=256,
            K=7168,
            tile_m=1,
            tile_n=4,
            num_threads=256,
            unroll_factor=1,
            simd_width=8,
        ](ctx, "N256_K7168_unroll1", iters, delay_cycles)

        # unroll_factor > 1: the clamped-iteration path.
        run_case[
            .bfloat16,
            M=1,
            N=576,
            K=7168,
            tile_m=1,
            tile_n=4,
            num_threads=256,
            unroll_factor=2,
            simd_width=8,
        ](ctx, "N576_K7168_unroll2", iters, delay_cycles)

        # Some threads own no K columns (K < num_threads * simd_width):
        # exercises the `tid * simd_width < k` guards in peel and clamp.
        run_case[
            .bfloat16,
            M=1,
            N=1024,
            K=512,
            tile_m=1,
            tile_n=2,
            num_threads=256,
            unroll_factor=1,
            simd_width=8,
        ](ctx, "N1024_K512_shortk", iters, delay_cycles)

        # Wide-N, gate off: wait-first order must stay correct too.
        run_case[
            .bfloat16,
            M=1,
            N=16384,
            K=2048,
            tile_m=1,
            tile_n=4,
            num_threads=256,
            unroll_factor=2,
            simd_width=8,
        ](ctx, "N16384_K2048_gate_off", iters, delay_cycles)

        # Wide-N, deep-K: the prefetch gate is on above N=4096 here, and
        # the peeled loop covers many K-iterations.
        run_case[
            .bfloat16,
            M=1,
            N=8192,
            K=7168,
            tile_m=1,
            tile_n=2,
            num_threads=128,
            unroll_factor=1,
            simd_width=8,
        ](ctx, "N8192_K7168_deepk_wide", iters, delay_cycles)

        # Wide-N, shallow-K: gate off, wait-first order, early release.
        run_case[
            .bfloat16,
            M=1,
            N=8192,
            K=4096,
            tile_m=1,
            tile_n=2,
            num_threads=128,
            unroll_factor=1,
            simd_width=8,
        ](ctx, "N8192_K4096_shallowk_gate_off", iters, delay_cycles)

        # GEMV -> GEMV: the second GEMV launches while the first still runs.
        run_chain_case[
            .bfloat16,
            N1=7168,
            K1=7168,
            N2=1024,
            tile_n1=2,
            tile_n2=2,
            num_threads=128,
            simd_width=8,
        ](ctx, "chain_N7168_K7168_then_N1024", iters, delay_cycles)

        # The same chain at a size where every block is resident at once,
        # with a stub behind it that proves the entry release overlapped.
        run_chain_case[
            .bfloat16,
            N1=1024,
            K1=7168,
            N2=512,
            tile_n1=2,
            tile_n2=2,
            num_threads=128,
            simd_width=8,
            require_overlap=True,
        ](ctx, "chain_overlap_N1024_K7168_then_N512", iters, 200_000)

        print("All GEMV PDL race tests passed!")
