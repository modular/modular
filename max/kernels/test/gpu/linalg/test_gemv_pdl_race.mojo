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

    # Static (Idx) dims are REQUIRED: `row_major(M, N)` with plain Ints
    # leaves `static_shape[1] == -1`, the prefetch gate turns off, and the
    # peel is never compiled -- the exact regression the review caught.
    var a_tensor = TileTensor(a_device, row_major(Idx[M], Idx[K])).as_imm()
    var w_tensor = TileTensor(w_device, row_major(Idx[N], Idx[K])).as_imm()
    var c_tensor = TileTensor(c_device, row_major(Idx[M], Idx[N]))

    comptime assert (
        type_of(c_tensor).static_shape[1] == N
    ), "output N must be static for the prefetch gate"
    # The gated cases must actually compile the peel; the gate-off case
    # must not. Keeps this test honest if the gate expression changes.
    comptime gated = gemv_split_k_pdl_prefetch_enabled[
        PDLLevel.OVERLAP_AT_BEGINNING, N, target_is_nvidia=True
    ]()
    comptime assert gated == (
        N <= 4096
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
            if diff > tolerance:
                if first_idx < 0:
                    first_idx = i
                    first_val = result_host[i]
                errors += 1
        if errors > 0:
            print(
                "FAIL: ",
                label,
                " iter ",
                iter,
                ": ",
                errors,
                "/",
                M * N,
                " mismatches; first at ",
                first_idx,
                " got ",
                first_val,
                " expected ",
                expected,
            )
            raise Error("GEMV PDL race detected in " + label)

    print("PASS: ", label, " x", num_iters, " iters")


def main() raises:
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

        print("All GEMV PDL race tests passed!")
