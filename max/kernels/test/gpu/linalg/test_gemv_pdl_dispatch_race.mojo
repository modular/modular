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
"""PDL race test for the SM100 matmul dispatcher's split-K GEMV launches.

The split-K GEMV releases its dependent grid on entry at
`PDLLevel.OVERLAP_AT_BEGINNING`, so every split-K launch the SM100 dispatcher
makes must compile its own wait or it can read a predecessor's unfinished
output. Each case drives `matmul_dispatch_sm100` with a real PDL chain and no
host sync between launches:

1. `slow_producer` releases its dependent on entry, sleeps, then publishes the
   activation. A GEMV that does not wait reads the stale sentinel.
2. The dispatcher launches the matmul, or two chained matmuls, at the
   production PDL level.
3. A PDL `consumer` copies out the final output; a missing release or a
   store visible after the dependent's wait shows up as a mismatch.

Cases: the small-M/N tuning table's split-K GEMV (M=3, N=384, K=7168), a
chain of two M=1 decode GEMVs through `dispatch_gemv`.
"""

from max.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from layout import Idx, TileTensor, row_major
from linalg.matmul.gpu.sm100_structured.default.dispatch import (
    matmul_dispatch_sm100,
)
from linalg.gemv import GEMVAlgorithm
from linalg.matmul.gpu.sm100_structured.default.tuning_configs import (
    _get_tuning_list_small_MN_gemms_bf16,
)
from std.math import ceildiv
from std.sys import get_defined_int
from std.sys.intrinsics import llvm_intrinsic

comptime dtype = DType.bfloat16


def slow_producer(
    act: MutPointer[Scalar[dtype], MutAnyOrigin],
    length_dev: Int32,
    delay_dev: Int32,
):
    """Releases the dependent at entry, spins, then writes `act` = 2."""
    var length = Int(length_dev)
    launch_dependent_grids()
    # nanosleep takes nanoseconds and caps at ~1ms per call.
    var spins = Int(delay_dev)
    while spins > 0:
        llvm_intrinsic["llvm.nvvm.nanosleep", NoneType](Int32(1_000_000))
        spins -= 1000
    var tid = thread_idx.x + block_idx.x * block_dim.x
    var stride = grid_dim.x * block_dim.x
    for i in range(tid, length, stride):
        act[i] = Scalar[dtype](2.0)


def consumer(
    src: MutPointer[Scalar[dtype], MutAnyOrigin],
    dst: MutPointer[Scalar[dtype], MutAnyOrigin],
    length_dev: Int32,
):
    """Waits for the matmul grid, then copies its output."""
    var length = Int(length_dev)
    wait_on_dependent_grids()
    var i = Int(thread_idx.x + block_idx.x * block_dim.x)
    if i < length:
        dst[i] = src[i]


def check_routes_to_small_mn_split_k(M: Int, N: Int, K: Int) raises:
    """Fails unless the tuning table sends (M, N, K) to the split-K GEMV.

    The small-M/N case is only worth running while the dispatcher routes it
    through `_small_MN_gemms_impl`; a retuned table would otherwise leave the
    test passing without covering that path.
    """
    for config in _get_tuning_list_small_MN_gemms_bf16():
        if (
            config.N == N
            and config.K == K
            and M >= config.M
            and M < config.M_end
            and config.kernel_kind == GEMVAlgorithm.GemvSplitK
        ):
            return
    raise Error("small-M/N table no longer routes this shape to split-K GEMV")


def run_case[
    M: Int, N1: Int, K1: Int, N2: Int, chained: Bool
](ctx: DeviceContext, label: String, num_iters: Int, delay_us: Int) raises:
    """Slow producer -> matmul (-> matmul) -> consumer through the dispatcher.

    The weights are all ones and the activation is 2, so the first matmul
    yields `2 * K1` per element and the second `2 * K1 * N1`. Static N and K
    with a dynamic M select the production tuning entries.
    """
    comptime K2 = N1
    comptime N_out = N2 if chained else N1
    comptime out_len = M * N_out

    var stale_host = ctx.enqueue_create_host_buffer[dtype](M * K1)
    var w1_host = ctx.enqueue_create_host_buffer[dtype](N1 * K1)
    var w2_host = ctx.enqueue_create_host_buffer[dtype](N2 * K2)
    var result_host = ctx.enqueue_create_host_buffer[dtype](out_len)
    for i in range(M * K1):
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
    var result_device = ctx.enqueue_create_buffer[dtype](out_len)
    ctx.enqueue_copy(w1_device, w1_host)
    ctx.enqueue_copy(w2_device, w2_host)

    var a_tt = TileTensor(a_device, row_major(Int64(M), Idx[K1])).as_imm()
    var w1_tt = TileTensor(w1_device, row_major(Idx[N1], Idx[K1])).as_imm()
    var c1_tt = TileTensor(c1_device, row_major(Int64(M), Idx[N1]))
    var c1_in = TileTensor(c1_device, row_major(Int64(M), Idx[K2])).as_imm()
    var w2_tt = TileTensor(w2_device, row_major(Idx[N2], Idx[K2])).as_imm()
    var c2_tt = TileTensor(c2_device, row_major(Int64(M), Idx[N2]))

    var expected = Scalar[dtype](2 * K1)
    comptime if chained:
        expected = Scalar[dtype](2 * K1 * N1)

    for iter in range(num_iters):
        ctx.enqueue_copy(a_device, stale_host)
        ctx.enqueue_memset[dtype](c1_device, 0)
        ctx.enqueue_memset[dtype](c2_device, 0)
        ctx.enqueue_memset[dtype](result_device, 0)

        ctx.enqueue_function[slow_producer](
            a_device,
            Int32(M * K1),
            Int32(delay_us),
            grid_dim=4,
            block_dim=64,
            attributes=pdl_launch_attributes(PDLLevel.OVERLAP_AT_END),
        )

        matmul_dispatch_sm100[
            transpose_b=True, pdl_level=PDLLevel.OVERLAP_AT_BEGINNING
        ](c1_tt, a_tt, w1_tt, ctx)
        comptime if chained:
            matmul_dispatch_sm100[
                transpose_b=True, pdl_level=PDLLevel.OVERLAP_AT_BEGINNING
            ](c2_tt, c1_in, w2_tt, ctx)

        var final_buf = c2_device if chained else c1_device
        ctx.enqueue_function[consumer](
            final_buf,
            result_device,
            Int32(out_len),
            grid_dim=ceildiv(out_len, 256),
            block_dim=256,
            attributes=pdl_launch_attributes(PDLLevel.OVERLAP_AT_END),
        )

        ctx.synchronize()
        ctx.enqueue_copy(result_host, result_device)
        ctx.synchronize()

        var errors = 0
        var tolerance = expected * Scalar[dtype](0.02)
        for i in range(out_len):
            var diff = result_host[i] - expected
            if diff < 0:
                diff = -diff
            # `not (diff <= tolerance)` also rejects NaN.
            if not (diff <= tolerance):
                errors += 1
        if errors > 0:
            raise Error(
                "dispatcher PDL race detected in "
                + label
                + " iter "
                + String(iter)
                + ": "
                + String(errors)
                + "/"
                + String(out_len)
                + " mismatches, first value "
                + String(result_host[0])
                + " expected "
                + String(expected)
            )

    print("PASS: ", label, " x", num_iters, " iters")


def main() raises:
    var iters = get_defined_int["ITERS", 30]()
    var delay_us = get_defined_int["DELAY_US", 5000]()

    with DeviceContext() as ctx:
        # Small-M/N tuning table entry (M in [2, 5), N=384, K=7168), split-K.
        check_routes_to_small_mn_split_k(3, 384, 7168)
        run_case[3, 384, 7168, 1024, False](
            ctx, "small_mn_split_k", iters, delay_us
        )

        # Two M=1 decode GEMVs through `dispatch_gemv`.
        run_case[1, 4096, 7168, 1024, True](
            ctx, "m1_gemv_then_gemv", iters, delay_us
        )

        print("All dispatcher PDL race tests passed!")
