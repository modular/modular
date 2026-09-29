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
"""Pins `gemv_split_k`'s `tile_m=1, grid.x=M>1` path at an all-fp32,
`weight_non_temporal=False` shape before any dispatch change routes traffic
there.

`is_minimax_router_gemm` gates this path, and widening it for an fp32 router
such as Kimi K3's reaches a combination no other test runs: `test_gemv.mojo`
covers all-fp32 `grid.x=M>1` only with the default `weight_non_temporal=True`,
and `weight_non_temporal=False` only with bf16 activations.

The shape is K3's router gate at TP8 (N=896, K=7168). A host fp32 reference
catches wrong math; bit-exact agreement with M separate `grid.x=1` launches of
the same kernel catches indexing errors in the `grid.x=M` path.
"""

from std.math import ceildiv
from std.random import randn, random_float64, seed
from std.sys import has_amd_gpu_accelerator, simd_width_of
from std.testing import assert_equal

from internal_utils import assert_almost_equal
from layout import TileTensor, row_major
from linalg.gemv import gemv_split_k
from max.gpu.host import DeviceContext, get_gpu_target

# K3's MoE router gate at TP8: N = routed-expert count, K = hidden_size. The
# weight is replicated (not TP-split, see `KimiK3MoEGate`), so this is the
# real per-device shape, not a sharded fraction of it.
comptime ROUTER_N = 896
comptime ROUTER_K = 7168

# The launch `gemv_gpu_dispatch` selects for `is_minimax_router_gemm`:
# `_gemv_split_k_dispatch[128, 2, 2, False]`.
comptime NUM_THREADS = 128
comptime TILE_N = 2
comptime UNROLL_FACTOR = 2
comptime TILE_M = 1


def _run_case[M: Int](ctx: DeviceContext) raises:
    """Runs `gemv_split_k` at `[M, ROUTER_K] x [ROUTER_N, ROUTER_K]^T`, fp32,
    `weight_non_temporal=False`, and checks the result two ways.
    """
    print(
        "== gemv_split_k router-gate shape M=",
        M,
        "N=",
        ROUTER_N,
        "K=",
        ROUTER_K,
    )

    comptime c_type = DType.float32
    comptime a_type = DType.float32
    comptime b_type = DType.float32
    comptime simd_width = simd_width_of[c_type, target=get_gpu_target()]()
    comptime check_bounds_n = ROUTER_N % TILE_N != 0

    var a_host = ctx.enqueue_create_host_buffer[a_type](M * ROUTER_K)
    var b_host = ctx.enqueue_create_host_buffer[b_type](ROUTER_N * ROUTER_K)
    var c_host = ctx.enqueue_create_host_buffer[c_type](M * ROUTER_N)
    var c_rowwise_host = ctx.enqueue_create_host_buffer[c_type](M * ROUTER_N)
    var c_expected = ctx.enqueue_create_host_buffer[c_type](M * ROUTER_N)

    for i in range(M * ROUTER_K):
        a_host[i] = random_float64(min=-1.0, max=1.0).cast[a_type]()
    randn(b_host.unsafe_ptr(), ROUTER_N * ROUTER_K)

    var a_dev = ctx.enqueue_create_buffer[a_type](M * ROUTER_K)
    var b_dev = ctx.enqueue_create_buffer[b_type](ROUTER_N * ROUTER_K)
    var c_dev = ctx.enqueue_create_buffer[c_type](M * ROUTER_N)
    var c_rowwise_dev = ctx.enqueue_create_buffer[c_type](M * ROUTER_N)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    var a_tt = TileTensor(a_dev, row_major(M, ROUTER_K)).as_imm()
    var b_tt = TileTensor(b_dev, row_major(ROUTER_N, ROUTER_K)).as_imm()
    var c_tt = TileTensor(c_dev, row_major(M, ROUTER_N))

    # Check 1: the full M-row launch -- this is `grid.x = ceildiv(M, 1) = M`,
    # the path with no prior coverage.
    comptime kernel_full = gemv_split_k[
        c_type,
        a_type,
        b_type,
        type_of(c_tt).LayoutType,
        type_of(a_tt).LayoutType,
        type_of(b_tt).LayoutType,
        type_of(c_tt).Engine,
        type_of(a_tt).Engine,
        type_of(b_tt).Engine,
        simd_width=simd_width,
        tile_m=TILE_M,
        tile_n=TILE_N,
        num_threads=NUM_THREADS,
        unroll_factor=UNROLL_FACTOR,
        weight_non_temporal=False,
        check_bounds_m=TILE_M > 1,
        check_bounds_n=check_bounds_n,
    ]
    ctx.enqueue_function[kernel_full](
        c_tt,
        a_tt,
        b_tt,
        Int32(M),
        Int32(ROUTER_N),
        Int32(ROUTER_K),
        grid_dim=(ceildiv(M, TILE_M), ceildiv(ROUTER_N, TILE_N)),
        block_dim=NUM_THREADS,
    )

    # Check 2: M independent grid.x=1 dispatches of the same kernel, the
    # launch every production call has exercised. The row views' layout is
    # static, so one instance binds the kernel type for every row.
    var a_row0_tt = TileTensor(
        a_dev.unsafe_ptr(), row_major(1, ROUTER_K)
    ).as_imm()
    var c_row0_tt = TileTensor(
        c_rowwise_dev.unsafe_ptr(), row_major(1, ROUTER_N)
    )
    comptime kernel_row = gemv_split_k[
        c_type,
        a_type,
        b_type,
        type_of(c_row0_tt).LayoutType,
        type_of(a_row0_tt).LayoutType,
        type_of(b_tt).LayoutType,
        type_of(c_row0_tt).Engine,
        type_of(a_row0_tt).Engine,
        type_of(b_tt).Engine,
        simd_width=simd_width,
        tile_m=TILE_M,
        tile_n=TILE_N,
        num_threads=NUM_THREADS,
        unroll_factor=UNROLL_FACTOR,
        weight_non_temporal=False,
        check_bounds_m=False,
        check_bounds_n=check_bounds_n,
    ]
    # Only their types were needed, to bind `kernel_row` above.
    _ = a_row0_tt
    _ = c_row0_tt
    for row in range(M):
        var a_row_tt = TileTensor(
            a_dev.unsafe_ptr().unsafe_offset(row * ROUTER_K),
            row_major(1, ROUTER_K),
        ).as_imm()
        var c_row_tt = TileTensor(
            c_rowwise_dev.unsafe_ptr().unsafe_offset(row * ROUTER_N),
            row_major(1, ROUTER_N),
        )
        ctx.enqueue_function[kernel_row](
            c_row_tt,
            a_row_tt,
            b_tt,
            Int32(1),
            Int32(ROUTER_N),
            Int32(ROUTER_K),
            grid_dim=(1, ceildiv(ROUTER_N, TILE_N)),
            block_dim=NUM_THREADS,
        )

    ctx.enqueue_copy(c_host, c_dev)
    ctx.enqueue_copy(c_rowwise_host, c_rowwise_dev)
    ctx.synchronize()

    # DRIV-199: keep device buffers alive past synchronize.
    _ = a_dev^
    _ = b_dev^
    _ = c_dev^
    _ = c_rowwise_dev^

    # Check 1: host fp32 reference -- catches an arithmetic error.
    for m in range(M):
        for n in range(ROUTER_N):
            var acc = Float32(0)
            for kk in range(ROUTER_K):
                acc += (
                    a_host[m * ROUTER_K + kk].cast[.float32]()
                    * b_host[n * ROUTER_K + kk].cast[.float32]()
                )
            c_expected[m * ROUTER_N + n] = acc

    assert_almost_equal(
        c_host.unsafe_ptr(),
        c_expected.unsafe_ptr(),
        num_elements=M * ROUTER_N,
        atol=1e-3,
        rtol=1e-2,
    )

    # Check 2: bit-exact, since the kernel and reduction order are identical;
    # a mismatch is an indexing bug in the grid.x=M path.
    for i in range(M * ROUTER_N):
        assert_equal(
            c_host[i],
            c_rowwise_host[i],
            "grid.x=M output disagrees with the M=1 reference at element "
            + String(i),
        )

    print("PASS")


def main() raises:
    comptime if not has_amd_gpu_accelerator():
        print("SKIP: AMD GPU not available")
        return

    seed(0)

    with DeviceContext() as ctx:
        # M=1: the only point every production call has actually exercised.
        _run_case[1](ctx)
        # M=2..16: the window `is_minimax_router_gemm` admits (`m <= 16`),
        # covering both ends and the specific M=9 K3's verify step needs.
        _run_case[2](ctx)
        _run_case[3](ctx)
        _run_case[5](ctx)
        _run_case[8](ctx)
        _run_case[9](ctx)
        _run_case[13](ctx)
        _run_case[16](ctx)

    print("ALL TESTS PASSED")
