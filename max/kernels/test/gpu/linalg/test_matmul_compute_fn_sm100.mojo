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
"""Tests the compute-closure `_matmul_gpu` on every SM100 dispatch path.

Each shape is chosen to land on one `matmul_dispatch_sm100` path: GEMV,
small-MN split-K GEMV, small-MN MMA-cp.async, unaligned-N split-K GEMV,
IEEE-fp32 split-K GEMV, a tuned (N, K) whose split-K outlier rows the closure
skips, and the tcgen05 tile GEMM. The closure scales by a power of two, so its
result must match the plain dispatch exactly, except for the tuned (N, K): its
plain dispatch takes a split-K row and sums K in another order, so that case
allows one bf16 ulp.
"""

from max.gpu.host import DeviceContext
from layout import Coord, Idx, TileTensor, row_major
from layout._fillers import random
from linalg.matmul.gpu import _matmul_gpu

from std.testing import assert_true


def test_dispatch_compute_fn[
    a_type: DType,
    c_type: DType,
    N: Int,
    K: Int,
    one_ulp: Bool = False,
](ctx: DeviceContext, m: Int) raises:
    """Checks that the compute-closure `_matmul_gpu` stores exactly the
    closure applied to what the plain dispatch stores, at the closure's
    index. With `one_ulp`, the two may differ by one bf16 ulp: the plain
    dispatch takes a split-K row the closure must skip, so they sum K in
    different orders."""
    comptime b_type = a_type
    var c_size = m * N

    var a_host_ptr = ctx.enqueue_create_host_buffer[a_type](m * K)
    var b_host_ptr = ctx.enqueue_create_host_buffer[b_type](N * K)
    var c_host_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)
    var c_ref_host_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)
    random(TileTensor(a_host_ptr, row_major(m, Idx[K])))
    random(TileTensor(b_host_ptr, row_major[N, K]()))

    var a_dev = ctx.enqueue_create_buffer[a_type](m * K)
    var b_dev = ctx.enqueue_create_buffer[b_type](N * K)
    var c_dev = ctx.enqueue_create_buffer[c_type](c_size)
    var c_ref_dev = ctx.enqueue_create_buffer[c_type](c_size)
    ctx.enqueue_copy(a_dev, a_host_ptr)
    ctx.enqueue_copy(b_dev, b_host_ptr)
    ctx.enqueue_memset(c_dev, 0)
    ctx.enqueue_memset(c_ref_dev, 0)

    var a_tensor = TileTensor(a_dev, row_major(m, Idx[K])).as_imm()
    var b_tensor = TileTensor(b_dev, row_major(Idx[N], Idx[K])).as_imm()
    var c_tensor = TileTensor(c_dev, row_major(m, Idx[N]))
    var c_ref_tensor = TileTensor(c_ref_dev, row_major(m, Idx[N]))

    var scale: Int = 2

    def scale_odd_rows_negated[
        dtype: DType, width: SIMDLength, *, alignment: Int
    ](idx: Coord, val: SIMD[dtype, width]) {var scale} -> SIMD[dtype, width]:
        comptime assert idx.rank == 2, "the matmul epilogue contract is rank 2"
        var row_scale = -scale if Int(idx[0]) % 2 == 1 else scale
        return val * SIMD[dtype, width](row_scale)

    _matmul_gpu[use_tensor_core=True, transpose_b=True](
        c_ref_tensor, a_tensor, b_tensor, ctx
    )
    _matmul_gpu[use_tensor_core=True, transpose_b=True](
        c_tensor, a_tensor, b_tensor, scale_odd_rows_negated, ctx
    )

    ctx.enqueue_copy(c_host_ptr, c_dev)
    ctx.enqueue_copy(c_ref_host_ptr, c_ref_dev)
    ctx.synchronize()

    var errors = 0
    for i in range(c_size):
        var row_scale = -2 if (i // N) % 2 == 1 else 2
        var expected = c_ref_host_ptr[i] * Scalar[c_type](row_scale)
        var slack = abs(expected.cast[DType.float32]()) / 128 if one_ulp else 0
        var diff = abs(
            c_host_ptr[i].cast[DType.float32]() - expected.cast[DType.float32]()
        )
        var steps = abs(Int(c_host_ptr[i].to_bits()) - Int(expected.to_bits()))
        # The relative slack alone admits two steps just below a power of
        # two, so `one_ulp` also bounds the encoding distance. Negated so that
        # a NaN counts as an error.
        if not (diff == 0 or (one_ulp and diff <= slack and steps <= 1)):
            errors += 1
            if errors <= 5:
                print(
                    "  MISMATCH [",
                    i // N,
                    ",",
                    i % N,
                    "]: got",
                    c_host_ptr[i],
                    "expected",
                    expected,
                )
    print(
        "  compute_fn",
        a_type,
        "M=",
        m,
        " N=",
        N,
        " K=",
        K,
        " errors=",
        errors,
    )
    assert_true(errors == 0, msg=String("COMPUTE_FN FAILED:", errors))


def main() raises:
    with DeviceContext() as ctx:
        # GEMV.
        test_dispatch_compute_fn[.bfloat16, .bfloat16, 4096, 4096](ctx, 1)
        # Small-MN split-K GEMV.
        test_dispatch_compute_fn[.bfloat16, .bfloat16, 384, 7168](ctx, 8)
        # Small-MN MMA-cp.async.
        test_dispatch_compute_fn[.bfloat16, .bfloat16, 384, 7168](ctx, 28)
        test_dispatch_compute_fn[.bfloat16, .bfloat16, 264, 4096](ctx, 33)
        # Unaligned-N split-K GEMV: N * 2 bytes is not 16-byte aligned.
        test_dispatch_compute_fn[.bfloat16, .bfloat16, 100, 4096](ctx, 4)
        # IEEE-fp32 split-K GEMV.
        test_dispatch_compute_fn[.float32, .float32, 128, 6144](ctx, 4)
        # Tuned (N, K) whose split-K outlier rows the compute epilogue skips.
        test_dispatch_compute_fn[.bfloat16, .bfloat16, 2112, 14336, True](
            ctx, 64
        )
        # tcgen05 tile GEMM.
        test_dispatch_compute_fn[.bfloat16, .bfloat16, 4096, 4096](ctx, 256)
