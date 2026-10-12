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
"""Tests the store-closure `matmul_dispatch_sm100` on every SM100 dispatch
path.

Each shape is chosen to land on one path: GEMV, small-MN split-K GEMV,
small-MN MMA-cp.async, unaligned-N split-K GEMV, IEEE-fp32 split-K GEMV, the
vendor BLAS low-perf route, and the tcgen05 tile GEMM. The closure scales by a
power of two, so its result must match the plain dispatch exactly.
"""

from std.random import rand

from max.gpu.host import DeviceContext
from layout import Coord, Idx, TileTensor, row_major
from linalg.matmul.gpu.sm100_structured.default.dispatch import (
    matmul_dispatch_sm100,
)
from std.testing import assert_equal

from std.utils.index import IndexList


def test_dispatch_epilogue_fn[
    a_type: DType,
    c_type: DType,
    N: Int,
    K: Int,
    *,
    kernel_writes_c: Bool = False,
](ctx: DeviceContext, m: Int) raises:
    """Checks that a store closure receives each output at its index and,
    unless the path runs the matmul into `c` first, that `c` is untouched."""
    print("epilogue_fn:", a_type, "->", c_type, m, "x", N, "x", K)
    var a_dev = ctx.enqueue_create_buffer[a_type](m * K)
    var b_dev = ctx.enqueue_create_buffer[a_type](N * K)
    var c_ref_dev = ctx.enqueue_create_buffer[c_type](m * N)
    var c_dev = ctx.enqueue_create_buffer[c_type](m * N)
    var d_dev = ctx.enqueue_create_buffer[c_type](m * N)

    with a_dev.map_to_host() as ha, b_dev.map_to_host() as hb:
        rand(ha.unsafe_ptr(), m * K, min=-1.0, max=1.0)
        rand(hb.unsafe_ptr(), N * K, min=-1.0, max=1.0)
    with c_dev.map_to_host() as hc:
        for i in range(m * N):
            hc[i] = Scalar[c_type](7)
    ctx.enqueue_memset(c_ref_dev, 0)
    ctx.enqueue_memset(d_dev, 0)

    var a = TileTensor(a_dev, row_major(m, Idx[K])).as_imm()
    var b = TileTensor(b_dev, row_major(Idx[N], Idx[K])).as_imm()
    var c_ref = TileTensor(c_ref_dev, row_major(m, Idx[N]))
    var c = TileTensor(c_dev, row_major(m, Idx[N]))
    var d = TileTensor(d_dev, row_major(m, Idx[N]))

    def store_odd_rows_negated[
        dtype: DType, width: SIMDLength, *, alignment: Int
    ](idx: IndexList[2], val: SIMD[dtype, width]) {var d}:
        var row_scale = -2 if idx[0] % 2 == 1 else 2
        d.store[width=width](
            Coord(idx),
            (val * SIMD[dtype, width](row_scale)).cast[c_type](),
        )

    matmul_dispatch_sm100[transpose_b=True](c_ref, a, b, ctx)
    matmul_dispatch_sm100[transpose_b=True](
        c, a, b, store_odd_rows_negated, ctx
    )

    with c_ref_dev.map_to_host() as h_ref, c_dev.map_to_host() as h_c, d_dev.map_to_host() as h_d:
        for i in range(m * N):
            var row_scale = -2 if (i // N) % 2 == 1 else 2
            assert_equal(h_d[i], h_ref[i] * Scalar[c_type](row_scale))
            comptime if not kernel_writes_c:
                assert_equal(h_c[i], Scalar[c_type](7))


def main() raises:
    with DeviceContext() as ctx:
        # GEMV.
        test_dispatch_epilogue_fn[.bfloat16, .bfloat16, 4096, 4096](ctx, 1)
        # Small-MN split-K GEMV.
        test_dispatch_epilogue_fn[.bfloat16, .bfloat16, 384, 7168](ctx, 8)
        # Small-MN MMA-cp.async.
        test_dispatch_epilogue_fn[.bfloat16, .bfloat16, 384, 7168](ctx, 28)
        test_dispatch_epilogue_fn[.bfloat16, .bfloat16, 264, 4096](ctx, 33)
        # Unaligned-N split-K GEMV: N * 2 bytes is not 16-byte aligned.
        test_dispatch_epilogue_fn[.bfloat16, .bfloat16, 100, 4096](ctx, 4)
        # IEEE-fp32 split-K GEMV.
        test_dispatch_epilogue_fn[.float32, .float32, 128, 6144](ctx, 4)
        # Vendor BLAS low-perf route: the matmul runs into `c` first.
        test_dispatch_epilogue_fn[
            .bfloat16, .bfloat16, 2112, 14336, kernel_writes_c=True
        ](ctx, 64)
        # tcgen05 tile GEMM.
        test_dispatch_epilogue_fn[.bfloat16, .bfloat16, 4096, 4096](ctx, 256)
