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
"""Tests the store-closure SM100 tile GEMM.

The configs cover the 1-SM and 2-SM kernels, the register- and shared-memory
based epilogues, swapAB (whose epilogue transposes the closure's index), and a
partial N tile. M is not a multiple of the block tile, so the last row of
blocks is partial. The closure scales by a power of two, so its result must
match the plain store of the same kernel exactly.
"""

from std.random import rand

from max.gpu.host import DeviceContext
from layout import Coord, Idx, TileTensor, row_major
from linalg.matmul.gpu.sm100_structured.default.matmul import (
    blackwell_matmul_tma_umma_warp_specialized,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.config import (
    MatmulConfig,
)
from std.testing import assert_equal

from std.utils.index import Index, IndexList


def test_epilogue_fn[
    N: Int,
    K: Int,
    *,
    mma_shape: IndexList[3],
    cluster_shape: IndexList[3],
    cta_group: Int,
    register_based_epilogue: Bool = True,
    swapAB: Bool = False,
](ctx: DeviceContext, m: Int) raises:
    """Checks that a store closure receives each output at its index and that
    the kernel itself leaves `c` untouched."""
    comptime dtype = DType.bfloat16
    print(
        "epilogue_fn: M=",
        m,
        " N=",
        N,
        " K=",
        K,
        " mma_shape=",
        mma_shape,
        " cta_group=",
        cta_group,
        " register_based_epilogue=",
        register_based_epilogue,
        " swapAB=",
        swapAB,
    )
    comptime config = MatmulConfig[dtype, dtype, dtype, True](
        cluster_shape=cluster_shape,
        mma_shape=mma_shape,
        cta_group=cta_group,
        AB_swapped=swapAB,
        register_based_epilogue=register_based_epilogue,
    )

    var a_dev = ctx.enqueue_create_buffer[dtype](m * K)
    var b_dev = ctx.enqueue_create_buffer[dtype](N * K)
    var c_ref_dev = ctx.enqueue_create_buffer[dtype](m * N)
    var c_dev = ctx.enqueue_create_buffer[dtype](m * N)
    var d_dev = ctx.enqueue_create_buffer[dtype](m * N)

    with a_dev.map_to_host() as ha, b_dev.map_to_host() as hb:
        rand(ha.unsafe_ptr(), m * K, min=-1.0, max=1.0)
        rand(hb.unsafe_ptr(), N * K, min=-1.0, max=1.0)
    with c_dev.map_to_host() as hc:
        for i in range(m * N):
            hc[i] = Scalar[dtype](7)
    ctx.enqueue_memset(c_ref_dev, 0)
    ctx.enqueue_memset(d_dev, 0)

    var a = TileTensor(a_dev, row_major(m, Idx[K])).as_imm()
    var b = TileTensor(b_dev, row_major(Idx[N], Idx[K])).as_imm()
    var c_ref = TileTensor(c_ref_dev, row_major(m, Idx[N]))
    var c = TileTensor(c_dev, row_major(m, Idx[N]))
    var d = TileTensor(d_dev, row_major(m, Idx[N]))

    def store_odd_rows_negated[
        _dtype: DType, width: SIMDLength, *, alignment: Int
    ](idx: IndexList[2], val: SIMD[_dtype, width]) {var d}:
        var row_scale = -2 if idx[0] % 2 == 1 else 2
        d.store[width=width](
            Coord(idx),
            (val * SIMD[_dtype, width](row_scale)).cast[dtype](),
        )

    blackwell_matmul_tma_umma_warp_specialized[transpose_b=True, config=config](
        c_ref, a, b, ctx
    )
    blackwell_matmul_tma_umma_warp_specialized[transpose_b=True, config=config](
        c, a, b, store_odd_rows_negated, ctx
    )

    with c_ref_dev.map_to_host() as h_ref, c_dev.map_to_host() as h_c, d_dev.map_to_host() as h_d:
        for i in range(m * N):
            var row_scale = -2 if (i // N) % 2 == 1 else 2
            assert_equal(h_d[i], h_ref[i] * Scalar[dtype](row_scale))
            assert_equal(h_c[i], Scalar[dtype](7))


def main() raises:
    with DeviceContext() as ctx:
        comptime for register_based_epilogue in [True, False]:
            test_epilogue_fn[
                1024,
                1024,
                mma_shape=Index(256, 128, 16),
                cluster_shape=Index(2, 1, 1),
                cta_group=2,
                register_based_epilogue=register_based_epilogue,
            ](ctx, 1000)
            test_epilogue_fn[
                1024,
                1024,
                mma_shape=Index(128, 128, 16),
                cluster_shape=Index(2, 1, 1),
                cta_group=1,
                register_based_epilogue=register_based_epilogue,
            ](ctx, 1000)
            test_epilogue_fn[
                2560,
                1024,
                mma_shape=Index(256, 128, 16),
                cluster_shape=Index(4, 4, 1),
                cta_group=2,
                register_based_epilogue=register_based_epilogue,
                swapAB=True,
            ](ctx, 100)
        # A partial N tile, in the plain and the transposed epilogue.
        comptime for swapAB in [False, True]:
            test_epilogue_fn[
                128,
                128,
                mma_shape=Index(64, 88, 16),
                cluster_shape=Index(2, 1, 1),
                cta_group=1,
                swapAB=swapAB,
            ](ctx, 64)
