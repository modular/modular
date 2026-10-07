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
"""Tests the compute-closure `multistage_gemm` on the NVIDIA kernel.

The shapes cover each store path of `multistage_gemm_kernel`: the bf16 output
staged through shared memory, the aligned fp32 output, and the scalar store
for an odd fp32 N. M is not a multiple of the block tile, so the last row of
blocks is partial.
"""

from std.random import rand

from max.gpu.host import DeviceContext
from layout import Idx, TileTensor, row_major
from linalg.matmul.gpu import multistage_gemm
from linalg.utils_gpu import MatmulKernels
from std.testing import assert_equal

from std.utils import IndexList


def test_compute_fn[
    a_type: DType, c_type: DType, N: Int, K: Int
](ctx: DeviceContext, m: Int) raises:
    """Checks that a compute closure's result is what the kernel stores
    into `c`, at the index the closure was given."""
    print("compute:", a_type, "->", c_type, m, "x", N, "x", K)
    comptime config = MatmulKernels[
        a_type, a_type, c_type, True
    ]().ampere_256x64_4

    var a_dev = ctx.enqueue_create_buffer[a_type](m * K)
    var b_dev = ctx.enqueue_create_buffer[a_type](N * K)
    var c_ref_dev = ctx.enqueue_create_buffer[c_type](m * N)
    var c_dev = ctx.enqueue_create_buffer[c_type](m * N)

    with a_dev.map_to_host() as ha, b_dev.map_to_host() as hb:
        rand(ha.unsafe_ptr(), m * K, min=-1.0, max=1.0)
        rand(hb.unsafe_ptr(), N * K, min=-1.0, max=1.0)
    ctx.enqueue_memset(c_ref_dev, 0)
    ctx.enqueue_memset(c_dev, 0)

    var a = TileTensor(a_dev, row_major(m, Idx[K])).as_imm()
    var b = TileTensor(b_dev, row_major(Idx[N], Idx[K])).as_imm()
    var c_ref = TileTensor(c_ref_dev, row_major(m, Idx[N]))
    var c = TileTensor(c_dev, row_major(m, Idx[N]))

    var scale: Int = 2

    def scale_odd_rows_negated[
        dtype: DType, width: SIMDLength, *, alignment: Int
    ](idx: IndexList[2], val: SIMD[dtype, width]) {var scale} -> SIMD[
        dtype, width
    ]:
        var row_scale = -scale if idx[0] % 2 == 1 else scale
        return val * SIMD[dtype, width](row_scale)

    multistage_gemm[transpose_b=True, config=config](c_ref, a, b, ctx)
    multistage_gemm[transpose_b=True, config=config](
        c, a, b, scale_odd_rows_negated, ctx
    )

    with c_ref_dev.map_to_host() as h_ref, c_dev.map_to_host() as h_c:
        for i in range(m * N):
            var row_scale = -2 if (i // N) % 2 == 1 else 2
            assert_equal(h_c[i], h_ref[i] * Scalar[c_type](row_scale))


def main() raises:
    with DeviceContext() as ctx:
        test_compute_fn[.bfloat16, .bfloat16, 512, 512](ctx, 130)
        test_compute_fn[.float32, .float32, 512, 512](ctx, 130)
        test_compute_fn[.float32, .float32, 257, 512](ctx, 130)
