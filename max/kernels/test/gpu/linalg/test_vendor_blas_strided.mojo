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
"""Checks vendor BLAS matmul on operands whose rows are padded.

Every operand's row stride exceeds its column count, so the vendor call must
pass the row stride as the leading dimension. The padding holds values that
would corrupt the result if read as data. On AMD, float32 runs through rocBLAS
and bfloat16 through hipBLASLt.
"""

from std.testing import assert_equal

import linalg.matmul.vendor.blas as vendor_blas
from max.gpu.host import DeviceContext
from layout import Coord, Idx, MixedLayout, TileTensor


def test_padded_rows[
    dtype: DType
](ctx: DeviceContext, transpose_b: Bool) raises:
    comptime M = 5
    comptime N = 64
    comptime K = 64
    comptime pad = 16
    comptime lda = K + pad
    comptime ldc = N + pad
    var b_rows = N if transpose_b else K
    var b_cols = K if transpose_b else N
    var ldb = b_cols + pad
    print(dtype, "transpose_b:", transpose_b)

    # Dyadic inputs make the FP32 dot products exact before the output cast.
    def a_val(m: Int, k: Int) -> Float32:
        return Float32((m + 3 * k) % 17 - 8) * 0.125

    def b_val(n: Int, k: Int) -> Float32:
        return Float32((3 * n + k) % 13 - 6) * 0.125

    var a_dev = ctx.enqueue_create_buffer[dtype](M * lda)
    var b_dev = ctx.enqueue_create_buffer[dtype](b_rows * ldb)
    var c_dev = ctx.enqueue_create_buffer[dtype](M * ldc)
    with c_dev.map_to_host() as hc:
        for i in range(M * ldc):
            hc[i] = Scalar[dtype](-777)
    with a_dev.map_to_host() as ha, b_dev.map_to_host() as hb:
        for i in range(M * lda):
            ha[i] = Scalar[dtype](100)
        for i in range(b_rows * ldb):
            hb[i] = Scalar[dtype](100)
        for m in range(M):
            for k in range(K):
                ha[m * lda + k] = a_val(m, k).cast[dtype]()
        for n in range(N):
            for k in range(K):
                var idx = n * ldb + k if transpose_b else k * ldb + n
                hb[idx] = b_val(n, k).cast[dtype]()

    var a = TileTensor(
        a_dev, MixedLayout(Coord(Idx[M], Idx[K]), Coord(Idx[lda], Idx[1]))
    )
    var b = TileTensor(
        b_dev,
        MixedLayout(Coord(b_rows, b_cols), Coord(ldb, Idx[1])),
    )
    var c = TileTensor(
        c_dev, MixedLayout(Coord(Idx[M], Idx[N]), Coord(Idx[ldc], Idx[1]))
    )
    vendor_blas.matmul(ctx, c, a, b, c_row_major=True, transpose_b=transpose_b)

    with c_dev.map_to_host() as hc:
        for m in range(M):
            for n in range(N):
                var expected = Float32(0)
                for k in range(K):
                    expected += a_val(m, k) * b_val(n, k)
                assert_equal(hc[m * ldc + n], expected.cast[dtype]())
            for n in range(N, ldc):
                assert_equal(hc[m * ldc + n], Scalar[dtype](-777))


def main() raises:
    with DeviceContext() as ctx:
        test_padded_rows[.float32](ctx, transpose_b=True)
        test_padded_rows[.float32](ctx, transpose_b=False)
        test_padded_rows[.bfloat16](ctx, transpose_b=True)
        test_padded_rows[.bfloat16](ctx, transpose_b=False)
