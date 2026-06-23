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

from std.random import rand

from std.gpu.host import DeviceContext
from layout import Coord, TileTensor, row_major
from nn.normalization import row_mean_of_squares
from std.testing import assert_almost_equal

from std.utils.index import Index, IndexList


def run_row_mean_of_squares_gpu[
    in_dtype: DType
](
    ctx: DeviceContext,
    rows: Int,
    cols: Int,
    rtol: Float64 = 1e-3,
    atol: Float64 = 1e-3,
) raises:
    print("== run_row_mean_of_squares_gpu rows=", rows, " cols=", cols)

    comptime out_dtype = DType.float32

    var data_h = ctx.enqueue_create_host_buffer[in_dtype](rows * cols)
    var out_h = ctx.enqueue_create_host_buffer[out_dtype](rows)

    rand[in_dtype](data_h.as_span())

    var data_d = ctx.enqueue_create_buffer[in_dtype](rows * cols)
    var out_d = ctx.enqueue_create_buffer[out_dtype](rows)

    var shape = Index(rows, cols)
    var data_buf = TileTensor(data_d, row_major(Coord(shape)))
    var out_buf = TileTensor(out_d, row_major(Coord(Index(rows, 1))))

    ctx.enqueue_copy(data_d, data_h)

    @always_inline
    @__copy_capture(data_buf)
    @parameter
    def input_fn[
        width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[in_dtype, width]:
        var idx = data_buf.layout(Coord(coords))
        return data_buf.raw_load[width=width](idx)

    @always_inline
    @__copy_capture(out_buf)
    @parameter
    def output_fn(row: Int, val: Scalar[out_dtype]) -> None:
        var idx = out_buf.layout(Coord(Index(row, 0)))
        out_buf.raw_store[width=1](idx, val)

    row_mean_of_squares[input_fn, output_fn, target="gpu"](shape, ctx)
    ctx.enqueue_copy(out_h, out_d)
    ctx.synchronize()

    # Float64 reference oracle.
    for r in range(rows):
        var acc = Float64(0)
        for c in range(cols):
            var v = data_h[r * cols + c].cast[DType.float64]()
            acc += v * v
        var expected = acc / Float64(cols)
        assert_almost_equal(Float64(out_h[r]), expected, rtol=rtol, atol=atol)

    _ = data_d
    _ = out_d


def main() raises:
    with DeviceContext() as ctx:
        # bfloat16 (primary): decode + prefill shapes, plus odd-N tail.
        run_row_mean_of_squares_gpu[DType.bfloat16](ctx, 16, 1536)
        run_row_mean_of_squares_gpu[DType.bfloat16](ctx, 16, 256)
        run_row_mean_of_squares_gpu[DType.bfloat16](ctx, 512, 1536)
        run_row_mean_of_squares_gpu[DType.bfloat16](ctx, 2048, 256)
        run_row_mean_of_squares_gpu[DType.bfloat16](ctx, 16, 1537)
        # A column count larger than one block can cover (grid-stride loop).
        run_row_mean_of_squares_gpu[DType.bfloat16](ctx, 4, 8192)

        # float32 must also be accepted (tighter tolerance).
        run_row_mean_of_squares_gpu[DType.float32](
            ctx, 16, 1536, rtol=1e-6, atol=1e-6
        )
        run_row_mean_of_squares_gpu[DType.float32](
            ctx, 16, 256, rtol=1e-6, atol=1e-6
        )
        run_row_mean_of_squares_gpu[DType.float32](
            ctx, 16, 1537, rtol=1e-6, atol=1e-6
        )
