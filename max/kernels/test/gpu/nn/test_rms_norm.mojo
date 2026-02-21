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

from math import sqrt
from random import rand

from gpu.host import DeviceContext
from layout._coord import Coord, Idx
from layout._layout import row_major
from layout._tile_tensor import TileTensor
from nn.normalization import *
from testing import assert_almost_equal

from utils.index import Index, IndexList


fn compute_rms[
    dtype: DType
](data: TileTensor[dtype, ...], size: Int, eps: Scalar[dtype]) -> Scalar[dtype]:
    comptime assert data.flat_rank == 1, "data.rank must be 1"
    comptime assert data.element_size == 1

    comptime accum_type = get_accum_type[dtype]()
    var sum_of_squares = Scalar[accum_type]()
    for i in range(size):
        var val = data[i][0].cast[accum_type]()
        sum_of_squares += val * val
    var result = sqrt(
        (sum_of_squares / Scalar[accum_type](data.numel()))
        + eps.cast[accum_type]()
    )
    return result.cast[dtype]()


fn run_rms_norm_gpu[
    rank: Int, //, dtype: DType, *, static_cols: Int = -1
](ctx: DeviceContext, shape: IndexList[rank], rtol: Float64 = 0.01) raises:
    print("== run_rms_norm_gpu")

    var cols = shape[rank - 1]
    var rows = shape.flattened_length() // cols

    var data_h = alloc[Scalar[dtype]](rows * cols)
    var res = alloc[Scalar[dtype]](rows * cols)
    var gamma_h = alloc[Scalar[dtype]](cols)

    rand[dtype](data_h, rows * cols)

    for i in range(cols):
        gamma_h[i] = (Float64(i + cols) / Float64(cols)).cast[dtype]()

    var data_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var gamma_d = ctx.enqueue_create_buffer[dtype](cols)

    var param_shape = Index(cols)

    var data_buf = TileTensor(data_d, row_major(Coord(shape)))
    var gamma = TileTensor(gamma_d, row_major(Coord(param_shape)))
    var epsilon = Scalar[dtype](0.001)
    var weight_offset = Scalar[dtype](0.0)

    ctx.enqueue_copy(data_d, data_h)
    ctx.enqueue_copy(gamma_d, gamma_h)

    @always_inline
    @__copy_capture(data_buf)
    @parameter
    fn input_fn[
        width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
        var idx = data_buf.layout(Coord(coords))
        return data_buf.ptr.load[width=width](idx)

    @always_inline
    @__copy_capture(data_buf)
    @parameter
    fn identity_output_fn[
        width: Int, alignment: Int
    ](coords: IndexList[rank], val: SIMD[dtype, width]) -> None:
        var idx = data_buf.layout(Coord(coords))
        data_buf.ptr.store[width=width, alignment=alignment](idx, val)

    rms_norm_gpu[input_fn, identity_output_fn, multiply_before_cast=True](
        shape, gamma, epsilon, weight_offset, ctx
    )
    ctx.enqueue_copy(res, data_d)
    ctx.synchronize()

    for r in range(rows):
        var vec = TileTensor(
            data_h + r * cols,
            row_major(Idx(cols)),
        )
        var rms_ref = compute_rms(vec, cols, epsilon)
        for c in range(cols):
            var idx = r * cols + c
            var val = (data_h[idx] / rms_ref) * (gamma_h[c] + weight_offset)
            assert_almost_equal(val, res[idx], rtol=rtol)

    _ = data_d
    _ = gamma_d

    data_h.free()
    res.free()
    gamma_h.free()


def main():
    with DeviceContext() as ctx:
        run_rms_norm_gpu[DType.float32](ctx, Index(5))
        run_rms_norm_gpu[DType.float32](ctx, Index(3, 4, 10, 20, 8))
        run_rms_norm_gpu[DType.float32](ctx, Index(1, 5, 6, 10, 128))
        run_rms_norm_gpu[DType.float32](ctx, Index(2, 5))
        run_rms_norm_gpu[DType.float32](ctx, Index(2, 55))
        run_rms_norm_gpu[DType.float32](ctx, Index(7, 557))
        run_rms_norm_gpu[DType.float32](ctx, Index(2, 8191))
        run_rms_norm_gpu[DType.float32](ctx, Index(2, 8192))
        run_rms_norm_gpu[DType.float32](ctx, Index(2, 16384))
        run_rms_norm_gpu[DType.float32](ctx, Index(2, 16385))
        run_rms_norm_gpu[DType.bfloat16](ctx, Index(3000, 32, 128), rtol=2e-2)
        run_rms_norm_gpu[DType.bfloat16](ctx, Index(2999, 31, 128), rtol=2e-2)

        # Test static shape dispatch.
        run_rms_norm_gpu[DType.bfloat16, static_cols=4096](
            ctx, Index(2, 4096), rtol=2e-2
        )
        run_rms_norm_gpu[DType.bfloat16, static_cols=16384](
            ctx, Index(2, 16384), rtol=2e-2
        )
