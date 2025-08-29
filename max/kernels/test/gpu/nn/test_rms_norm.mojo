# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from buffer import NDBuffer
from gpu.host import DeviceContext
from nn.normalization import *
from testing import assert_almost_equal

from utils.index import Index, IndexList


fn compute_rms[
    dtype: DType
](data: NDBuffer[dtype, 1], size: Int, eps: Scalar[dtype]) -> Scalar[dtype]:
    var sum_of_squares = Scalar[dtype]()
    for i in range(size):
        sum_of_squares += data[i] * data[i]
    return sqrt((sum_of_squares / len(data)) + eps).cast[dtype]()


fn run_rms_norm_gpu[
    dtype: DType, rank: Int
](ctx: DeviceContext, shape: IndexList[rank], rtol: Float64 = 0.01) raises:
    print("== run_rms_norm_gpu")

    var cols = shape[rank - 1]
    var rows = shape.flattened_length() // cols

    var data_h = UnsafePointer[Scalar[dtype]].alloc(rows * cols)
    var res = UnsafePointer[Scalar[dtype]].alloc(rows * cols)
    var gamma_h = UnsafePointer[Scalar[dtype]].alloc(cols)

    for i in range(rows * cols):
        var val = Scalar[dtype](i)
        data_h[i] = val

    for i in range(cols):
        gamma_h[i] = ((i + cols) / cols).cast[dtype]()

    var data_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var gamma_d = ctx.enqueue_create_buffer[dtype](cols)

    var param_shape = Index(cols)

    var data_buf = NDBuffer[dtype, rank](data_d.unsafe_ptr(), shape)
    var gamma = NDBuffer[dtype, 1](gamma_d.unsafe_ptr(), param_shape)
    var epsilon = Scalar[dtype](0.001)
    var weight_offset = Scalar[dtype](0.0)

    ctx.enqueue_copy(data_d, data_h)
    ctx.enqueue_copy(gamma_d, gamma_h)

    @__copy_capture(data_buf)
    @always_inline
    @parameter
    fn input_fn[
        width: Int, _rank: Int
    ](idx: IndexList[_rank]) -> SIMD[dtype, width]:
        return data_buf.load[width=width](rebind[IndexList[rank]](idx))

    @always_inline
    @__copy_capture(data_buf)
    @parameter
    fn identity_output_fn[
        width: Int, alignment: Int
    ](idx: IndexList[rank], val: SIMD[dtype, width]) -> None:
        data_buf.store[width=width, alignment=alignment](idx, val)

    rms_norm_gpu[input_fn, identity_output_fn, multiply_before_cast=True](
        shape, gamma, epsilon, weight_offset, ctx
    )
    ctx.enqueue_copy(res, data_d)
    ctx.synchronize()

    for r in range(rows):
        var vec = NDBuffer[dtype, 1](data_h + r * cols, cols)
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
