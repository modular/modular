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

from random import random_float64
from sys import env_get_dtype

from benchmark import Bench, BenchConfig, Bencher, BenchId
from buffer import NDBuffer
from gpu.host import DeviceContext
from internal_utils import env_get_shape, int_list_to_tuple
from nn.normalization import layer_norm_gpu, rms_norm_gpu

from utils.index import IndexList


fn bench_layer_norm_gpu[
    dtype: DType, rank: Int
](
    ctx: DeviceContext,
    mut b: Bench,
    fn_name: String,
    shape: IndexList[rank],
) raises:
    var cols = shape[rank - 1]
    var rows = shape.flattened_length() // cols

    var data_h = UnsafePointer[Scalar[dtype]].alloc(rows * cols)
    var res = UnsafePointer[Scalar[dtype]].alloc(rows * cols)
    var gamma_h = UnsafePointer[Scalar[dtype]].alloc(cols)
    var beta_h = UnsafePointer[Scalar[dtype]].alloc(cols)

    for i in range(rows * cols):
        var val = Scalar[dtype](random_float64(0, 100).cast[dtype]())
        data_h[i] = val

    for i in range(cols):
        gamma_h[i] = ((i + cols) / cols).cast[dtype]()
        beta_h[i] = (i / cols).cast[dtype]()

    var data_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var gamma_d = ctx.enqueue_create_buffer[dtype](cols)
    var beta_d = ctx.enqueue_create_buffer[dtype](cols)

    var param_shape = IndexList[1](cols)

    var data_buf = NDBuffer[dtype, rank](data_d.unsafe_ptr(), shape)
    var gamma = NDBuffer[dtype, 1](gamma_d.unsafe_ptr(), param_shape)
    var beta = NDBuffer[dtype, 1](beta_d.unsafe_ptr(), param_shape)
    var epsilon = Scalar[dtype]()

    ctx.enqueue_copy(data_d, data_h)
    ctx.enqueue_copy(gamma_d, gamma_h)
    ctx.enqueue_copy(beta_d, beta_h)

    @__copy_capture(data_buf)
    @always_inline
    @parameter
    fn input_fn[
        width: Int, _rank: Int
    ](idx: IndexList[_rank]) -> SIMD[dtype, width]:
        return data_buf.load[width=width](rebind[IndexList[rank]](idx))

    @__copy_capture(gamma)
    @always_inline
    @parameter
    fn gamma_fn[
        width: Int, rank: Int
    ](idx: IndexList[rank]) -> SIMD[dtype, width]:
        return gamma.load[width=width](idx[0])

    @always_inline
    @__copy_capture(shape, beta, epsilon, data_buf)
    @parameter
    fn bench_fn(mut b: Bencher) raises:
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            layer_norm_gpu[input_fn, gamma_fn](
                shape, beta, epsilon, data_buf, ctx=ctx
            )

        b.iter_custom[kernel_launch](ctx)

    b.bench_function[bench_fn](
        BenchId("layer_norm", input_id=String(fn_name, dtype, shape, sep="/"))
    )

    ctx.synchronize()

    _ = data_d
    _ = gamma_d
    _ = beta_d

    data_h.free()
    res.free()
    gamma_h.free()
    beta_h.free()


fn bench_rms_norm_gpu[
    dtype: DType, rank: Int
](
    ctx: DeviceContext,
    mut b: Bench,
    fn_name: String,
    shape: IndexList[rank],
) raises:
    var cols = shape[rank - 1]
    var rows = shape.flattened_length() // cols

    var data_h = UnsafePointer[Scalar[dtype]].alloc(rows * cols)
    var res = UnsafePointer[Scalar[dtype]].alloc(rows * cols)
    var gamma_h = UnsafePointer[Scalar[dtype]].alloc(cols)

    for i in range(rows * cols):
        var val = Scalar[dtype](random_float64(0, 100).cast[dtype]())
        data_h[i] = val

    for i in range(cols):
        gamma_h[i] = ((i + cols) / cols).cast[dtype]()

    var data_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var gamma_d = ctx.enqueue_create_buffer[dtype](cols)

    var param_shape = IndexList[1](cols)

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
        width: Int
    ](idx: IndexList[rank], val: SIMD[dtype, width]) -> None:
        data_buf.store(idx, val)

    @always_inline
    @__copy_capture(shape, gamma, epsilon, weight_offset)
    @parameter
    fn bench_fn(mut b: Bencher) raises:
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            rms_norm_gpu[
                input_fn, identity_output_fn, multiply_before_cast=True
            ](shape, gamma, epsilon, weight_offset, ctx)

        b.iter_custom[kernel_launch](ctx)

    b.bench_function[bench_fn](
        BenchId("rms_norm", input_id=String(fn_name, "/", dtype, "/", shape)),
    )

    ctx.synchronize()

    _ = data_d
    _ = gamma_d

    data_h.free()
    res.free()
    gamma_h.free()


def main():
    alias dtype = env_get_dtype["dtype", DType.bfloat16]()
    alias shape = int_list_to_tuple[env_get_shape["shape", "256x256"]()]()

    var m = Bench(BenchConfig(num_repetitions=1))
    with DeviceContext() as ctx:

        @parameter
        if len(shape) == 2:
            bench_layer_norm_gpu[dtype](ctx, m, "layer_norm_gpu", shape)
        elif len(shape) == 3:
            bench_rms_norm_gpu[dtype](ctx, m, "rms_norm_gpu", shape)

    m.dump_report()
