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

from sys import alignof, env_get_int, env_get_string, simdwidthof

from algorithm._gpu.reduction import reduce_launch
from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from buffer import NDBuffer
from buffer.dimlist import DimList, _make_tuple
from gpu.host import DeviceContext
from gpu.host._compile import get_gpu_target
from internal_utils import DeviceNDBuffer
from testing import assert_equal

from utils import IndexList, StaticTuple
from utils.index import product


fn alignof_simd[type: DType, simd_target: __mlir_type.`!kgen.target`]() -> Int:
    # TODO: move this utility function to a module.
    alias pack_size = simdwidthof[type, target=simd_target]()
    return alignof[SIMD[type, pack_size]]()


fn run_reduce[
    reduce_fn: fn[type: DType, width: Int] (
        SIMD[type, width], SIMD[type, width]
    ) capturing [_] -> SIMD[type, width],
    type: DType,
    rank: Int,
    num_reductions: Int = 1,
](mut m: Bench, shape: IndexList[rank], ctx: DeviceContext,) raises:
    print("run_reduce", shape)
    var axis = rank - 1
    var out_shape = shape
    out_shape[axis] = 1
    alias init: Scalar[type] = Scalar[type](0.0)

    var in_size = shape.flattened_length()
    var out_size = product(shape, rank - 1)

    alias align = alignof_simd[type, simd_target = get_gpu_target()]()
    var expected_vals = UnsafePointer[Scalar[type], alignment=align].alloc(
        out_size
    )

    var in_host = UnsafePointer[Scalar[type]].alloc(in_size)
    var res_host = UnsafePointer[Scalar[type]].alloc(out_size)

    for i in range(in_size):
        in_host[i] = (i // shape[axis]) + 1

    # TODO: use reduce_fn to make this generic.
    for i in range(out_size):
        expected_vals[i] = shape[axis] * Scalar[type](i + 1)

    var vec_device = DeviceNDBuffer[type, rank](shape, ctx=ctx)
    var res_device = DeviceNDBuffer[type, rank](out_shape, ctx=ctx)
    var input_buf_device = vec_device.tensor
    var output_buf_device = res_device.tensor

    ctx.enqueue_copy(vec_device.buffer, in_host)

    @always_inline
    @parameter
    fn reduce_wrapper[
        type: DType, width: Int, reduction_idx: Int
    ](lhs: SIMD[type, width], rhs: SIMD[type, width]) -> SIMD[type, width]:
        constrained[reduction_idx < num_reductions, "invalid reduction idx"]()

        return reduce_fn[type, width](lhs, rhs)

    @__copy_capture(input_buf_device)
    @parameter
    fn input_fn[
        type: DType,
        width: Int,
        _rank: Int,
    ](coords: IndexList[_rank]) -> SIMD[type, width]:
        return rebind[SIMD[type, width]](
            input_buf_device.load[width=width](rebind[IndexList[rank]](coords))
        )

    @__copy_capture(output_buf_device)
    @parameter
    fn output_fn[
        _type: DType, width: Int, _rank: Int
    ](
        coords: IndexList[_rank],
        val: StaticTuple[SIMD[_type, width], num_reductions],
    ):
        output_buf_device[rebind[IndexList[rank]](coords)] = rebind[
            Scalar[type]
        ](val[0])

    @__copy_capture(axis)
    @parameter
    @always_inline
    fn bench_func(mut b: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            reduce_launch[
                num_reductions, input_fn, output_fn, reduce_wrapper, rank, type
            ](shape, axis, init, ctx)

        b.iter_custom[kernel_launch](ctx)

    m.bench_function[bench_func](
        BenchId("reduce", input_id=String(type, "/shape=", shape)),
        ThroughputMeasure(BenchMetric.elements, in_size),
    )

    ctx.synchronize()
    ctx.enqueue_copy(res_host, res_device.buffer)

    for i in range(out_size):
        assert_equal(res_host[i], expected_vals[i])

    _ = vec_device
    _ = res_device

    in_host.free()
    res_host.free()


@parameter
fn reduce_add[
    type: DType,
    width: Int,
](x: SIMD[type, width], y: SIMD[type, width]) -> SIMD[type, width]:
    return x + y


def main():
    alias dtype = DType._from_str(env_get_string["dtype", "DType.bfloat16"]())

    # DimList(1, 1, 4096) baby-llama-LPTG-kernels
    alias M = env_get_int["M", 1]()
    alias N = env_get_int["N", 1]()
    alias K = env_get_int["K", 4096]()

    var m = Bench()
    with DeviceContext() as ctx:
        alias dims = IndexList[3](M, N, K)
        run_reduce[reduce_add, dtype](
            m,
            dims,
            ctx,
        )

    m.dump_report()
