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

from sys import align_of, env_get_int, env_get_string, simd_width_of
from sys.info import _TargetType

from algorithm._gpu.reduction import reduce_launch
from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from layout import LayoutTensor, Layout, RuntimeLayout
from buffer.dimlist import DimList
from gpu.host import DeviceContext, get_gpu_target
from internal_utils import (
    CacheBustingBuffer,
    arg_parse,
    env_get_shape,
    int_list_to_tuple,
    update_bench_config_args,
)
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from testing import assert_equal

from utils import IndexList, StaticTuple
from utils.index import product


fn align_of_simd[dtype: DType, simd_target: _TargetType]() -> Int:
    # TODO: move this utility function to a module.
    comptime pack_size = simd_width_of[dtype, target=simd_target]()
    return align_of[SIMD[dtype, pack_size]]()


fn run_reduce[
    reduce_fn: fn[dtype: DType, width: Int](
        SIMD[dtype, width], SIMD[dtype, width]
    ) capturing[_] -> SIMD[dtype, width],
    dtype: DType,
    rank: Int,
    num_reductions: Int = 1,
    cache_busting: Bool = True,
](mut m: Bench, shape: IndexList[rank], axis: Int, ctx: DeviceContext,) raises:
    print("run_reduce", shape)

    var out_shape = shape
    out_shape[axis] = 1
    comptime init: Scalar[dtype] = Scalar[dtype](0.0)
    comptime align = align_of_simd[dtype, simd_target = get_gpu_target()]()

    var in_size = shape.flattened_length()
    var out_size = in_size // shape[axis]

    var cb_in = CacheBustingBuffer[dtype](in_size, align, ctx, cache_busting)

    # Allocate & initialize host data
    var expected_vals = UnsafePointer[Scalar[dtype]].alloc(
        out_size, alignment=align
    )

    var in_host = UnsafePointer[Scalar[dtype]].alloc(cb_in.alloc_size())
    var res_host = UnsafePointer[Scalar[dtype]].alloc(out_size)

    for i in range(cb_in.alloc_size()):
        in_host[i] = 1

    # TODO: use reduce_fn to make this generic.
    for i in range(out_size):
        expected_vals[i] = Scalar[dtype](shape[axis]) * Scalar[dtype](1)

    var res_buffer = ctx.enqueue_create_buffer[dtype](in_size)

    comptime res_layout = Layout.row_major[rank]()
    var res_device = LayoutTensor[dtype, res_layout](
        res_buffer, RuntimeLayout[res_layout].row_major(out_shape)
    )

    ctx.enqueue_copy(cb_in.device_buffer(), in_host)

    @always_inline
    @parameter
    fn reduce_wrapper[
        dtype: DType, width: Int, reduction_idx: Int
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime assert reduction_idx < num_reductions, "invalid reduction idx"

        return reduce_fn[dtype, width](lhs, rhs)

    @__copy_capture(res_device)
    @parameter
    fn output_fn[
        _dtype: DType, width: Int, _rank: Int
    ](
        coords: IndexList[_rank],
        val: StaticTuple[SIMD[_dtype, width], num_reductions],
    ):
        res_device.store[width=width](
            rebind[IndexList[rank]](coords), rebind[SIMD[dtype, width]](val[0])
        )

    @__copy_capture(axis)
    @parameter
    @always_inline
    fn bench_func(mut b: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext, iteration: Int) raises:
            var input_lt = LayoutTensor[
                dtype, Layout.row_major[rank](), MutAnyOrigin
            ](
                cb_in.offset_ptr(iteration),
                RuntimeLayout[Layout.row_major[rank]()].row_major(shape),
            )

            @__copy_capture(input_lt)
            @parameter
            fn input_fn[
                dtype: DType,
                width: Int,
                _rank: Int,
            ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
                return rebind[SIMD[dtype, width]](
                    input_lt.load[width=width](rebind[IndexList[rank]](coords))
                )

            reduce_launch[
                num_reductions, input_fn, output_fn, reduce_wrapper, rank, dtype
            ](shape, axis, StaticTuple[_, num_reductions](init), ctx)

        b.iter_custom[kernel_launch](ctx)

    m.bench_function[bench_func](
        BenchId(
            "reduce",
            input_id=String(
                dtype,
                "/shape=",
                shape,
                "/axis=",
                axis,
                "/cache_busting=",
                cache_busting,
            ),
        ),
        [ThroughputMeasure(BenchMetric.elements, in_size)],
    )

    ctx.synchronize()
    ctx.enqueue_copy(res_host, res_buffer)

    for i in range(out_size):
        assert_equal(res_host[i], expected_vals[i])

    _ = cb_in
    _ = res_device

    in_host.free()
    res_host.free()
    expected_vals.free()


@parameter
fn reduce_add[
    dtype: DType,
    width: Int,
](x: SIMD[dtype, width], y: SIMD[dtype, width]) -> SIMD[dtype, width]:
    return x + y


def main():
    comptime dtype = DType._from_str(env_get_string["dtype", "DType.float16"]())

    comptime shape_in_list = env_get_shape["shape", "1x1x4096"]()
    comptime shape = int_list_to_tuple[shape_in_list]()
    comptime axis = env_get_int["axis", 1]()
    comptime cache_busting = True

    var m = Bench()
    with DeviceContext() as ctx:
        comptime dims = shape
        run_reduce[reduce_add, dtype, cache_busting=cache_busting](
            m,
            dims,
            axis,
            ctx,
        )

    m.dump_report()
