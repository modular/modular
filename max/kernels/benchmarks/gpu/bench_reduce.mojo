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

from std.sys import align_of, get_defined_int, get_defined_string, simd_width_of
from std.sys.info import CompilationTarget

from max.algorithm.backend.gpu.reduction import reduce_launch
from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from layout import Coord, TileTensor, row_major
from max.gpu.host import DeviceContext
from std.memory import Layout as AllocLayout, dealloc
from internal_utils import (
    CacheBustingBuffer,
    get_defined_shape,
    int_list_to_tuple,
)
from std.testing import assert_equal

from std.utils import IndexList, StaticTuple


def align_of_simd[dtype: DType, simd_target: CompilationTarget]() -> Int:
    # TODO: move this utility function to a module.
    comptime pack_size = simd_width_of[dtype, target=simd_target]()
    return align_of[SIMD[dtype, pack_size]]()


def run_reduce[
    ReduceFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[dtype: DType, width: SIMDLength](
        SIMD[dtype, width], SIMD[dtype, width]
    ) -> SIMD[dtype, width],
    //,
    dtype: DType,
    rank: Int,
    num_reductions: Int = 1,
    cache_busting: Bool = True,
    *,
    axis: Int,
](
    mut m: Bench,
    shape: IndexList[rank],
    reduce_fn: ReduceFnType,
    ctx: DeviceContext,
) raises:
    print("run_reduce", shape)

    var out_shape = shape
    out_shape[axis] = 1
    comptime init: Scalar[dtype] = Scalar[dtype](0.0)
    comptime align = align_of_simd[dtype, simd_target=ctx.target]()

    var in_size = shape.flattened_length()
    var out_size = in_size // shape[axis]

    var cb_in = CacheBustingBuffer[dtype](in_size, align, ctx, cache_busting)

    # Allocate & initialize host data
    var expected_vals_alloc = alloc(
        AllocLayout[Scalar[dtype], alignment=.of_bytes[align]()](
            length=out_size
        )
    ).into_managed()
    var expected_vals = expected_vals_alloc.unsafe_ptr()

    var in_host = List(length=cb_in.alloc_size(), fill=Scalar[dtype](1))
    var res_host = List(length=out_size, fill=Scalar[dtype](0))

    # TODO: use reduce_fn to make this generic.
    for i in range(out_size):
        expected_vals.unsafe_offset(i).write(
            Scalar[dtype](shape[axis]) * Scalar[dtype](1)
        )

    var res_buffer = ctx.enqueue_create_buffer[dtype](out_size)

    var res_device = TileTensor(res_buffer, row_major(out_size)).reshape(
        Coord(out_shape)
    )

    ctx.enqueue_copy(cb_in.device_buffer(), in_host)

    @inline(.always)
    def reduce_wrapper[
        dtype: DType, width: SIMDLength, reduction_idx: Int
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) {var reduce_fn} -> SIMD[
        dtype, width
    ]:
        comptime assert reduction_idx < num_reductions, "invalid reduction idx"

        return reduce_fn[dtype, width](lhs, rhs)

    def output_fn[
        _dtype: DType, width: SIMDLength
    ](coords: Coord, val: StaticTuple[SIMD[_dtype, width], num_reductions]) {
        var res_device
    }:
        res_device.store[alignment=align_of[dtype]()](
            coords,
            rebind[SIMD[dtype, width]](val[0]),
        )

    def kernel_launch(
        ctx: DeviceContext, iteration: Int
    ) raises {mut cb_in, mut res_device, imm}:
        var input_tile = TileTensor(
            cb_in.offset_ptr(iteration),
            row_major(Coord(shape)),
        )

        def input_fn[
            dtype: DType,
            width: Int,
        ](coords: Coord) {var input_tile} -> SIMD[dtype, width]:
            return rebind[SIMD[dtype, width]](
                input_tile.load[width=width, alignment=align_of[dtype]()](
                    coords
                )
            )

        reduce_launch[reduce_dim=axis](
            shape,
            StaticTuple[_, num_reductions](init),
            input_fn,
            output_fn,
            reduce_wrapper,
            ctx,
        )

    @inline(.always)
    def bench_func(mut b: Bencher) raises {imm}:
        bencher_iter_custom(b, kernel_launch, ctx)

    m.bench_function(
        bench_func,
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
        assert_equal(res_host[i], expected_vals[unsafe_offset=i])

    _ = cb_in
    _ = res_device

    dealloc(expected_vals_alloc^)
    _ = in_host^
    _ = res_host^


def reduce_add[
    dtype: DType,
    width: SIMDLength,
](x: SIMD[dtype, width], y: SIMD[dtype, width]) -> SIMD[dtype, width]:
    return x + y


def main() raises:
    comptime dtype = DType._from_str(
        get_defined_string["dtype", "DType.float16"]()
    ).value()

    comptime shape_in_list = get_defined_shape["shape", "1x1x4096"]()
    comptime shape = int_list_to_tuple[shape_in_list]()
    comptime axis = get_defined_int["axis", 1]()
    comptime cache_busting = True

    var m = Bench()
    with DeviceContext() as ctx:
        comptime dims = shape
        run_reduce[dtype, cache_busting=cache_busting, axis=axis](
            m,
            dims,
            reduce_add,
            ctx,
        )

    m.dump_report()
