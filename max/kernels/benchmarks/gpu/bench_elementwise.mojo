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

from std.collections.string import StaticString
from std.math import erf, exp, rsqrt, log, sin, sqrt, tanh
from std.sys import (
    align_of,
    get_defined_string,
    simd_width_of,
    size_of,
)

from std.algorithm.functional import elementwise
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from std.gpu.host import DeviceContext, get_gpu_target
from std.gpu.host.info import _is_sm10x_gpu
from internal_utils import (
    arg_parse,
    parse_shape,
    CacheBustingBuffer,
    ScalarArray,
)

from std.utils import IndexList
from std.utils.index import product

from layout import TileTensor, Coord, row_major


def add_const_fn(x: SIMD) -> type_of(x):
    return x + 42


def copy_fn(x: SIMD) -> type_of(x):
    return x


def simd_sqrt(x: SIMD) -> type_of(x):
    return sqrt(x)


@no_inline
def run_elementwise[
    rank: Int,
    //,
    dtype: DType,
    kernel_fn: def[dtype: DType, width: Int](SIMD[dtype, width]) thin -> SIMD[
        dtype, width
    ],
](
    mut m: Bench,
    fn_name: StaticString,
    dims: IndexList[rank],
    *,
    name: StaticString,
    ctx: DeviceContext,
) raises:
    # Blackwell support 32B ld/st, see KERN-2037
    comptime pack_size = 32 // size_of[dtype]() if _is_sm10x_gpu(
        ctx.default_device_info
    ) else simd_width_of[dtype, target=get_gpu_target()]()
    comptime align = align_of[SIMD[dtype, pack_size], target=get_gpu_target()]()
    var N = product(dims, rank)

    # Cache busting buffers: sized to exceed 2x GPU cache.
    var cb_in = CacheBustingBuffer[dtype](N, pack_size, ctx)
    var cb_out = CacheBustingBuffer[dtype](N, pack_size, ctx)

    var in_host_ptr = ScalarArray[dtype](
        count=cb_in.alloc_size(), alignment=align
    )
    var out_host_ptr = ScalarArray[dtype](
        count=cb_out.alloc_size(), alignment=align
    )

    var in_host = TileTensor(in_host_ptr.unsafe_ptr(), row_major(Coord(dims)))
    var out_host = TileTensor(out_host_ptr.unsafe_ptr(), row_major(Coord(dims)))

    for i in range(cb_in.alloc_size()):
        in_host_ptr[i] = Scalar[dtype](i)

    ctx.enqueue_copy(cb_in.device_buffer(), in_host.ptr)

    @parameter
    @__copy_capture(cb_in, cb_out)
    @always_inline
    def bench_func(mut b: Bencher):
        @parameter
        @__copy_capture(N)
        @always_inline
        def kernel_launch(ctx: DeviceContext, iteration: Int) raises:
            var in_tensor = TileTensor(
                cb_in.offset_ptr(iteration), row_major(Coord(dims))
            )
            var out_tensor = TileTensor(
                cb_out.offset_ptr(iteration), row_major(Coord(dims))
            )

            @always_inline
            @__copy_capture(in_tensor, out_tensor)
            @parameter
            def func[
                simd_width: Int, rank_: Int, alignment: Int = 1
            ](idx0: IndexList[rank_]):
                var idx = rebind[IndexList[rank]](idx0)
                var coord = Coord(idx)
                comptime assert out_tensor.flat_rank >= coord.flat_rank
                comptime assert in_tensor.flat_rank >= coord.flat_rank

                out_tensor.store[alignment=align](
                    coord,
                    kernel_fn(
                        in_tensor.load[width=simd_width, alignment=align](coord)
                    ),
                )

            elementwise[func, pack_size, target="gpu"](
                dims,
                ctx,
            )

        b.iter_custom[kernel_launch](ctx)

    var num_bytes = 2 * N * size_of[dtype]()
    m.bench_function[bench_func](
        BenchId(
            "elementwise",
            input_id=String(
                "/",
                fn_name,
                "/",
                dtype,
                "/",
                name,
            ),
        ),
        [ThroughputMeasure(BenchMetric.bytes, num_bytes)],
    )

    ctx.synchronize()
    ctx.enqueue_copy(out_host.ptr, cb_out.device_buffer())

    _ = cb_in
    _ = cb_out
    _ = (in_host_ptr^, out_host_ptr^)


def list_to_static_tuple[x: List[Int]]() -> IndexList[len(x)]:
    var t = IndexList[len(x)]()

    comptime for i in range(len(x)):
        comptime xi = x[i]
        t[i] = xi
    return t


def main() raises:
    var op = arg_parse("op", "sqrt")
    comptime dtype = DType._from_str(
        get_defined_string["dtype", "DType.bfloat16"]()
    )
    comptime dims_str = get_defined_string["dims", "1x1024x3072"]()
    comptime dims = list_to_static_tuple[parse_shape[dims_str]()]()
    var m = Bench()
    with DeviceContext() as ctx:
        if op == "sqrt":
            run_elementwise[dtype, simd_sqrt](
                m, "sqrt", dims, name=dims_str, ctx=ctx
            )
        elif op == "rsqrt":
            run_elementwise[dtype, rsqrt](
                m, "rsqrt", dims, name=dims_str, ctx=ctx
            )
        elif op == "log":
            run_elementwise[dtype, log](m, "log", dims, name=dims_str, ctx=ctx)
        elif op == "sin":
            run_elementwise[dtype, sin](m, "sin", dims, name=dims_str, ctx=ctx)
        elif op == "tanh":
            run_elementwise[dtype, tanh](
                m, "tanh", dims, name=dims_str, ctx=ctx
            )
        elif op == "exp":
            run_elementwise[dtype, exp](m, "exp", dims, name=dims_str, ctx=ctx)
        elif op == "erf":
            run_elementwise[dtype, erf](m, "erf", dims, name=dims_str, ctx=ctx)
        elif op == "add_const":
            run_elementwise[dtype, add_const_fn](
                m, "add_const", dims, name=dims_str, ctx=ctx
            )
        elif op == "copy":
            run_elementwise[dtype, copy_fn](
                m, "copy", dims, name=dims_str, ctx=ctx
            )
    m.dump_report()
