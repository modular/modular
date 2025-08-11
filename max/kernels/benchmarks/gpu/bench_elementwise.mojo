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

from collections.string import StaticString
from math import align_up, erf, exp, isqrt, log, sin, sqrt, tanh
from sys import (
    alignof,
    env_get_int,
    env_get_string,
    simdwidthof,
    sizeof,
)
from sys.intrinsics import strided_load

from algorithm.functional import elementwise
from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from buffer import NDBuffer
from buffer.buffer import _compute_ndbuffer_offset
from gpu.host import DeviceContext
from gpu.host import get_gpu_target
from internal_utils import arg_parse, parse_shape

from utils import IndexList
from utils.index import product


fn add_const_fn(x: SIMD) -> __type_of(x):
    return x + 42


fn copy_fn(x: SIMD) -> __type_of(x):
    return x


fn simd_sqrt(x: SIMD) -> __type_of(x):
    return sqrt(x)


@always_inline
fn _simd_load_internal[
    simd_width: Int
](buffer: NDBuffer, index: Int) -> SIMD[buffer.type, simd_width]:
    @parameter
    if buffer.type is DType.bool:
        var v = buffer.data.bitcast[UInt8]().load[width=simd_width](index)
        return v.cast[buffer.type]()
    return buffer.data.load[width=simd_width](index)


@always_inline
fn simd_load[
    simd_width: Int
](
    buffer: NDBuffer,
    index: IndexList[buffer.rank],
) -> SIMD[
    buffer.type, simd_width
]:
    var flat_index = _compute_ndbuffer_offset(buffer, index)

    if buffer.is_contiguous():
        return _simd_load_internal[simd_width](buffer, flat_index)

    var stride = buffer.stride[buffer.rank - 1]()
    if stride == 0:
        return buffer.data.load(flat_index)

    if buffer.type is DType.bool:
        var v = strided_load[simd_width](
            buffer.data.bitcast[UInt8]().offset(flat_index),
            stride,
        )
        return v.cast[buffer.type]()
    return strided_load[simd_width](buffer.data.offset(flat_index), stride)


@always_inline
fn simd_store[
    simd_width: Int
](
    buffer: NDBuffer,
    index: IndexList[buffer.rank],
    val: SIMD[buffer.type, simd_width],
):
    var flat_index = _compute_ndbuffer_offset(buffer, index)

    # We have to cast bools into their runtime storage type.
    @parameter
    if buffer.type is DType.bool:
        buffer.data.bitcast[UInt8]().store(flat_index, val.cast[DType.uint8]())
    else:
        buffer.data.store(flat_index, val)


@no_inline
fn run_elementwise[
    rank: Int, //,
    dtype: DType,
    kernel_fn: fn[dtype: DType, width: Int] (SIMD[dtype, width]) -> SIMD[
        dtype, width
    ],
    *,
    emulate_graph_compiler: Bool,
    use_aligned_memory: Bool,
](
    mut m: Bench,
    fn_name: StaticString,
    dims: IndexList[rank],
    *,
    name: StaticString,
    ctx: DeviceContext,
) raises:
    alias pack_size = simdwidthof[dtype, target = get_gpu_target()]()
    alias align = alignof[
        SIMD[dtype, pack_size], target = get_gpu_target()
    ]() if use_aligned_memory else 1
    var N = product(dims, rank)

    # Choose a size larger than the two times the L2 cache
    # 128 MiB is larger that twice the L2 cache on the A100, A10, and L4.
    var stride = align_up(N, pack_size)
    var N_cache = (
        align_up(128 * 1024 * 1024, stride * sizeof[dtype]()) // sizeof[dtype]()
    )

    var in_host_ptr = UnsafePointer[Scalar[dtype], alignment=align].alloc(
        N_cache
    )
    var out_host_ptr = UnsafePointer[Scalar[dtype], alignment=align].alloc(
        N_cache
    )

    var in_host = NDBuffer[dtype, rank](in_host_ptr, dims)
    var out_host = NDBuffer[dtype, rank](out_host_ptr, dims)

    for i in range(N_cache):
        in_host_ptr[i] = i

    var in_buffer = ctx.enqueue_create_buffer[dtype](N_cache)
    var out_buffer = ctx.enqueue_create_buffer[dtype](N_cache)

    ctx.enqueue_copy(in_buffer, in_host.data)

    @parameter
    @__copy_capture(stride, N_cache)
    @always_inline
    fn bench_func(mut b: Bencher):
        @parameter
        @__copy_capture(N, stride)
        @always_inline
        fn kernel_launch(ctx: DeviceContext, iteration: Int) raises:
            # cycle through chunks of N_cache to ensure the tensor is not in the cache each iteration
            var offset = (iteration * stride) % N_cache
            var in_tensor = NDBuffer[dtype, rank](
                in_buffer.unsafe_ptr() + offset, dims
            )
            var out_tensor = NDBuffer[dtype, rank](
                out_buffer.unsafe_ptr() + offset, dims
            )

            @always_inline
            @__copy_capture(in_tensor, out_tensor)
            @parameter
            fn func[
                simd_width: Int, rank_: Int, alignment: Int = 1
            ](idx0: IndexList[rank_]):
                var idx = rebind[IndexList[rank]](idx0)

                @parameter
                if emulate_graph_compiler:
                    # In this mode we use the simd_store / simd_load that are copied
                    # from MOGG.mojo. This is used to emulate what the graph compiler
                    # would generate for the elementwise operations.
                    simd_store(
                        out_tensor,
                        idx,
                        kernel_fn(simd_load[simd_width](in_tensor, idx)),
                    )
                else:
                    out_tensor.store[alignment=align](
                        idx,
                        kernel_fn(
                            in_tensor.load[width=simd_width, alignment=align](
                                idx
                            )
                        ),
                    )

            elementwise[func, pack_size, target="gpu"](
                dims,
                ctx,
            )

        b.iter_custom[kernel_launch](ctx)

    var num_bytes = 2 * N * sizeof[dtype]()
    m.bench_function[bench_func](
        BenchId(
            "elementwise",
            input_id=String(
                "/",
                "aligned" if use_aligned_memory else "unaligned",
                "/graph_compiler_emulated" if emulate_graph_compiler else "",
                "/",
                fn_name,
                "/",
                dtype,
                "/",
                name,
            ),
        ),
        ThroughputMeasure(BenchMetric.bytes, num_bytes),
    )

    ctx.synchronize()
    ctx.enqueue_copy(out_host.data, out_buffer)

    _ = in_buffer
    _ = out_buffer
    in_host_ptr.free()
    out_host_ptr.free()


fn list_to_static_tuple[x: List[Int]]() -> IndexList[len(x)]:
    var t = IndexList[len(x)]()

    @parameter
    for i in range(len(x)):
        t[i] = x[i]
    return t


fn main() raises:
    var op = arg_parse("op", "sqrt")
    alias dtype = DType._from_str(env_get_string["dtype", "DType.bfloat16"]())
    alias rank = env_get_int["rank", 3]()
    alias dims_str = env_get_string["dims", "1x1024x3072"]()
    alias dims = list_to_static_tuple[parse_shape[dims_str]()]()
    alias aligned_memory_config = env_get_int[
        "aligned_memory_config", 0
    ]()  # bool
    alias emulate_graph_compiler = env_get_int[
        "emulate_graph_compiler", 0
    ]()  # bool

    var m = Bench()
    with DeviceContext() as ctx:

        @parameter
        if emulate_graph_compiler and aligned_memory_config:
            # The graph compiler simd_load and store are not
            # compatible with aligned load/store since it
            # does a dynamic check on the stride.
            return

        if op == "sqrt":
            run_elementwise[
                dtype,
                simd_sqrt,
                use_aligned_memory = aligned_memory_config != 0,
                emulate_graph_compiler = emulate_graph_compiler != 0,
            ](m, "sqrt", dims, name=dims_str, ctx=ctx)

        elif op == "isqrt":
            run_elementwise[
                dtype,
                isqrt,
                use_aligned_memory = aligned_memory_config != 0,
                emulate_graph_compiler = emulate_graph_compiler != 0,
            ](
                m,
                "isqrt",
                dims,
                name=dims_str,
                ctx=ctx,
            )

        elif op == "log":
            run_elementwise[
                dtype,
                log,
                use_aligned_memory = aligned_memory_config != 0,
                emulate_graph_compiler = emulate_graph_compiler != 0,
            ](m, "log", dims, name=dims_str, ctx=ctx)

        elif op == "sin":
            run_elementwise[
                dtype,
                sin,
                use_aligned_memory = aligned_memory_config != 0,
                emulate_graph_compiler = emulate_graph_compiler != 0,
            ](m, "sin", dims, name=dims_str, ctx=ctx)

        elif op == "tanh":
            run_elementwise[
                dtype,
                tanh,
                use_aligned_memory = aligned_memory_config != 0,
                emulate_graph_compiler = emulate_graph_compiler != 0,
            ](m, "tanh", dims, name=dims_str, ctx=ctx)

        elif op == "exp":
            run_elementwise[
                dtype,
                exp,
                use_aligned_memory = aligned_memory_config != 0,
                emulate_graph_compiler = emulate_graph_compiler != 0,
            ](m, "exp", dims, name=dims_str, ctx=ctx)

        elif op == "erf":
            run_elementwise[
                dtype,
                erf,
                use_aligned_memory = aligned_memory_config != 0,
                emulate_graph_compiler = emulate_graph_compiler != 0,
            ](m, "erf", dims, name=dims_str, ctx=ctx)

        elif op == "add_const":
            run_elementwise[
                dtype,
                add_const_fn,
                use_aligned_memory = aligned_memory_config != 0,
                emulate_graph_compiler = emulate_graph_compiler != 0,
            ](
                m,
                "add_const",
                dims,
                name=dims_str,
                ctx=ctx,
            )

        elif op == "copy":
            run_elementwise[
                dtype,
                copy_fn,
                use_aligned_memory = aligned_memory_config != 0,
                emulate_graph_compiler = emulate_graph_compiler != 0,
            ](m, "copy", dims, name=dims_str, ctx=ctx)
    m.dump_report()
