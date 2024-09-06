# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-build %s

from math import sqrt, isqrt, log, sin, tanh, exp, erf, fma, ceildiv, align_up
from sys import alignof, sizeof, triple_is_nvidia_cuda, simdwidthof, env_get_int

from algorithm.functional import elementwise
from buffer import DimList, NDBuffer
from buffer.dimlist import _make_tuple
from gpu.host.device_context import DeviceContext, DeviceBuffer
from gpu.host._compile import _get_nvptx_target
from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from internal_utils import DeviceNDBuffer
from utils import StaticIntTuple
from utils.index import product
from sys.intrinsics import strided_load
from buffer.buffer import _compute_ndbuffer_offset


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
        var v = buffer.data.bitcast[DType.uint8]().load[width=simd_width](index)
        return v.cast[buffer.type]()
    return buffer.data.load[width=simd_width](index)


@always_inline
fn simd_load[
    simd_width: Int
](
    buffer: NDBuffer,
    index: StaticIntTuple[buffer.rank],
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
            buffer.data.bitcast[DType.uint8]().offset(flat_index),
            stride,
        )
        return v.cast[buffer.type]()
    return strided_load[simd_width](buffer.data.offset(flat_index), stride)


@always_inline
fn simd_store[
    simd_width: Int
](
    buffer: NDBuffer,
    index: StaticIntTuple[buffer.rank],
    val: SIMD[buffer.type, simd_width],
):
    var flat_index = _compute_ndbuffer_offset(buffer, index)

    # We have to cast bools into their runtime storage type.
    @parameter
    if buffer.type is DType.bool:
        var v = val.cast[DType.uint8]()
        buffer.data.bitcast[DType.uint8]().store[width=simd_width](
            flat_index, v
        )
    else:
        buffer.data.store[width=simd_width](flat_index, val)


@no_inline
fn run_elementwise[
    rank: Int, //,
    type: DType,
    kernel_fn: fn[type: DType, width: Int] (SIMD[type, width]) -> SIMD[
        type, width
    ],
    *,
    emulate_graph_compiler: Bool,
    use_aligned_memory: Bool,
](
    inout m: Bench,
    fn_name: String,
    dims: StaticIntTuple[rank],
    *,
    name: String,
    ctx: DeviceContext,
) raises:
    alias pack_size = simdwidthof[type, target = _get_nvptx_target()]()
    alias align = alignof[
        SIMD[type, pack_size], target = _get_nvptx_target()
    ]() if use_aligned_memory else 1
    var N = product(dims, rank)

    # Choose a size larger than the two times the L2 cache
    # 128 MiB is larger that twice the L2 cache on the A100, A10, and L4.
    var stride = align_up(N, pack_size)
    var N_cache = align_up(
        128 * 1024 * 1024, stride * sizeof[type]()
    ) // sizeof[type]()

    var in_host_ptr = UnsafePointer[Scalar[type], alignment=align].alloc(
        N_cache
    )
    var out_host_ptr = UnsafePointer[Scalar[type], alignment=align].alloc(
        N_cache
    )

    var in_host = NDBuffer[type, rank](in_host_ptr, dims)
    var out_host = NDBuffer[type, rank](out_host_ptr, dims)

    for i in range(N_cache):
        in_host_ptr[i] = i

    var in_buffer = ctx.create_buffer[type](N_cache)
    var out_buffer = ctx.create_buffer[type](N_cache)

    ctx.enqueue_copy_to_device(in_buffer, in_host.data)

    @parameter
    @__copy_capture(stride, N_cache)
    @always_inline
    fn bench_func(inout b: Bencher):
        @parameter
        @__copy_capture(N, stride)
        @always_inline
        fn kernel_launch(ctx: DeviceContext, iteration: Int) raises:
            # cycle through chunks of N_cache to ensure the tensor is not in the cache each iteration
            var offset = (iteration * stride) % N_cache
            var in_tensor = NDBuffer[type, rank](in_buffer.ptr + offset, dims)
            var out_tensor = NDBuffer[type, rank](out_buffer.ptr + offset, dims)

            @always_inline
            @__copy_capture(in_tensor, out_tensor)
            @parameter
            fn func[simd_width: Int, rank_: Int](idx0: StaticIntTuple[rank_]):
                var idx = rebind[StaticIntTuple[rank]](idx0)

                @parameter
                if emulate_graph_compiler:
                    """In this mode we use the simd_store / simd_load that are copied
                    from MOGG.mojo. This is used to emulate what the graph compiler
                    would generate for the elementwise operations."""
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

            elementwise[func, pack_size, target="cuda"](
                dims,
                ctx,
            )

        b.iter_custom[kernel_launch](ctx)

    var num_bytes = 2 * N * sizeof[type]()
    m.bench_function[bench_func](
        BenchId(
            "elementwise",
            input_id="/"
            + ("aligned" if use_aligned_memory else "unaligned")
            + ("/graph_compiler_emulated" if emulate_graph_compiler else "")
            + "/"
            + fn_name
            + "/"
            + str(type)
            + "/"
            + name,
        ),
        ThroughputMeasure(BenchMetric.bytes, num_bytes),
    )

    ctx.synchronize()
    ctx.enqueue_copy_from_device(out_host.data, out_buffer)

    _ = in_buffer
    _ = out_buffer
    in_host_ptr.free()
    out_host_ptr.free()


fn main() raises:
    var m = Bench()

    # TODO: expand to all the params
    alias phony = env_get_int["phony", 1]()
    constrained[phony == 1]()

    alias types = List[DType](DType.bfloat16, DType.float32)

    var shape_list = List[DimList](
        DimList(1, 1024, 3072),  # baby-replit-CE-kernels
        DimList(1, 8, 3, 1025, 128),  # baby-replit-TG-kernels
    )

    with DeviceContext() as ctx:
        for j in range(len(shape_list)):
            var dims = StaticIntTuple[1](shape_list[j].product().get())

            @parameter
            for i in range(len(types)):

                @parameter
                for aligned_memory_config in range(2):

                    @parameter
                    for emulate_graph_compiler in range(2):

                        @parameter
                        if emulate_graph_compiler and aligned_memory_config:
                            # The graph compiler simd_load and store are not
                            # compatible with aligned load/store since it
                            # does a dynamic check on the stride.
                            continue

                        run_elementwise[
                            types[i],
                            simd_sqrt,
                            use_aligned_memory = aligned_memory_config != 0,
                            emulate_graph_compiler = emulate_graph_compiler
                            != 0,
                        ](m, "sqrt", dims, name=str(shape_list[j]), ctx=ctx)
                        run_elementwise[
                            types[i],
                            isqrt,
                            use_aligned_memory = aligned_memory_config != 0,
                            emulate_graph_compiler = emulate_graph_compiler
                            != 0,
                        ](
                            m,
                            "isqrt",
                            dims,
                            name=str(shape_list[j]),
                            ctx=ctx,
                        )
                        run_elementwise[
                            types[i],
                            log,
                            use_aligned_memory = aligned_memory_config != 0,
                            emulate_graph_compiler = emulate_graph_compiler
                            != 0,
                        ](m, "log", dims, name=str(shape_list[j]), ctx=ctx)
                        run_elementwise[
                            types[i],
                            sin,
                            use_aligned_memory = aligned_memory_config != 0,
                            emulate_graph_compiler = emulate_graph_compiler
                            != 0,
                        ](m, "sin", dims, name=str(shape_list[j]), ctx=ctx)
                        run_elementwise[
                            types[i],
                            tanh,
                            use_aligned_memory = aligned_memory_config != 0,
                            emulate_graph_compiler = emulate_graph_compiler
                            != 0,
                        ](m, "tanh", dims, name=str(shape_list[j]), ctx=ctx)
                        run_elementwise[
                            types[i],
                            exp,
                            use_aligned_memory = aligned_memory_config != 0,
                            emulate_graph_compiler = emulate_graph_compiler
                            != 0,
                        ](m, "exp", dims, name=str(shape_list[j]), ctx=ctx)
                        run_elementwise[
                            types[i],
                            erf,
                            use_aligned_memory = aligned_memory_config != 0,
                            emulate_graph_compiler = emulate_graph_compiler
                            != 0,
                        ](m, "erf", dims, name=str(shape_list[j]), ctx=ctx)
                        run_elementwise[
                            types[i],
                            add_const_fn,
                            use_aligned_memory = aligned_memory_config != 0,
                            emulate_graph_compiler = emulate_graph_compiler
                            != 0,
                        ](
                            m,
                            "add_const",
                            dims,
                            name=str(shape_list[j]),
                            ctx=ctx,
                        )
                        run_elementwise[
                            types[i],
                            copy_fn,
                            use_aligned_memory = aligned_memory_config != 0,
                            emulate_graph_compiler = emulate_graph_compiler
                            != 0,
                        ](m, "copy", dims, name=str(shape_list[j]), ctx=ctx)
    m.dump_report()
