# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s -t

from math import sqrt, rsqrt, log, sin, tanh, exp, erf, fma
from sys.info import triple_is_nvidia_cuda

from algorithm.functional import _elementwise_impl_gpu
from buffer import DimList, NDBuffer
from buffer.dimlist import _make_tuple
from gpu.host.device_context import DeviceContext
from gpu.host._compile import _get_nvptx_target
from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from internal_utils import DeviceNDBuffer
from utils.index import product
from sys.intrinsics import strided_load


fn add_const_fn(x: SIMD) -> __type_of(x):
    return x + 42


fn copy_fn(x: SIMD) -> __type_of(x):
    return x


fn simd_sqrt(x: SIMD) -> __type_of(x):
    return sqrt(x)


@always_inline
fn _compute_flat_index[
    type: DType, rank: Int, iters: Int, static_shape: DimList
](
    buffer: NDBuffer[type, rank, static_shape],
    index: StaticIntTuple[rank],
) -> Int:
    var flat_index: Int = 0

    @parameter
    for i in range(iters):
        flat_index = fma(index[i], buffer.dynamic_stride[i], flat_index)

    return flat_index


@always_inline
fn _simd_load_internal[
    simd_width: Int, type: DType, rank: Int, static_shape: DimList
](buffer: NDBuffer[type, rank, static_shape], index: Int) -> SIMD[
    type, simd_width
]:
    @parameter
    if type is DType.bool:
        var v = SIMD[size=simd_width].load(
            buffer.data.bitcast[DType.uint8](), index
        )
        return v.cast[type]()
    else:
        return SIMD[size=simd_width].load(buffer.data, index)


@always_inline
fn simd_load[
    type: DType, simd_width: Int, rank: Int, input_0_static_shape: DimList
](
    buffer: NDBuffer[type, rank, input_0_static_shape],
    index: StaticIntTuple[rank],
) -> SIMD[type, simd_width]:
    var flat_index = _compute_flat_index[
        type, rank, rank, input_0_static_shape
    ](buffer, index)
    var stride = buffer.dynamic_stride[rank - 1]

    if stride != 0:
        return Scalar.load(buffer.data, flat_index)
    elif stride > 1:

        @parameter
        if type is DType.bool:
            var v = strided_load[DType.uint8, simd_width](
                buffer.data.bitcast[DType.uint8]().offset(flat_index),
                stride,
            )
            return v.cast[type]()
        else:
            return strided_load[type, simd_width](
                buffer.data.offset(flat_index), stride
            )
    return _simd_load_internal[simd_width, type, rank, input_0_static_shape](
        buffer, flat_index
    )


@always_inline
fn simd_store[
    type: DType, simd_width: Int, rank: Int
](
    buffer: NDBuffer[type, rank],
    index: StaticIntTuple[rank],
    val: SIMD[type, simd_width],
):
    var flat_index = _compute_flat_index[type, rank, rank](buffer, index)

    # We have to cast bools into their runtime storage type.
    @parameter
    if type is DType.bool:
        var v = val.cast[DType.uint8]()
        SIMD[size=simd_width].store(
            buffer.data.bitcast[DType.uint8](), flat_index, v
        )
    else:
        SIMD[size=simd_width].store(buffer.data, flat_index, val)


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

    var in_host_ptr = UnsafePointer[Scalar[type]].alloc(N, alignment=align)
    var out_host_ptr = UnsafePointer[Scalar[type]].alloc(N, alignment=align)

    var in_host = NDBuffer[type, rank](in_host_ptr, dims)
    var out_host = NDBuffer[type, rank](out_host_ptr, dims)

    for i in range(N):
        in_host_ptr[i] = i

    var in_buffer = DeviceNDBuffer[type, rank](dims, ctx=ctx)
    var out_buffer = DeviceNDBuffer[type, rank](dims, ctx=ctx)

    ctx.enqueue_copy_to_device(in_buffer.buffer, in_host.data)

    var in_tensor = in_buffer.tensor
    var out_tensor = out_buffer.tensor

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
            simd_store[type, simd_width, rank](
                out_tensor,
                idx,
                kernel_fn(
                    simd_load[type, simd_width, rank, in_tensor.shape](
                        in_tensor, idx
                    )
                ),
            )
        else:
            out_tensor.store[alignment=align](
                idx,
                kernel_fn(
                    in_tensor.load[width=simd_width, alignment=align](idx)
                ),
            )

    @parameter
    @always_inline
    fn bench_func(inout b: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            _elementwise_impl_gpu[func, pack_size](
                dims,
                ctx,
            )

        b.iter_custom[kernel_launch](ctx)

    var num_bytes = N * sizeof[type]()
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
    ctx.enqueue_copy_from_device(out_host.data, out_buffer.buffer)

    _ = in_buffer
    _ = out_buffer
    in_host_ptr.free()
    out_host_ptr.free()


fn main() raises:
    var m = Bench()

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
                            rsqrt,
                            use_aligned_memory = aligned_memory_config != 0,
                            emulate_graph_compiler = emulate_graph_compiler
                            != 0,
                        ](
                            m,
                            "rsqrt",
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
