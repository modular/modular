# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug -I%S/../../test %s -t

from math import exp
from sys.info import triple_is_nvidia_cuda

from algorithm.functional import _elementwise_impl_gpu
from benchmark._cuda import run
from buffer import DimList, NDBuffer
from gpu.host.device_context import DeviceContext
from gpu.host._compile import _get_nvptx_target
from testing import assert_equal
from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from internal_utils import DeviceNDBuffer
from utils.index import Index, product


# CHECK-LABEL: run_elementwise
fn run_elementwise[
    type: DType, dims: StaticIntTuple[2]
](inout m: Bench, ctx: DeviceContext) raises:
    alias pack_size = simdwidthof[type, target = _get_nvptx_target()]()
    alias align = alignof[SIMD[type, pack_size]]()
    alias N = product(dims, 2)

    var in_host_ptr = DTypePointer[type].alloc(N, alignment=align)
    var out_host_ptr = DTypePointer[type].alloc(N, alignment=align)

    var in_host = NDBuffer[type, 2, DimList(dims[0], dims[1])](in_host_ptr)
    var out_host = NDBuffer[type, 2, DimList(dims[0], dims[1])](out_host_ptr)

    for i in range(dims[0]):
        for j in range(dims[1]):
            in_host[Index(i, j)] = i + j

    var in_buffer = DeviceNDBuffer[type, 2, DimList(dims[0], dims[1])](ctx)
    var out_buffer = DeviceNDBuffer[type, 2, DimList(dims[0], dims[1])](ctx)

    ctx.enqueue_copy_to_device(in_buffer.buffer, in_host.data)

    alias constant: Scalar[type] = 42
    var in_tensor = in_buffer.tensor
    var out_tensor = out_buffer.tensor

    @always_inline
    @__copy_capture(in_tensor, out_tensor)
    @parameter
    fn func[simd_width: Int, rank: Int](idx0: StaticIntTuple[rank]):
        var idx = rebind[StaticIntTuple[2]](idx0)

        @parameter
        if simd_width == 1:
            alias alignment = alignof[SIMD[type, pack_size]]()
            out_tensor.store[width=simd_width, alignment=alignment](
                idx,
                in_tensor.load[width=simd_width, alignment=alignment](idx)
                + constant,
            )
        else:
            out_tensor.store[width=simd_width](
                idx,
                in_tensor.load[width=simd_width](idx) + constant,
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

    var num_bytes = N * type.sizeof()
    m.bench_function[bench_func](
        BenchId("elementwise", input_id=str(type) + "/" + str(dims)),
        ThroughputMeasure(BenchMetric.bytes, num_bytes),
    )

    ctx.synchronize()
    ctx.enqueue_copy_from_device(out_host.data, out_buffer.buffer)

    for i in range(dims[0]):
        for j in range(dims[1]):
            assert_equal(
                out_host[Index(i, j)],
                i + j + constant,
            )

    _ = in_buffer
    _ = out_buffer
    in_host_ptr.free()
    out_host_ptr.free()


# CHECK-LABEL: run_elementwise_uneven_simd
fn run_elementwise_uneven_simd[
    type: DType, dims: StaticIntTuple[2]
](inout m: Bench, ctx: DeviceContext) raises:
    alias pack_size = simdwidthof[type, target = _get_nvptx_target()]()
    alias align = alignof[SIMD[type, pack_size]]()
    alias N = product(dims, 2)

    var in_host_ptr = DTypePointer[type].alloc(N, alignment=align)
    var out_host_ptr = DTypePointer[type].alloc(N, alignment=align)

    var in_host = NDBuffer[type, 2, DimList(dims[0], dims[1])](in_host_ptr)
    var out_host = NDBuffer[type, 2, DimList(dims[0], dims[1])](out_host_ptr)

    for i in range(dims[0]):
        for j in range(dims[1]):
            in_host[Index(i, j)] = i + j

    var in_device = DeviceNDBuffer[type, 2, DimList(dims[0], dims[1])](ctx)
    var out_device = DeviceNDBuffer[type, 2, DimList(dims[0], dims[1])](ctx)

    ctx.enqueue_copy_to_device(in_device.buffer, in_host.data)

    var in_tensor = in_device.tensor
    var out_tensor = out_device.tensor

    alias constant: Scalar[type] = 42

    @always_inline
    @__copy_capture(in_tensor, out_tensor)
    @parameter
    fn func[simd_width: Int, rank: Int](idx0: StaticIntTuple[rank]):
        var idx = rebind[StaticIntTuple[2]](idx0)

        @parameter
        if simd_width == 1:
            alias alignment = alignof[SIMD[type, pack_size]]()
            out_tensor.store[width=simd_width, alignment=alignment](
                idx,
                in_tensor.load[width=simd_width, alignment=alignment](idx)
                + constant,
            )
        else:
            out_tensor.store[width=simd_width](
                idx,
                in_tensor.load[width=simd_width](idx) + constant,
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

    var num_bytes = N * type.sizeof()
    m.bench_function[bench_func](
        BenchId(
            "elementwise_uneven_simd", input_id=str(type) + "/" + str(dims)
        ),
        ThroughputMeasure(BenchMetric.bytes, num_bytes),
    )

    ctx.synchronize()
    ctx.enqueue_copy_from_device(out_host.data, out_device.buffer)

    for i in range(dims[0]):
        for j in range(dims[1]):
            assert_equal(
                out_host[Index(i, j)],
                (i + j) + constant,
            )

    _ = in_device
    _ = out_device
    in_host_ptr.free()
    out_host_ptr.free()


fn run_elementwise_transpose_copy[
    type: DType, dims_in: StaticIntTuple[3]
](inout m: Bench, ctx: DeviceContext) raises:
    alias pack_size = simdwidthof[type, target = _get_nvptx_target()]()
    alias dims_out = StaticIntTuple[3](dims_in[1], dims_in[0], dims_in[2])
    alias dimlist_out = DimList(dims_out[0], dims_out[1], dims_out[2])

    alias align = alignof[SIMD[type, pack_size]]()
    alias N = product(dims_in, 3)

    var in_host_ptr = DTypePointer[type].alloc(N, alignment=align)
    var out_host_ptr = DTypePointer[type].alloc(N, alignment=align)
    var in_host = NDBuffer[
        type, 3, DimList(dims_in[0], dims_in[1], dims_in[2])
    ](in_host_ptr)
    var out_host = NDBuffer[
        type, 3, DimList(dims_in[1], dims_in[0], dims_in[2])
    ](out_host_ptr)

    var flattened_length = in_host.num_elements()
    alias stride_in = dims_in[1] * dims_in[2]
    for i in range(dims_in[0]):
        for j in range(dims_in[1]):
            for k in range(dims_in[2]):
                in_host[Index(i, j, k)] = i * stride_in + j * dims_in[2] + k

    var in_device = DeviceNDBuffer[type, 3, dimlist_out](
        ctx, Index(dims_in[2], stride_in, 1)
    )
    var out_device = DeviceNDBuffer[type, 3, dimlist_out](ctx)

    ctx.enqueue_copy_to_device(in_device.buffer, in_host.data)

    var in_transposed_tensor = in_device.tensor
    var out_tensor = out_device.tensor

    @always_inline
    @__copy_capture(in_transposed_tensor, out_tensor)
    @parameter
    fn func[simd_width: Int, rank: Int](idx0: StaticIntTuple[rank]):
        var idx = rebind[StaticIntTuple[3]](idx0)

        out_tensor.store[width=simd_width](
            idx, in_transposed_tensor.load[width=simd_width](idx)
        )

    @parameter
    @always_inline
    fn bench_func(inout b: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            _elementwise_impl_gpu[func, pack_size](
                dims_out,
                ctx,
            )

        b.iter_custom[kernel_launch](ctx)

    var num_bytes = product(dims_in, 3) * type.sizeof()
    m.bench_function[bench_func](
        BenchId(
            "elementwise_transpose_copy",
            input_id=str(type) + "/" + str(dims_in),
        ),
        ThroughputMeasure(BenchMetric.bytes, num_bytes),
    )

    ctx.synchronize()
    ctx.enqueue_copy_from_device(out_host.data, out_device.buffer)

    alias stride_out = dims_out[1] * dims_out[2]
    for i in range(dims_out[0]):
        for j in range(dims_out[1]):
            for k in range(dims_out[2]):
                assert_equal(
                    out_host[Index(i, j, k)],
                    (i + j * dims_out[0]) * dims_out[2] + k,
                )

    _ = in_device
    _ = out_device
    in_host_ptr.free()
    out_host_ptr.free()


fn main() raises:
    var m = Bench()

    alias types = List[DType](DType.float32, DType.bfloat16, DType.float16)
    # TODO: replace the following dims with high priority models.
    with DeviceContext() as ctx:

        @parameter
        for i in range(len(types)):
            run_elementwise[types[i], StaticIntTuple[2](512, 8)](m, ctx)
            run_elementwise_uneven_simd[types[i], StaticIntTuple[2](8192, 3)](
                m, ctx
            )
            run_elementwise_transpose_copy[
                types[i], StaticIntTuple[3](1024, 4, 10)
            ](m, ctx)
    m.dump_report()
