# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s -t

from math import sqrt, rsqrt, log, sin, tanh, exp, erf
from sys.info import triple_is_nvidia_cuda

from algorithm.functional import _elementwise_impl_gpu
from benchmark._cuda import run
from buffer import DimList, NDBuffer
from buffer.list import _make_tuple
from gpu.host.device_context import DeviceContext
from gpu.host._compile import _get_nvptx_target
from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from internal_utils import DeviceNDBuffer
from utils.index import product


fn add_const_fn(x: SIMD) -> __type_of(x):
    return x + 42


fn copy_fn(x: SIMD) -> __type_of(x):
    return x


# CHECK-LABEL: run_elementwise
fn run_elementwise[
    type: DType,
    kernel_fn: fn[type: DType, width: Int] (SIMD[type, width]) -> SIMD[
        type, width
    ],
    rank: Int,
](
    inout m: Bench,
    fn_name: String,
    dims: StaticIntTuple[rank],
    ctx: DeviceContext,
) raises:
    alias pack_size = simdwidthof[type, target = _get_nvptx_target()]()
    alias align = alignof[SIMD[type, pack_size]]()
    var N = product(dims, rank)

    var in_host_ptr = DTypePointer[type].alloc(N, alignment=align)
    var out_host_ptr = DTypePointer[type].alloc(N, alignment=align)

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

        out_tensor.store[width=simd_width](
            idx, kernel_fn(in_tensor.load[width=simd_width](idx))
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
            "elementwise", input_id=fn_name + "/" + str(type) + "/" + str(dims)
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

    alias shape_list = List[DimList](
        DimList(1, 1024, 3072),  # baby-replit-CE-kernels
        DimList(1, 8, 3, 1025, 128),  # baby-replit-TG-kernels
    )

    with DeviceContext() as ctx:

        @parameter
        for j in range(len(shape_list)):
            alias dims = _make_tuple[len(shape_list[j])](shape_list[j])

            @parameter
            for i in range(len(types)):
                run_elementwise[types[i], sqrt](m, "sqrt", dims, ctx)
                run_elementwise[types[i], rsqrt](m, "rsqrt", dims, ctx)
                run_elementwise[types[i], log](m, "log", dims, ctx)
                run_elementwise[types[i], sin](m, "sin", dims, ctx)
                run_elementwise[types[i], tanh](m, "tanh", dims, ctx)
                run_elementwise[types[i], exp](m, "exp", dims, ctx)
                run_elementwise[types[i], erf](m, "erf", dims, ctx)
                run_elementwise[types[i], add_const_fn](
                    m, "add_const", dims, ctx
                )
                run_elementwise[types[i], copy_fn](m, "copy", dims, ctx)
    m.dump_report()
