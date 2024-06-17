# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s | FileCheck %s

from math import isclose
from random import rand, random_float64, seed

from buffer import NDBuffer
from buffer.list import DimList
from gpu.host.device_context import DeviceContext
from nn.softmax import _softmax_cpu, _softmax_gpu
from testing import assert_true


# CHECK-LABEL: test_gpu_softmax
fn test_gpu_softmax(ctx: DeviceContext) raises:
    print("== test_gpu_softmax")

    alias type = DType.float32
    alias rank = 3
    var shape = StaticIntTuple[rank](3, 5, 515)
    var in_host_ptr = DTypePointer[type].alloc(shape.flattened_length())
    var in_device_ptr = ctx.create_buffer[type](shape.flattened_length())
    var in_host = NDBuffer[type, rank](in_host_ptr, shape)
    var in_device = NDBuffer[type, rank](in_device_ptr.ptr, shape)
    var out_host_ptr = DTypePointer[type].alloc(shape.flattened_length())
    var out_ref_ptr = DTypePointer[type].alloc(shape.flattened_length())
    var out_device_ptr = ctx.create_buffer[type](shape.flattened_length())
    var out_host = NDBuffer[type, rank](out_host_ptr, shape)
    var out_ref = NDBuffer[type, rank](out_ref_ptr, shape)
    var out_device = NDBuffer[type, rank](out_device_ptr.ptr, shape)

    rand[type](in_host_ptr, shape.flattened_length())
    ctx.enqueue_copy_to_device(in_device_ptr, in_host_ptr)

    @__copy_capture(in_device)
    @parameter
    fn input_fn_device[
        _simd_width: Int, _rank: Int
    ](coords: StaticIntTuple[_rank]) -> SIMD[type, _simd_width]:
        return in_device.load[width=_simd_width](
            rebind[StaticIntTuple[rank]](coords)
        )

    @parameter
    fn input_fn_host[
        _simd_width: Int, _rank: Int
    ](coords: StaticIntTuple[_rank]) -> SIMD[type, _simd_width]:
        return in_host.load[width=_simd_width](
            rebind[StaticIntTuple[rank]](coords)
        )

    _softmax_gpu[
        type, 1, rank, DimList.create_unknown[rank](), input_fn_device
    ](shape, out_device, rank - 1, ctx)

    _softmax_cpu[
        type,
        1,
        rank,
        DimList.create_unknown[rank](),
        input_fn_host,
    ](shape, out_ref, rank - 1)

    ctx.synchronize()
    ctx.enqueue_copy_from_device(out_host_ptr, out_device_ptr)

    # CHECK-NOT: ERROR
    for i in range(shape.flattened_length()):
        if not isclose(
            out_ref.flatten()[i],
            out_host.flatten()[i],
            atol=1e-4,
            rtol=1e-5,
        ):
            print("ERROR. Mismatch at flattened idx:", i)

    in_host_ptr.free()
    out_host_ptr.free()
    out_ref_ptr.free()

    _ = in_device_ptr
    _ = out_device_ptr


# CHECK-LABEL: test_gpu_softmax_half
def test_gpu_softmax_half[test_type: DType](ctx: DeviceContext):
    print("== test_gpu_softmax_half")
    alias seed_val = 42
    seed(seed_val)

    alias ref_type = DType.float32
    alias rank = 3

    var shape = StaticIntTuple[rank](3, 5, 515)
    var length = shape.flattened_length()

    var in_host_ref_ptr = DTypePointer[ref_type].alloc(length)
    var in_device_ref_ptr = ctx.create_buffer[ref_type](length)
    var in_host_test_ptr = DTypePointer[test_type].alloc(length)
    var in_device_test_ptr = ctx.create_buffer[test_type](length)
    var in_device_ref = NDBuffer[ref_type, rank](in_device_ref_ptr.ptr, shape)
    var in_device_test = NDBuffer[test_type, rank](
        in_device_test_ptr.ptr, shape
    )

    var out_host_ref_ptr = DTypePointer[ref_type].alloc(length)
    var out_device_ref_ptr = ctx.create_buffer[ref_type](length)
    var out_host_test_ptr = DTypePointer[test_type].alloc(length)
    var out_device_test_ptr = ctx.create_buffer[test_type](length)

    var out_device_ref = NDBuffer[ref_type, rank](out_device_ref_ptr.ptr, shape)
    var out_device_test = NDBuffer[test_type, rank](
        out_device_test_ptr.ptr, shape
    )

    # first fill BF16 pointer with random values, then cast to FP32 to
    # circumvent precision loss on casting of input. Skew the values to simulate
    # precision loss
    for i in range(length):
        # TODO use randn when GCC Float64 -> Float16 truncation is fixed #33932
        in_host_test_ptr[i] = (
            random_float64(1, 10).cast[DType.float32]().cast[test_type]()
        )
        in_host_ref_ptr[i] = in_host_test_ptr[i].cast[ref_type]()

    ctx.enqueue_copy_to_device(in_device_test_ptr, in_host_test_ptr)
    ctx.enqueue_copy_to_device(in_device_ref_ptr, in_host_ref_ptr)

    @__copy_capture(in_device_ref)
    @parameter
    fn input_fn_ref[
        _simd_width: Int, _rank: Int
    ](coords: StaticIntTuple[_rank]) -> SIMD[ref_type, _simd_width]:
        return in_device_ref[rebind[StaticIntTuple[rank]](coords)]

    @__copy_capture(in_device_test)
    @parameter
    fn input_fn_test[
        _simd_width: Int, _rank: Int
    ](coords: StaticIntTuple[_rank]) -> SIMD[test_type, _simd_width]:
        return in_device_test[rebind[StaticIntTuple[rank]](coords)]

    _softmax_gpu[
        ref_type,
        1,
        rank,
        DimList.create_unknown[rank](),
        input_fn_ref,
    ](shape, out_device_ref, rank - 1, ctx)

    _softmax_gpu[
        test_type,
        1,
        rank,
        DimList.create_unknown[rank](),
        input_fn_test,
    ](shape, out_device_test, rank - 1, ctx)

    ctx.synchronize()
    ctx.enqueue_copy_from_device(out_host_ref_ptr, out_device_ref_ptr)
    ctx.enqueue_copy_from_device(out_host_test_ptr, out_device_test_ptr)

    for i in range(length):
        var ref_val = out_host_ref_ptr[i]
        var test_val = out_host_test_ptr[i].cast[ref_type]()
        assert_true(isclose(ref_val, test_val, atol=1e-2))

    _ = in_device_ref_ptr
    _ = in_device_test_ptr


def main():
    try:
        var ctx = DeviceContext()
        test_gpu_softmax(ctx)
        test_gpu_softmax_half[DType.bfloat16](ctx)
        test_gpu_softmax_half[DType.float16](ctx)
        _ = ctx
    except e:
        print("CUDA_ERROR:", e)
