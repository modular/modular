# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s | FileCheck %s

from math import iota, isclose, abs
from random import rand, random_float64, seed
from sys.info import simdwidthof

from buffer import Buffer, NDBuffer
from buffer.list import Dim, DimList
from gpu.host import Context, synchronize
from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _free,
    _malloc,
    _memset,
)
from nn.softmax import softmax, softmax_2_pass
from testing import assert_true


# CHECK-LABEL: test_gpu_softmax
fn test_gpu_softmax() raises:
    print("== test_gpu_softmax")

    alias type = DType.float32
    alias rank = 3
    var shape = StaticIntTuple[rank](3, 5, 515)
    var in_host_ptr = DTypePointer[type].alloc(shape.flattened_length())
    var in_device_ptr = _malloc[type](shape.flattened_length())
    var in_host = NDBuffer[type, rank](in_host_ptr, shape)
    var in_device = NDBuffer[type, rank](in_device_ptr, shape)
    var out_host_ptr = DTypePointer[type].alloc(shape.flattened_length())
    var out_ref_ptr = DTypePointer[type].alloc(shape.flattened_length())
    var out_device_ptr = _malloc[type](shape.flattened_length())
    var out_host = NDBuffer[type, rank](out_host_ptr, shape)
    var out_ref = NDBuffer[type, rank](out_ref_ptr, shape)
    var out_device = NDBuffer[type, rank](out_device_ptr, shape)

    rand[type](in_host_ptr, shape.flattened_length())
    _copy_host_to_device(in_device_ptr, in_host_ptr, shape.flattened_length())

    @__copy_capture(in_device)
    @parameter
    fn input_fn_device[
        _simd_width: Int, _rank: Int
    ](coords: StaticIntTuple[_rank]) -> SIMD[type, _simd_width]:
        return in_device[rebind[StaticIntTuple[rank]](coords)]

    @parameter
    fn input_fn_host[
        _simd_width: Int, _rank: Int
    ](coords: StaticIntTuple[_rank]) -> SIMD[type, _simd_width]:
        return in_host[rebind[StaticIntTuple[rank]](coords)]

    softmax[
        type, 1, rank, DimList.create_unknown[rank](), input_fn_device, "cuda"
    ](shape, out_device, rank - 1)

    softmax[
        type,
        1,
        rank,
        DimList.create_unknown[rank](),
        input_fn_host,
        "cpu",
    ](shape, out_ref, rank - 1)

    synchronize()
    _copy_device_to_host(out_host_ptr, out_device_ptr, shape.flattened_length())

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

    _free(in_device_ptr)
    _free(out_device_ptr)


def test_gpu_softmax_half[test_type: DType]():
    alias seed_val = 42
    seed(seed_val)

    alias ref_type = DType.float32
    alias rank = 3

    var shape = StaticIntTuple[rank](3, 5, 515)
    var length = shape.flattened_length()

    var in_host_ref_ptr = DTypePointer[ref_type].alloc(length)
    var in_device_ref_ptr = _malloc[ref_type](length)
    var in_host_test_ptr = DTypePointer[test_type].alloc(length)
    var in_device_test_ptr = _malloc[test_type](length)
    var in_device_ref = NDBuffer[ref_type, rank](in_device_ref_ptr, shape)
    var in_device_test = NDBuffer[test_type, rank](in_device_test_ptr, shape)

    var out_host_ref_ptr = DTypePointer[ref_type].alloc(length)
    var out_device_ref_ptr = _malloc[ref_type](length)
    var out_host_test_ptr = DTypePointer[test_type].alloc(length)
    var out_device_test_ptr = _malloc[test_type](length)

    var out_device_ref = NDBuffer[ref_type, rank](out_device_ref_ptr, shape)
    var out_device_test = NDBuffer[test_type, rank](out_device_test_ptr, shape)

    # first fill BF16 pointer with random values, then cast to FP32 to
    # circumvent precision loss on casting of input. Skew the values to simulate
    # precision loss
    for i in range(length):
        # TODO use randn when GCC Float64 -> Float16 truncation is fixed #33932
        in_host_test_ptr[i] = (
            random_float64(1, 10).cast[DType.float32]().cast[test_type]()
        )
        in_host_ref_ptr[i] = in_host_test_ptr[i].cast[ref_type]()

    _copy_host_to_device(in_device_test_ptr, in_host_test_ptr, length)
    _copy_host_to_device(in_device_ref_ptr, in_host_ref_ptr, length)

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

    softmax[
        ref_type,
        1,
        rank,
        DimList.create_unknown[rank](),
        input_fn_ref,
        "cuda",
    ](shape, out_device_ref, rank - 1)

    softmax[
        test_type,
        1,
        rank,
        DimList.create_unknown[rank](),
        input_fn_test,
        "cuda",
    ](shape, out_device_test, rank - 1)

    synchronize()
    _copy_device_to_host(out_host_ref_ptr, out_device_ref_ptr, length)
    _copy_device_to_host(out_host_test_ptr, out_device_test_ptr, length)

    for i in range(length):
        var ref_val = out_host_ref_ptr[i]
        var test_val = out_host_test_ptr[i].cast[ref_type]()
        assert_true(isclose(ref_val, test_val, atol=1e-2))


def main():
    with Context() as ctx:
        test_gpu_softmax()
        test_gpu_softmax_half[DType.bfloat16]()
        test_gpu_softmax_half[DType.float16]()
