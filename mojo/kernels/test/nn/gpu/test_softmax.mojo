# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo %s | FileCheck %s

from math import iota, isclose
from random import rand
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


def main():
    with Context() as ctx:
        test_gpu_softmax()
