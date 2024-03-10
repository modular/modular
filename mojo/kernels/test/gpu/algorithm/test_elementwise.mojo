# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo %s | FileCheck %s

from math import exp
from sys.info import triple_is_nvidia_cuda

from algorithm.functional import _elementwise_impl
from builtin.io import _printf
from gpu import *
from gpu.host import Context, Dim, Function, Stream
from gpu.host._compile import _get_nvptx_target
from benchmark.cuda import run
from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _free,
    _malloc,
    _memset,
)
from gpu.host.sync import synchronize
from memory.buffer import NDBuffer
from tensor import Tensor

from utils.index import Index

alias pack_size = simdwidthof[Float32, target = _get_nvptx_target()]()


# CHECK-LABEL: run_elementwise
fn run_elementwise() raises:
    print("== run_elementwise")

    var in_host = Tensor[DType.float32](2, 3, 2)
    var out_host = Tensor[DType.float32](2, 3, 2)

    var flattened_length = in_host.num_elements()
    for i in range(2):
        for j in range(3):
            for k in range(2):
                in_host[Index(i, j, k)] = i + j + k

    var in_device = _malloc[DType.float32](flattened_length)
    var out_device = _malloc[DType.float32](flattened_length)

    _copy_host_to_device(in_device, in_host.data(), flattened_length)

    var in_buffer = NDBuffer[DType.float32, 3](in_device, Index(2, 3, 2))
    var out_buffer = NDBuffer[DType.float32, 3](out_device, Index(2, 3, 2))

    @always_inline
    @__copy_capture(in_buffer, out_buffer)
    @parameter
    fn func[simd_width: Int, rank: Int](idx0: StaticIntTuple[rank]):
        var idx = rebind[StaticIntTuple[3]](idx0)

        alias alignment = 1 if simd_width == 1 else alignof[
            SIMD[DType.float32, pack_size]
        ]()

        out_buffer.aligned_simd_store[simd_width, alignment](
            idx, in_buffer.aligned_simd_load[simd_width, alignment](idx) + 42
        )

    _elementwise_impl[func, pack_size, 3, True, target="cuda"](
        StaticIntTuple[3](2, 3, 2),
    )
    synchronize()

    _copy_device_to_host(out_host.data(), out_device, flattened_length)

    # CHECK: 42.0
    # CHECK: 43.0
    # CHECK: 43.0
    # CHECK: 44.0
    # CHECK: 44.0
    # CHECK: 45.0
    # CHECK: 43.0
    # CHECK: 44.0
    # CHECK: 44.0
    # CHECK: 45.0
    # CHECK: 45.0
    # CHECK: 46.0
    for i in range(2):
        for j in range(3):
            for k in range(2):
                print(out_host[Index(i, j, k)])

    _ = in_host ^
    _ = out_host ^

    _free(in_device)
    _free(out_device)


# CHECK-LABEL: run_elementwise_uneven_simd
fn run_elementwise_uneven_simd() raises:
    print("== run_elementwise_uneven_simd")

    var in_host = Tensor[DType.float32](3, 3)
    var out_host = Tensor[DType.float32](3, 3)

    var flattened_length = in_host.num_elements()
    for i in range(3):
        for j in range(3):
            in_host[Index(i, j)] = i + j

    var in_device = _malloc[DType.float32](flattened_length)
    var out_device = _malloc[DType.float32](flattened_length)

    _copy_host_to_device(in_device, in_host.data(), flattened_length)

    var in_buffer = NDBuffer[DType.float32, 2](in_device, Index(3, 3))
    var out_buffer = NDBuffer[DType.float32, 2](out_device, Index(3, 3))

    @always_inline
    @__copy_capture(in_buffer, out_buffer)
    @parameter
    fn func[simd_width: Int, rank: Int](idx0: StaticIntTuple[rank]):
        var idx = rebind[StaticIntTuple[2]](idx0)

        alias alignment = 1 if simd_width == 1 else alignof[
            SIMD[DType.float32, pack_size]
        ]()

        out_buffer.aligned_simd_store[simd_width, alignment](
            idx, in_buffer.aligned_simd_load[simd_width, alignment](idx) + 42
        )

    _elementwise_impl[func, pack_size, 2, True, target="cuda"](
        StaticIntTuple[2](3, 3),
    )
    synchronize()
    _copy_device_to_host(out_host.data(), out_device, flattened_length)

    # CHECK: 42.0
    # CHECK: 43.0
    # CHECK: 44.0
    # CHECK: 43.0
    # CHECK: 44.0
    # CHECK: 45.0
    # CHECK: 44.0
    # CHECK: 45.0
    # CHECK: 46.0
    for i in range(3):
        for j in range(3):
            print(out_host[Index(i, j)])

    _ = in_host ^
    _ = out_host ^

    _free(in_device)
    _free(out_device)


# CHECK-LABEL: run_elementwise_transpose_copy
fn run_elementwise_transpose_copy() raises:
    print("== run_elementwise_transpose_copy")
    var in_host = Tensor[DType.float32](2, 4, 5)
    var out_host = Tensor[DType.float32](4, 2, 5)

    var flattened_length = in_host.num_elements()
    for i in range(2):
        for j in range(4):
            for k in range(5):
                in_host[Index(i, j, k)] = i * 4 * 5 + j * 5 + k

    var in_device = _malloc[DType.float32](flattened_length)
    var out_device = _malloc[DType.float32](flattened_length)

    _copy_host_to_device(in_device, in_host.data(), flattened_length)

    var in_buffer_transposed = NDBuffer[DType.float32, 3](
        in_device, Index(4, 2, 5), Index(5, 20, 1)
    )
    var out_buffer = NDBuffer[DType.float32, 3](out_device, Index(4, 2, 5))

    @always_inline
    @__copy_capture(in_buffer_transposed, out_buffer)
    @parameter
    fn func[simd_width: Int, rank: Int](idx0: StaticIntTuple[rank]):
        var idx = rebind[StaticIntTuple[3]](idx0)

        out_buffer.simd_store[simd_width](
            idx, in_buffer_transposed.simd_load[simd_width](idx)
        )

    _elementwise_impl[func, 4, 3, True, target="cuda"](
        StaticIntTuple[3](4, 2, 5),
    )
    synchronize()
    _copy_device_to_host(out_host.data(), out_device, flattened_length)

    # CHECK:0.0
    # CHECK:1.0
    # CHECK:2.0
    # CHECK:3.0
    # CHECK:4.0
    # CHECK:20.0
    # CHECK:21.0
    # CHECK:22.0
    # CHECK:23.0
    # CHECK:24.0
    # CHECK:5.0
    # CHECK:6.0
    # CHECK:7.0
    # CHECK:8.0
    # CHECK:9.0
    # CHECK:25.0
    # CHECK:26.0
    # CHECK:27.0
    # CHECK:28.0
    # CHECK:29.0
    # CHECK:10.0
    # CHECK:11.0
    # CHECK:12.0
    # CHECK:13.0
    # CHECK:14.0
    # CHECK:30.0
    # CHECK:31.0
    # CHECK:32.0
    # CHECK:33.0
    # CHECK:34.0
    # CHECK:15.0
    # CHECK:16.0
    # CHECK:17.0
    # CHECK:18.0
    # CHECK:19.0
    # CHECK:35.0
    # CHECK:36.0
    # CHECK:37.0
    # CHECK:38.0
    # CHECK:39.0
    for i in range(4):
        for j in range(2):
            for k in range(5):
                print(out_host[Index(i, j, k)])

    _ = in_host ^
    _ = out_host ^

    _free(in_device)
    _free(out_device)


# CHECK-NOT: CUDA_ERROR
fn main():
    try:
        with Context() as ctx:
            run_elementwise()
            run_elementwise_uneven_simd()
            run_elementwise_transpose_copy()
    except e:
        print("CUDA_ERROR:", e)
