# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo %s | FileCheck %s

from math import exp, isclose
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
from closed_source_memory.buffer import NDBuffer
from tensor import Tensor

from utils.index import Index

alias pack_size = simdwidthof[Float32, target = _get_nvptx_target()]()


# CHECK-LABEL: run_elementwise
fn run_elementwise() raises:
    print("== run_elementwise")

    alias length = 16

    var in_host = Tensor[DType.float32](length)
    var out_host = Tensor[DType.float32](length)

    var flattened_length = in_host.num_elements()
    for i in range(length):
        in_host[i] = Float32(i) - length // 2

    var in_device = _malloc[DType.float32](flattened_length)
    var out_device = _malloc[DType.float32](flattened_length)

    _copy_host_to_device(in_device, in_host.data(), flattened_length)

    var in_buffer = NDBuffer[DType.float32, 1](in_device, Index(length))
    var out_buffer = NDBuffer[DType.float32, 1](out_device, Index(length))

    @always_inline
    @__copy_capture(out_buffer, in_buffer)
    @parameter
    fn func[simd_width: Int, rank: Int](idx0: StaticIntTuple[rank]):
        var idx = rebind[StaticIntTuple[1]](idx0)

        out_buffer.simd_store[simd_width](
            idx, exp(in_buffer.load[width=simd_width](idx))
        )

    _elementwise_impl[func, pack_size, 1, True, target="cuda"](
        StaticIntTuple[1](length),
    )
    synchronize()

    _copy_device_to_host(out_host.data(), out_device, flattened_length)

    for i in range(length):
        # CHECK-NOT: Values did not match
        if not isclose(out_host[i], exp(in_host[i])):
            print("Values did not match")

    _ = in_host ^
    _ = out_host ^

    _free(in_device)
    _free(out_device)


# CHECK-NOT: CUDA_ERROR
fn main():
    try:
        with Context() as ctx:
            run_elementwise()
    except e:
        print("CUDA_ERROR:", e)
