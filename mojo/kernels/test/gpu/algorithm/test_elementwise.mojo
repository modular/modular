# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: nvptx_backend
# REQUIRES: has_cuda_device
# RUN: %mojo %s | FileCheck %s

from math import exp
from sys.info import triple_is_nvidia_cuda

from algorithm.functional import _elementwise_impl
from builtin.io import _printf
from gpu import *
from gpu.host import Context, Dim, Function, Stream
from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _free,
    _malloc,
    _memset,
)
from memory.buffer import NDBuffer
from tensor import Tensor

from utils.index import Index

# CHECK-LABEL: run_elementwise
fn run_elementwise() raises:
    print("== run_elementwise")

    var in_host = Tensor[DType.float32](2, 3, 2)
    var out_host = Tensor[DType.float32](2, 3, 2)

    let flattened_length = in_host.num_elements()
    for i in range(2):
        for j in range(3):
            for k in range(2):
                in_host[Index(i, j, k)] = i + j + k

    let in_device = _malloc[DType.float32](flattened_length)
    let out_device = _malloc[DType.float32](flattened_length)

    _copy_host_to_device(in_device, in_host.data(), flattened_length)

    let in_buffer = NDBuffer[3, DimList.create_unknown[3](), DType.float32](
        in_device, Index(2, 3, 2)
    )
    let out_buffer = NDBuffer[3, DimList.create_unknown[3](), DType.float32](
        out_device, Index(2, 3, 2)
    )

    @always_inline
    @parameter
    fn func[simd_width: Int, rank: Int](idx0: StaticIntTuple[rank]):
        constrained[simd_width == 1, "simd width must be 1 on gpus"]()
        let idx = rebind[StaticIntTuple[3]](idx0)
        out_buffer[idx] = in_buffer[idx] + 42

    _elementwise_impl[3, 1, True, func, target="cuda"](
        StaticIntTuple[3](2, 3, 2),
        OutputChainPtr(),
    )

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


# CHECK-NOT: CUDA_ERROR
fn main():
    try:
        with Context() as ctx:
            run_elementwise()
    except e:
        print("CUDA_ERROR:", e)
