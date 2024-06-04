# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s | FileCheck %s

from math import cos, sin
from pathlib import Path

from gpu.host import Context, Function, Stream, synchronize
from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _free,
    _malloc,
)
from testing import assert_true


fn run_func[
    type: DType, kernel_fn: fn (SIMD[type, 1]) capturing -> SIMD[type, 1]
](out_prefix: String, val: SIMD[type, 1], ref_: SIMD[type, 1]) raises:
    print("test trignometric functions on gpu")

    var out = _malloc[type](1)

    @parameter
    @__copy_capture(out)
    fn kernel(lhs: SIMD[type, 1]):
        var result = kernel_fn(lhs)
        out[0] = result

    var func = Function[kernel]()

    func(val, grid_dim=1, block_dim=1)
    synchronize()
    var out_h = DTypePointer[type].alloc(1)
    _copy_device_to_host(out_h, out, 1)
    assert_true(math.isclose(out_h[0], ref_))
    _free(out)


# CHECK-NOT: CUDA_ERROR
def main():
    @parameter
    fn cos_fn(val: SIMD[DType.float16, 1]) -> SIMD[DType.float16, 1]:
        return cos(val)

    @parameter
    fn cos_fn(val: SIMD[DType.float32, 1]) -> SIMD[DType.float32, 1]:
        return cos(val)

    @parameter
    fn sin_fn(val: SIMD[DType.float16, 1]) -> SIMD[DType.float16, 1]:
        return sin(val)

    @parameter
    fn sin_fn(val: SIMD[DType.float32, 1]) -> SIMD[DType.float32, 1]:
        return sin(val)

    try:
        with Context() as ctx:
            run_func[DType.float32, cos_fn]("./cos", 10, -0.83907192945480347)
            run_func[DType.float16, cos_fn]("./cos", 10, -0.8388671875)
            run_func[DType.float32, sin_fn]("./sin", 10, -0.54402029514312744)
            run_func[DType.float16, sin_fn]("./sin", 10, -0.5439453125)
    except e:
        print("CUDA_ERROR:", e)
