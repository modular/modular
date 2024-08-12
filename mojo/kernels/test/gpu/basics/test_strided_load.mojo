# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s

from gpu.host._compile import _compile_code
from testing import assert_true, assert_equal
from memory.reference import _GPUAddressSpace
from collections import Optional
from sys.intrinsics import strided_load


fn strided_load_kernel[
    *, type: DType = DType.uint32, width: Int = 1
](ptr: UnsafePointer[Scalar[type], AddressSpace.GENERIC], stride: Int) -> SIMD[
    type, width
]:
    return strided_load[type, width](ptr, stride)


def test_strided_load():
    print(
        _compile_code[strided_load_kernel[width=4], emission_kind="ptx"]().asm
    )


def main():
    test_strided_load()
