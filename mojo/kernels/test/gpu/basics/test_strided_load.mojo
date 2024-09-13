# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from collections import OptionalReg
from sys.intrinsics import strided_load

from gpu.host._compile import _compile_code
from memory.reference import _GPUAddressSpace
from testing import assert_equal, assert_true


fn strided_load_kernel[
    *, type: DType = DType.uint32, width: Int = 1
](ptr: UnsafePointer[Scalar[type], AddressSpace.GENERIC], stride: Int) -> SIMD[
    type, width
]:
    return strided_load[width](ptr, stride)


def test_strided_load():
    print(
        _compile_code[strided_load_kernel[width=4], emission_kind="ptx"]().asm
    )


def main():
    test_strided_load()
