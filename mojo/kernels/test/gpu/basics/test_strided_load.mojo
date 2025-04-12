# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from sys.intrinsics import strided_load

from gpu import AddressSpace
from gpu.host._compile import _compile_code_asm
from memory import UnsafePointer
from testing import assert_true


fn strided_load_kernel[
    *, type: DType = DType.uint32, width: Int = 1
](
    output: UnsafePointer[SIMD[type, width]],
    ptr: UnsafePointer[Scalar[type], address_space = AddressSpace.GENERIC],
    stride: Int,
):
    output[] = strided_load[width](ptr, stride)


def test_strided_load():
    assert_true(
        "@llvm.masked.gather"
        in _compile_code_asm[
            strided_load_kernel[width=4], emission_kind="llvm"
        ]()
    )


def main():
    test_strided_load()
