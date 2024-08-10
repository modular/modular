# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s

from gpu.host._compile import _compile_code
from testing import assert_true
from sys.intrinsics import prefetch


def test_prefetch():
    fn do_prefetch[
        type: DType, *, offset: Int = 0
    ](addr: UnsafePointer[Scalar[type]]):
        prefetch(addr + offset)

    assert_true(
        "prefetch.global.L2 " in _compile_code[do_prefetch[DType.float16]]().asm
    )
    assert_true(
        "prefetch.global.L2 " in _compile_code[do_prefetch[DType.float32]]().asm
    )
    assert_true(
        "prefetch.global.L2 " in _compile_code[do_prefetch[DType.int32]]().asm
    )

    assert_true(
        "prefetch.global.L2 "
        in _compile_code[do_prefetch[DType.int64, offset=42]]().asm
    )


def main():
    test_prefetch()
