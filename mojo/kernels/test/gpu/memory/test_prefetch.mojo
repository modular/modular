# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from sys.intrinsics import prefetch

from gpu.host._compile import _compile_code_asm, _get_gpu_target
from memory import UnsafePointer
from testing import assert_true


fn do_prefetch[
    type: DType, *, offset: Int = 0
](addr: UnsafePointer[Scalar[type]]):
    prefetch(addr + offset)


def test_prefetch_mi300x():
    assert_true(
        "llvm.prefetch "
        in _compile_code_asm[
            do_prefetch[DType.float16], target = _get_gpu_target["mi300x"]()
        ]()
    )
    assert_true(
        "llvm.prefetch "
        in _compile_code_asm[
            do_prefetch[DType.float32], target = _get_gpu_target["mi300x"]()
        ]()
    )
    assert_true(
        "llvm.prefetch "
        in _compile_code_asm[
            do_prefetch[DType.int32], target = _get_gpu_target["mi300x"]()
        ]()
    )

    assert_true(
        "llvm.prefetch "
        in _compile_code_asm[
            do_prefetch[DType.int64, offset=42],
            target = _get_gpu_target["mi300x"](),
        ]()
    )


def test_prefetch_nvidia():
    assert_true(
        "prefetch.global.L2 "
        in _compile_code_asm[
            do_prefetch[DType.float16], target = _get_gpu_target["sm_80"]()
        ]()
    )
    assert_true(
        "prefetch.global.L2 "
        in _compile_code_asm[
            do_prefetch[DType.float32], target = _get_gpu_target["sm_80"]()
        ]()
    )
    assert_true(
        "prefetch.global.L2 "
        in _compile_code_asm[
            do_prefetch[DType.int32], target = _get_gpu_target["sm_80"]()
        ]()
    )

    assert_true(
        "prefetch.global.L2 "
        in _compile_code_asm[
            do_prefetch[DType.int64, offset=42],
            target = _get_gpu_target["sm_80"](),
        ]()
    )


def main():
    test_prefetch_nvidia()
