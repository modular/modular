# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from sys.info import _is_sm_9x

from gpu.host._compile import _compile_code_asm, _get_gpu_target
from testing import *


fn check_sm() -> Bool:
    alias v = _is_sm_9x()
    return v


def test_is_sm_9x():
    assert_true(
        "ret i1 true"
        in _compile_code_asm[
            check_sm,
            emission_kind="llvm",
            target = _get_gpu_target["sm_90"](),
        ]()
    )
    assert_true(
        "ret i1 true"
        in _compile_code_asm[
            check_sm,
            emission_kind="llvm",
            target = _get_gpu_target["sm_90a"](),
        ]()
    )


def main():
    test_is_sm_9x()
