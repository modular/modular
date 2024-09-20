# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from time import sleep

from gpu.host._compile import _compile_code_asm, _get_nvptx_target
from testing import *


fn sleep_intrinsics():
    sleep(0.0000001)


@always_inline
fn _verify_sleep_intrinsics(asm: String) raises -> None:
    assert_true("nanosleep.u32" in asm)


def test_sleep_intrinsics_sm80():
    alias asm = _compile_code_asm[
        sleep_intrinsics, target = _get_nvptx_target()
    ]()
    _verify_sleep_intrinsics(asm)


def test_sleep_intrinsics_sm90():
    alias asm = _compile_code_asm[
        sleep_intrinsics, target = _get_nvptx_target["sm_90"]()
    ]()
    _verify_sleep_intrinsics(asm)


def main():
    test_sleep_intrinsics_sm80()
    test_sleep_intrinsics_sm90()
