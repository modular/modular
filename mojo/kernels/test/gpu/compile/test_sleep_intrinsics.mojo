# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from time import sleep
from collections.string import StringSlice

from gpu.host._compile import _compile_code_asm, _get_gpu_target
from testing import *


fn sleep_intrinsics():
    sleep(0.0000001)


@always_inline
fn _verify_sleep_intrinsics_nvidia(asm: StringSlice) raises -> None:
    assert_true("nanosleep.u32" in asm)


@always_inline
fn _verify_sleep_intrinsics_mi300x(asm: StringSlice) raises -> None:
    assert_true("s_sleep" in asm)


def test_sleep_intrinsics_sm80():
    var asm = _compile_code_asm[
        sleep_intrinsics, target = _get_gpu_target["sm_80"]()
    ]()
    _verify_sleep_intrinsics_nvidia(asm)


def test_sleep_intrinsics_sm90():
    var asm = _compile_code_asm[
        sleep_intrinsics, target = _get_gpu_target["sm_90"]()
    ]()
    _verify_sleep_intrinsics_nvidia(asm)


def test_sleep_intrinsics_mi300x():
    var asm = _compile_code_asm[
        sleep_intrinsics, target = _get_gpu_target["mi300x"]()
    ]()
    _verify_sleep_intrinsics_mi300x(asm)


def main():
    test_sleep_intrinsics_sm80()
    test_sleep_intrinsics_sm90()
    test_sleep_intrinsics_mi300x()
