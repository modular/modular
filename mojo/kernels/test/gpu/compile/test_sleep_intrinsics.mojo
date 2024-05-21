# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from gpu.host._compile import _compile_code, _get_nvptx_target
from gpu.time import sleep
from testing import *


@always_inline
fn _get_nvptx_target_sm90() -> __mlir_type.`!kgen.target`:
    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_90", `,
        `features = "+ptx81", `,
        `data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
        `simd_bit_width = 128> : !kgen.target`,
    ]


fn sleep_intrinsics():
    sleep(0.0000001)


@always_inline
fn _verify_sleep_intrinsics(asm: String) raises -> None:
    assert_true("test_sleep_internsics_sleep" in asm)
    assert_true("mov.b32" in asm)
    assert_true("nanosleep.u32" in asm)


def test_sleep_intrinsics_sm80():
    alias asm = str(
        _compile_code[sleep_intrinsics, target = _get_nvptx_target()]().asm
    )
    _verify_sleep_intrinsics(asm)


def test_sleep_intrinsics_sm90():
    alias asm = str(
        _compile_code[sleep_intrinsics, target = _get_nvptx_target_sm90()]().asm
    )
    _verify_sleep_intrinsics(asm)


def main():
    @parameter
    if not is_defined["MODULAR_PRODUCTION"]():
        test_sleep_intrinsics_sm80()
        test_sleep_intrinsics_sm90()
