# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from gpu.host._compile import _compile_code, _get_nvptx_target
from gpu.time import *
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


fn clock_functions():
    _ = clock()
    _ = clock64()
    _ = now()


@always_inline
fn _verify_clock_functions(asm: String) raises -> None:
    assert_true("mov.u32" in asm)
    assert_true("mov.u64" in asm)


def test_clock_functions_sm80():
    alias asm = str(
        _compile_code[
            clock_functions,
            target = _get_nvptx_target(),
        ]().asm
    )
    _verify_clock_functions(asm)


def test_clock_functions_sm90():
    alias asm = str(
        _compile_code[clock_functions, target = _get_nvptx_target_sm90()]().asm
    )
    _verify_clock_functions(asm)


fn time_functions(some_value: Int) -> Int:
    var tmp = some_value

    @always_inline
    @parameter
    fn something():
        tmp += 1

    _ = time_function[something]()

    return tmp


@always_inline
fn _verify_time_functions(asm: String) raises -> None:
    assert_true("mov.u64" in asm)
    assert_true("add.s64" in asm)


def test_time_functions_sm80():
    alias asm = str(
        _compile_code[time_functions, target = _get_nvptx_target()]().asm
    )
    _verify_time_functions(asm)


def test_time_functions_sm90():
    alias asm = str(
        _compile_code[time_functions, target = _get_nvptx_target_sm90()]().asm
    )
    _verify_time_functions(asm)


def main():
    @parameter
    if not is_defined["MODULAR_PRODUCTION"]():
        test_clock_functions_sm80()
        test_clock_functions_sm90()
        test_time_functions_sm80()
        test_time_functions_sm90()
