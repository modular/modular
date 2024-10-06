# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from time import now, sleep, time_function

from gpu.host._compile import _compile_code_asm, _get_nvptx_target
from gpu.intrinsics import *
from testing import *


fn sleep_function(val: Float64):
    sleep(val)


def test_sleep_function():
    assert_true("nanosleep.u32 " in _compile_code_asm[sleep_function]())


fn clock_functions():
    _ = clock()
    _ = clock64()
    _ = now()


@always_inline
fn _verify_clock_functions(asm: String) raises -> None:
    assert_true("mov.u32" in asm)
    assert_true("mov.u64" in asm)


def test_clock_functions_sm80():
    _verify_clock_functions(_compile_code_asm[clock_functions]())


def test_clock_functions_sm90():
    alias asm = _compile_code_asm[
        clock_functions, target = _get_nvptx_target["sm_90"]()
    ]()
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
    alias asm = _compile_code_asm[
        time_functions, target = _get_nvptx_target()
    ]()
    _verify_time_functions(asm)


def test_time_functions_sm90():
    alias asm = _compile_code_asm[
        time_functions, target = _get_nvptx_target["sm_90"]()
    ]()
    _verify_time_functions(asm)


def main():
    test_sleep_function()
    test_clock_functions_sm80()
    test_clock_functions_sm90()
    test_time_functions_sm80()
    test_time_functions_sm90()
