# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from gpu.host._compile import _compile_code_asm, _get_nvptx_target
from gpu.intrinsics import ldg
from memory import UnsafePointer
from testing import *


fn register_intrinsics(
    i8: UnsafePointer[Int8],
    ui8: UnsafePointer[UInt8],
    i16: UnsafePointer[Int16],
    ui16: UnsafePointer[UInt16],
    i32: UnsafePointer[Int32],
    ui32: UnsafePointer[UInt32],
    i64: UnsafePointer[Int64],
    ui64: UnsafePointer[UInt64],
    f32: UnsafePointer[Float32],
    f64: UnsafePointer[Float64],
):
    # Note we perform the store purely to avoid the compiler from optimizing
    # away the statements.

    i8.store[width=1](ldg(i8))
    ui8.store[width=1](ldg(ui8))
    i16.store[width=1](ldg(i16))
    ui16.store[width=1](ldg(ui16))
    i32.store[width=1](ldg(i32))
    ui32.store[width=1](ldg(ui32))
    i64.store[width=1](ldg(i64))
    ui64.store[width=1](ldg(ui64))
    f32.store[width=1](ldg(f32))
    f64.store[width=1](ldg(f64))


@always_inline
fn _verify_register_intrinsics(asm: String) raises -> None:
    assert_true("ld.global.nc.u8" in asm)
    assert_true("ld.global.nc.u16" in asm)
    assert_true("ld.global.nc.u32" in asm)
    assert_true("ld.global.nc.u64" in asm)
    assert_true("ld.global.nc.f32" in asm)
    assert_true("ld.global.nc.f64" in asm)


def test_register_intrinsics_sm80():
    alias asm = _compile_code_asm[
        register_intrinsics, target = _get_nvptx_target()
    ]()
    _verify_register_intrinsics(asm)


def test_register_intrinsics_sm90():
    alias asm = _compile_code_asm[
        register_intrinsics,
        target = _get_nvptx_target["sm_90"](),
    ]()
    _verify_register_intrinsics(asm)


def main():
    test_register_intrinsics_sm80()
    test_register_intrinsics_sm90()
