# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from gpu.host._compile import _compile_code, _get_nvptx_target
from gpu.intrinsics import ldg
from memory import UnsafePointer
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

    Scalar.store(i8, ldg(i8))
    Scalar.store(ui8, ldg(ui8))
    Scalar.store(i16, ldg(i16))
    Scalar.store(ui16, ldg(ui16))
    Scalar.store(i32, ldg(i32))
    Scalar.store(ui32, ldg(ui32))
    Scalar.store(i64, ldg(i64))
    Scalar.store(ui64, ldg(ui64))
    Scalar.store(f32, ldg(f32))
    Scalar.store(f64, ldg(f64))


@always_inline
fn _verify_register_intrinsics(asm: String) raises -> None:
    assert_true("ld.global.nc.u8" in asm)
    assert_true("ld.global.nc.u16" in asm)
    assert_true("ld.global.nc.u32" in asm)
    assert_true("ld.global.nc.u64" in asm)
    assert_true("ld.global.nc.f32" in asm)
    assert_true("ld.global.nc.f64" in asm)


def test_register_intrinsics_sm80():
    alias asm = str(
        _compile_code[register_intrinsics, target = _get_nvptx_target()]().asm
    )
    _verify_register_intrinsics(asm)


def test_register_intrinsics_sm90():
    alias asm = _compile_code[
        register_intrinsics,
        target = _get_nvptx_target_sm90(),
    ]().asm
    _verify_register_intrinsics(asm)


def main():
    test_register_intrinsics_sm80()
    test_register_intrinsics_sm90()
