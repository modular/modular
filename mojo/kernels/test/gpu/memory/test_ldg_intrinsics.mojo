# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from gpu.host._compile import _compile_code, _get_nvptx_target
from gpu.intrinsics import ldg
from memory.unsafe import DTypePointer
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
    i8: DTypePointer[DType.int8],
    ui8: DTypePointer[DType.uint8],
    i16: DTypePointer[DType.int16],
    ui16: DTypePointer[DType.uint16],
    i32: DTypePointer[DType.int32],
    ui32: DTypePointer[DType.uint32],
    i64: DTypePointer[DType.int64],
    ui64: DTypePointer[DType.uint64],
    f32: DTypePointer[DType.float32],
    f64: DTypePointer[DType.float64],
):
    # Note we perform the store purely to avoid the compiler from optimizing
    # away the statements.

    i8.store(ldg(i8))
    ui8.store(ldg(ui8))
    i16.store(ldg(i16))
    ui16.store(ldg(ui16))
    i32.store(ldg(i32))
    ui32.store(ldg(ui32))
    i64.store(ldg(i64))
    ui64.store(ldg(ui64))
    f32.store(ldg(f32))
    f64.store(ldg(f64))


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
        _compile_code[
            __type_of(register_intrinsics),
            register_intrinsics,
            target = _get_nvptx_target(),
        ]().asm
    )
    _verify_register_intrinsics(asm)


def test_register_intrinsics_sm90():
    alias asm = str(
        _compile_code[
            __type_of(register_intrinsics),
            register_intrinsics,
            target = _get_nvptx_target_sm90(),
        ]().asm
    )
    _verify_register_intrinsics(asm)


def main():
    @parameter
    if not is_defined["MODULAR_PRODUCTION"]():
        test_register_intrinsics_sm80()
        test_register_intrinsics_sm90()
