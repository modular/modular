# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen -disable-prebuilt-packages -O0 -emit-asm --target-triple=nvptx64-nvidia-cuda --target-cpu=sm_90 --target-features="" %s | FileCheck %s

from memory.unsafe import DTypePointer
from gpu.intrinsics import ldg


# CHECK-LABEL: register_intrinsics
@export
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

    # CHECK: ld.global.nc.u8
    i8.store(ldg(i8))
    # CHECK: ld.global.nc.u8
    ui8.store(ldg(ui8))
    # CHECK: ld.global.nc.u16
    i16.store(ldg(i16))
    # CHECK: ld.global.nc.u16
    ui16.store(ldg(ui16))
    # CHECK: ld.global.nc.u32
    i32.store(ldg(i32))
    # CHECK: ld.global.nc.u32
    ui32.store(ldg(ui32))
    # CHECK: ld.global.nc.u64
    i64.store(ldg(i64))
    # CHECK: ld.global.nc.u64
    ui64.store(ldg(ui64))
    # CHECK: ld.global.nc.f32
    f32.store(ldg(f32))
    # CHECK: ld.global.nc.f64
    f64.store(ldg(f64))
