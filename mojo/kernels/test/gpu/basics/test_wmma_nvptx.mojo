# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen -disable-prebuilt-packages -emit-asm --target-triple=nvptx64-nvidia-cuda --target-cpu=sm_90 --target-features="" %s | FileCheck %s

from gpu.wmma import _SM80_16x8x8_F16F16F16F16_TN


# CHECK-LABEL: SM80_16x8x8_F16F16F16F16_TN
@export
fn SM80_16x8x8_F16F16F16F16_TN(
    a00: Scalar[DType.float16],
    a01: Scalar[DType.float16],
    a10: Scalar[DType.float16],
    a11: Scalar[DType.float16],
    b00: Scalar[DType.float16],
    b01: Scalar[DType.float16],
    c00: Scalar[DType.float16],
    c01: Scalar[DType.float16],
    c10: Scalar[DType.float16],
    c11: Scalar[DType.float16],
) -> StaticTuple[4, Scalar[DType.float16]]:
    var r00 = Scalar[DType.float16](0)
    var r01 = Scalar[DType.float16](0)
    var r02 = Scalar[DType.float16](0)
    var r04 = Scalar[DType.float16](0)
    # CHECK: mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16
    _SM80_16x8x8_F16F16F16F16_TN(
        r00, r01, r02, r04, a00, a01, a10, a11, b00, b01, c00, c01, c10, c11
    )

    return StaticTuple[4, Scalar[DType.float16]](r00, r01, r02, r04)
