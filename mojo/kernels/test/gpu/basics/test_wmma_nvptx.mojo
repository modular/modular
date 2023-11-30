# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen -emit-asm --target-triple=nvptx64-nvidia-cuda --target-cpu=sm_90 --target-features="" %s | FileCheck %s

from gpu.mma import mma


# CHECK-LABEL: SM80_16x8x8_F16F16F16F16_TN
@export
fn SM80_16x8x8_F16F16F16F16_TN(
    a: SIMD[DType.float16, 4],
    b: SIMD[DType.float16, 2],
    c: SIMD[DType.float16, 4],
) -> SIMD[DType.float16, 4]:
    var d = SIMD[DType.float16, 4]()
    # CHECK: mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%f1, %f2}, {%r1, %r2},{%r3},{%f3, %f4};
    mma(d, a, b, c)

    return d


# CHECK-LABEL: SM80_m16n8k4_F32TF32TF32F32_TN
@export
fn SM80_m16n8k4_F32TF32TF32F32_TN(
    a: SIMD[DType.float32, 2],
    b: SIMD[DType.float32, 1],
    c: SIMD[DType.float32, 4],
) -> SIMD[DType.float32, 4]:
    var d = SIMD[DType.float32, 4]()

    # CHECK: mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%f1,%f2,%f3,%f4}, {%r1,%r2}, {%r3}, {%f5,%f6,%f7,%f8};
    mma(d, a, b, c)

    return d


# CHECK-LABEL: SM80_m16n8k8_F32TF32TF32F32_TN
@export
fn SM80_m16n8k8_F32TF32TF32F32_TN(
    a: SIMD[DType.float32, 4],
    b: SIMD[DType.float32, 1],
    c: SIMD[DType.float32, 4],
) -> SIMD[DType.float32, 4]:
    var d = SIMD[DType.float32, 4]()

    # CHECK: mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%f1,%f2,%f3,%f4}, {%r1,%r2,%r3,%r4}, {%r5,%r5}, {%f5,%f6,%f7,%f8};
    mma(d, a, b.join(b), c)

    return d
