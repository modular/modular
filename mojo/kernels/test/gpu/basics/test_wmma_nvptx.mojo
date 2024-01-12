# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# TODO(#22563): Remove the use of `-disable-prebuilt-packages`.
# RUN: kgen -emit-asm --target-triple=nvptx64-nvidia-cuda --target-cpu=sm_90 --target-features="" -disable-prebuilt-packages %s | FileCheck %s

from gpu.mma import mma


# CHECK-LABEL: SM80_16x8x8_F16F16F16F16_TN
@export
fn SM80_16x8x8_F16F16F16F16_TN(
    a: SIMD[DType.float16, 4],
    b: SIMD[DType.float16, 2],
    c: SIMD[DType.float16, 4],
) -> SIMD[DType.float16, 4]:
    var d = SIMD[DType.float16, 4]()
    # CHECK: mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16
    # CHECK-NEXT:  {%r10, %r11},
    # CHECK-NEXT:  {%r5, %r6},
    # CHECK-NEXT:  {%r9},
    # CHECK-NEXT:  {%r1, %r2};
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

    # CHECK: mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32
    # CHECK-NEXT: {%f7, %f8, %f9, %f10},
    # CHECK-NEXT: {%r2, %r1},
    # CHECK-NEXT: {%r3},
    # CHECK-NEXT: {%f3, %f4, %f5, %f6};
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

    # CHECK: mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
    # CHECK-NEXT: {%f5, %f6, %f7, %f8}
    # CHECK-NEXT: {%r1, %r2, %r3, %r4}
    # CHECK-NEXT: {%r5, %r5}
    # CHECK-NEXT: {%f1, %f2, %f3, %f4}
    mma(d, a, b.join(b), c)

    return d
