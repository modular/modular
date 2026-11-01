; Test that ConstantFP vector splats (from LLVM PR#74502) survive the LLVM 17
; bitcode writer on the Apple AIR/Metal path.
;
; After LLVM PR#74502, UseConstantFPForFixedLengthSplat defaults to true, so
; equal-element FP vectors are stored as ConstantFP with a vector type.  The
; LLVM 17 / Metal AIR reader expects CST_CODE_DATA for these, not CST_CODE_FLOAT.
; BitcodeWriter17 must expand the splat before writing.

; RUN: kgen-llvm-opt --disable-optimization-passes %s -o %t.air
; RUN: xcrun -sdk macosx air-objdump --disassemble %t.air | FileCheck %s

; REQUIRES: system-darwin, air-objdump

target triple = "air64-apple-macosx"

; Global vector float constants using splat values.
; These become ConstantFP with a vector type after LLVM PR#74502.
@g_float4 = global <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>
@g_float2 = global <2 x float> <float 2.000000e+00, float 2.000000e+00>
@g_half4 = global <4 x half> <half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00>
@g_double2 = global <2 x double> <double 1.000000e+00, double 1.000000e+00>

; Function that returns a <4 x float> splat — the splat is used inline.
; CHECK-LABEL: define {{.*}} @ret_float4_splat
define <4 x float> @ret_float4_splat() {
  ret <4 x float> <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
}

; Function that adds a <4 x float> splat constant.
; CHECK-LABEL: define {{.*}} @add_float4_splat
define <4 x float> @add_float4_splat(<4 x float> %x) {
  %s = fadd <4 x float> %x, <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>
  ret <4 x float> %s
}

; Function using a half-precision splat.
; CHECK-LABEL: define {{.*}} @ret_half4_splat
define <4 x half> @ret_half4_splat() {
  ret <4 x half> <half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00>
}

; Function using a double-precision splat.
; CHECK-LABEL: define {{.*}} @ret_double2_splat
define <2 x double> @ret_double2_splat() {
  ret <2 x double> <double 2.000000e+00, double 2.000000e+00>
}

; Scalar float alongside a vector splat — ensures the scalar path is unaffected.
; CHECK-LABEL: define {{.*}} @mixed_splat_and_scalar
define float @mixed_splat_and_scalar(<4 x float> %v) {
  %splat_sum = fadd <4 x float> %v, <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>
  %e = extractelement <4 x float> %splat_sum, i32 0
  %r = fadd float %e, 4.200000e+01
  ret float %r
}
