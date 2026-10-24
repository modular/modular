; Test that integer vector splats and null scalar constants survive the LLVM 17
; bitcode writer on the Apple AIR/Metal path.
;
; ConstantInt/ConstantFP must be written before the isNullValue() check so null
; scalars use typed encodings (CST_CODE_INTEGER / CST_CODE_FLOAT) rather than
; CST_CODE_NULL, which older bitcode readers mishandle for scalar types.

; RUN: kgen-llvm-opt --disable-optimization-passes %s -o %t.air
; RUN: xcrun -sdk macosx air-objdump --disassemble %t.air | FileCheck %s

; REQUIRES: system-darwin, air-objdump

target triple = "air64-apple-macosx"

; CHECK: @g_i32_4 = global <4 x i32>
@g_i32_4 = global <4 x i32> <i32 5, i32 5, i32 5, i32 5>

; CHECK: @g_i16_8 = global <8 x i16>
@g_i16_8 = global <8 x i16> <i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3>

; CHECK: @g_i32_zero = global <4 x i32>
@g_i32_zero = global <4 x i32> zeroinitializer

; CHECK: @g_i32_scalar_zero = global i32 0
@g_i32_scalar_zero = global i32 0

; CHECK: @g_float_zero = global float 0.0
@g_float_zero = global float 0.000000e+00

; CHECK-LABEL: define {{.*}} @ret_i32x4_splat
define <4 x i32> @ret_i32x4_splat() {
  ret <4 x i32> <i32 7, i32 7, i32 7, i32 7>
}

; CHECK-LABEL: define {{.*}} @use_int_zero
define i32 @use_int_zero(i32 %x) {
  %r = add i32 %x, 0
  ret i32 %r
}

; CHECK-LABEL: define {{.*}} @use_float_zero
define float @use_float_zero(float %x) {
  %r = fadd float %x, 0.000000e+00
  ret float %r
}
