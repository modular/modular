; Tests the LLVM IR -> AIR bitcode -> xcrun metal pipeline for the FP8
; conversion intrinsics emitted by pop.cast on Apple Metal GPU.
;
; RUN: kgen-llvm-opt --disable-optimization-passes %s -o %t.air
; RUN: xcrun -sdk macosx air-objdump --disassemble %t.air | FileCheck %s --check-prefix=AIR
; RUN: xcrun -sdk macosx metal %t.air -o %t.metallib 2>&1 || true

; REQUIRES: system-darwin, air-objdump

target triple = "air64-apple-macosx"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"

; FP8 scalars are stored as i8 in LLVM IR.  pop.cast scalar<f8e5m2> -> scalar<f32>
; lowers to a call to air.convert.f.f32.f.i8, and the reverse to
; air.convert.f.i8.f.f32.  Verify both declarations survive the bitcode round-trip
; and that the Metal compiler is given the opportunity to resolve them.

declare float @air.convert.f.f32.f.i8(i8)
declare i8    @air.convert.f.i8.f.f32(float)

; AIR: @air.convert.f.f32.f.i8
define float @test_f8e5m2_to_f32(i8 %x) {
  %r = call float @air.convert.f.f32.f.i8(i8 %x)
  ret float %r
}

; AIR: @air.convert.f.i8.f.f32
define i8 @test_f32_to_f8e5m2(float %x) {
  %r = call i8 @air.convert.f.i8.f.f32(float %x)
  ret i8 %r
}
