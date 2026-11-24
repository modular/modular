// RUN: kgen-opt -split-input-file -lower-global-pop-to-llvm %s | FileCheck %s

// Tests pop.cast lowering for FP8 scalar types on Apple Metal GPU.
//
// FP8 scalars are stored as i8 in LLVM IR, so the generated air.convert.*
// names use "i8" rather than the FP8 encoding name.  These tests verify
// end-to-end correctness of the cast legalization path on Metal.

// f8e5m2 <-> f32
module attributes {M.target_info = #M.target<
    triple = "air64-apple-macosx",
    arch = "apple-m5",
    features = "+metal4_0,+air2_8_0",
    data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32",
    simd_bit_width = 128>} {

// CHECK-LABEL: @test_f8e5m2_to_f32
// CHECK: llvm.call @air.convert.f.f32.f.i8(
kgen.func @test_f8e5m2_to_f32(%x: !kgen.scalar<f8e5m2>) {
  %r = pop.cast %x : !kgen.scalar<f8e5m2> to !kgen.scalar<f32>
  llvm.return
}

// CHECK-LABEL: @test_f32_to_f8e5m2
// CHECK: llvm.call @air.convert.f.i8.f.f32(
kgen.func @test_f32_to_f8e5m2(%x: !kgen.scalar<f32>) {
  %r = pop.cast %x : !kgen.scalar<f32> to !kgen.scalar<f8e5m2>
  llvm.return
}

}

// -----

// f8e4m3fn <-> f16
module attributes {M.target_info = #M.target<
    triple = "air64-apple-macosx",
    arch = "apple-m5",
    features = "+metal4_0,+air2_8_0",
    data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32",
    simd_bit_width = 128>} {

// CHECK-LABEL: @test_f8e4m3fn_to_f16
// CHECK: llvm.call @air.convert.f.f16.f.i8(
kgen.func @test_f8e4m3fn_to_f16(%x: !kgen.scalar<f8e4m3fn>) {
  %r = pop.cast %x : !kgen.scalar<f8e4m3fn> to !kgen.scalar<f16>
  llvm.return
}

// CHECK-LABEL: @test_f16_to_f8e4m3fn
// CHECK: llvm.call @air.convert.f.i8.f.f16(
kgen.func @test_f16_to_f8e4m3fn(%x: !kgen.scalar<f16>) {
  %r = pop.cast %x : !kgen.scalar<f16> to !kgen.scalar<f8e4m3fn>
  llvm.return
}

}
