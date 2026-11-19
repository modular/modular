// RUN: kgen-opt -split-input-file -lower-global-pop-to-llvm %s | FileCheck %s

// Tests that air.simdgroup_matrix_* intrinsics (both multiply_accumulate
// and widening_multiply_accumulate) are correctly mangled during
// ConvertPOPCallToAIRIntrinsic lowering.

// Case 1: F16 x F16 -> F32
module attributes {M.target_info = #M.target<
    triple = "air64-apple-macosx",
    arch = "apple-m5",
    features = "+metal4_0,+air2_8_0",
    data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32",
    simd_bit_width = 128>} {

// CHECK: llvm.func @air.simdgroup_matrix_16x16x16_multiply_accumulate.f.f.v8f32.v8f16.v8f16.v8f32(
// CHECK-LABEL: @test_mma_f16_f16
// CHECK: llvm.call @air.simdgroup_matrix_16x16x16_multiply_accumulate.f.f.v8f32.v8f16.v8f16.v8f32(
  kgen.func @test_mma_f16_f16(
      %a: !kgen.simd<8, f16>, %ta: !kgen.scalar<bool>, %b: !kgen.simd<8, f16>,
      %tb: !kgen.scalar<bool>, %c: !kgen.simd<8, f32>) {
    // The "llvm." prefix is stripped during lowering; the compiler mangles the
    // base name with scalar-type chars and vector type suffixes.
    %r = pop.call_llvm_intrinsic
      "llvm.air.simdgroup_matrix_16x16x16_multiply_accumulate",
      (%a, %ta, %b, %tb, %c) :
      (!kgen.simd<8, f16>, !kgen.scalar<bool>, !kgen.simd<8, f16>, !kgen.scalar<bool>, !kgen.simd<8, f32>)
      -> !kgen.simd<8, f32>
    llvm.return
  }
}

// -----

// Case 2: BF16 x F16 -> F32
module attributes {M.target_info = #M.target<
    triple = "air64-apple-macosx",
    arch = "apple-m5",
    features = "+metal4_0,+air2_8_0",
    data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32",
    simd_bit_width = 128>} {

// CHECK: llvm.func @air.simdgroup_matrix_16x16x16_multiply_accumulate.f.f.v8f32.v8bf16.v8f16.v8f32(
// CHECK-LABEL: @test_mma_bf16_f16
// CHECK: llvm.call @air.simdgroup_matrix_16x16x16_multiply_accumulate.f.f.v8f32.v8bf16.v8f16.v8f32(
  kgen.func @test_mma_bf16_f16(
      %a: !kgen.simd<8, bf16>, %ta: !kgen.scalar<bool>, %b: !kgen.simd<8, f16>,
      %tb: !kgen.scalar<bool>, %c: !kgen.simd<8, f32>) {
    %r = pop.call_llvm_intrinsic
      "llvm.air.simdgroup_matrix_16x16x16_multiply_accumulate",
      (%a, %ta, %b, %tb, %c) :
      (!kgen.simd<8, bf16>, !kgen.scalar<bool>, !kgen.simd<8, f16>, !kgen.scalar<bool>, !kgen.simd<8, f32>)
      -> !kgen.simd<8, f32>
    llvm.return
  }
}

// -----

// Case 3: I8 x I8 -> I32 widening_multiply_accumulate (signed x signed).
// Verifies the compiler also handles the widening integer variant.
module attributes {M.target_info = #M.target<
    triple = "air64-apple-macosx",
    arch = "apple-m5",
    features = "+metal4_0,+air2_8_0",
    data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32",
    simd_bit_width = 128>} {

// CHECK: llvm.func @air.simdgroup_matrix_16x16x16_widening_multiply_accumulate.s.s.v8i32.v8i8.v8i8.v8i32(
// CHECK-LABEL: @test_widening_mma_i8_i8
// CHECK: llvm.call @air.simdgroup_matrix_16x16x16_widening_multiply_accumulate.s.s.v8i32.v8i8.v8i8.v8i32(
  kgen.func @test_widening_mma_i8_i8(
      %a: !kgen.simd<8, si8>, %ta: !kgen.scalar<bool>, %b: !kgen.simd<8, si8>,
      %tb: !kgen.scalar<bool>, %c: !kgen.simd<8, si32>) {
    %r = pop.call_llvm_intrinsic
      "llvm.air.simdgroup_matrix_16x16x16_widening_multiply_accumulate",
      (%a, %ta, %b, %tb, %c) :
      (!kgen.simd<8, si8>, !kgen.scalar<bool>, !kgen.simd<8, si8>, !kgen.scalar<bool>, !kgen.simd<8, si32>)
      -> !kgen.simd<8, si32>
    llvm.return
  }
}

// -----

// Case 4: UI8 x UI8 -> UI32 widening (unsigned x unsigned)
module attributes {M.target_info = #M.target<
    triple = "air64-apple-macosx",
    arch = "apple-m5",
    features = "+metal4_0,+air2_8_0",
    data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32",
    simd_bit_width = 128>} {

// CHECK: llvm.func @air.simdgroup_matrix_16x16x16_widening_multiply_accumulate.u.u.v8i32.v8i8.v8i8.v8i32(
// CHECK-LABEL: @test_widening_mma_u8_u8
// CHECK: llvm.call @air.simdgroup_matrix_16x16x16_widening_multiply_accumulate.u.u.v8i32.v8i8.v8i8.v8i32(
  kgen.func @test_widening_mma_u8_u8(
      %a: !kgen.simd<8, ui8>, %ta: !kgen.scalar<bool>, %b: !kgen.simd<8, ui8>,
      %tb: !kgen.scalar<bool>, %c: !kgen.simd<8, ui32>) {
    %r = pop.call_llvm_intrinsic
      "llvm.air.simdgroup_matrix_16x16x16_widening_multiply_accumulate",
      (%a, %ta, %b, %tb, %c) :
      (!kgen.simd<8, ui8>, !kgen.scalar<bool>, !kgen.simd<8, ui8>, !kgen.scalar<bool>, !kgen.simd<8, ui32>)
      -> !kgen.simd<8, ui32>
    llvm.return
  }
}

// -----

// Case 5: BF16 x BF16 -> F32 (same-type float inputs)
module attributes {M.target_info = #M.target<
    triple = "air64-apple-macosx",
    arch = "apple-m5",
    features = "+metal4_0,+air2_8_0",
    data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32",
    simd_bit_width = 128>} {

// CHECK: llvm.func @air.simdgroup_matrix_16x16x16_multiply_accumulate.f.f.v8f32.v8bf16.v8bf16.v8f32(
// CHECK-LABEL: @test_mma_bf16_bf16
// CHECK: llvm.call @air.simdgroup_matrix_16x16x16_multiply_accumulate.f.f.v8f32.v8bf16.v8bf16.v8f32(
  kgen.func @test_mma_bf16_bf16(
      %a: !kgen.simd<8, bf16>, %ta: !kgen.scalar<bool>, %b: !kgen.simd<8, bf16>,
      %tb: !kgen.scalar<bool>, %c: !kgen.simd<8, f32>) {
    %r = pop.call_llvm_intrinsic
      "llvm.air.simdgroup_matrix_16x16x16_multiply_accumulate",
      (%a, %ta, %b, %tb, %c) :
      (!kgen.simd<8, bf16>, !kgen.scalar<bool>, !kgen.simd<8, bf16>, !kgen.scalar<bool>, !kgen.simd<8, f32>)
      -> !kgen.simd<8, f32>
    llvm.return
  }
}

// -----

// Case 6: F32 x F32 -> F32 (all same type)
module attributes {M.target_info = #M.target<
    triple = "air64-apple-macosx",
    arch = "apple-m5",
    features = "+metal4_0,+air2_8_0",
    data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32",
    simd_bit_width = 128>} {

// CHECK: llvm.func @air.simdgroup_matrix_16x16x16_multiply_accumulate.f.f.v8f32.v8f32.v8f32.v8f32(
// CHECK-LABEL: @test_mma_f32_f32
// CHECK: llvm.call @air.simdgroup_matrix_16x16x16_multiply_accumulate.f.f.v8f32.v8f32.v8f32.v8f32(
  kgen.func @test_mma_f32_f32(
      %a: !kgen.simd<8, f32>, %ta: !kgen.scalar<bool>, %b: !kgen.simd<8, f32>,
      %tb: !kgen.scalar<bool>, %c: !kgen.simd<8, f32>) {
    %r = pop.call_llvm_intrinsic
      "llvm.air.simdgroup_matrix_16x16x16_multiply_accumulate",
      (%a, %ta, %b, %tb, %c) :
      (!kgen.simd<8, f32>, !kgen.scalar<bool>, !kgen.simd<8, f32>, !kgen.scalar<bool>, !kgen.simd<8, f32>)
      -> !kgen.simd<8, f32>
    llvm.return
  }
}

// -----

// Case 7: I16 x I16 -> I32 widening (signed x signed, 16-bit inputs)
module attributes {M.target_info = #M.target<
    triple = "air64-apple-macosx",
    arch = "apple-m5",
    features = "+metal4_0,+air2_8_0",
    data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32",
    simd_bit_width = 128>} {

// CHECK: llvm.func @air.simdgroup_matrix_16x16x16_widening_multiply_accumulate.s.s.v8i32.v8i16.v8i16.v8i32(
// CHECK-LABEL: @test_widening_mma_i16_i16
// CHECK: llvm.call @air.simdgroup_matrix_16x16x16_widening_multiply_accumulate.s.s.v8i32.v8i16.v8i16.v8i32(
  kgen.func @test_widening_mma_i16_i16(
      %a: !kgen.simd<8, si16>, %ta: !kgen.scalar<bool>, %b: !kgen.simd<8, si16>,
      %tb: !kgen.scalar<bool>, %c: !kgen.simd<8, si32>) {
    %r = pop.call_llvm_intrinsic
      "llvm.air.simdgroup_matrix_16x16x16_widening_multiply_accumulate",
      (%a, %ta, %b, %tb, %c) :
      (!kgen.simd<8, si16>, !kgen.scalar<bool>, !kgen.simd<8, si16>, !kgen.scalar<bool>, !kgen.simd<8, si32>)
      -> !kgen.simd<8, si32>
    llvm.return
  }
}

// -----

// Case 8: UI8 x SI8 -> SI32 widening (mixed signedness)
module attributes {M.target_info = #M.target<
    triple = "air64-apple-macosx",
    arch = "apple-m5",
    features = "+metal4_0,+air2_8_0",
    data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32",
    simd_bit_width = 128>} {

// CHECK: llvm.func @air.simdgroup_matrix_16x16x16_widening_multiply_accumulate.u.s.v8i32.v8i8.v8i8.v8i32(
// CHECK-LABEL: @test_widening_mma_u8_s8
// CHECK: llvm.call @air.simdgroup_matrix_16x16x16_widening_multiply_accumulate.u.s.v8i32.v8i8.v8i8.v8i32(
  kgen.func @test_widening_mma_u8_s8(
      %a: !kgen.simd<8, ui8>, %ta: !kgen.scalar<bool>, %b: !kgen.simd<8, si8>,
      %tb: !kgen.scalar<bool>, %c: !kgen.simd<8, si32>) {
    %r = pop.call_llvm_intrinsic
      "llvm.air.simdgroup_matrix_16x16x16_widening_multiply_accumulate",
      (%a, %ta, %b, %tb, %c) :
      (!kgen.simd<8, ui8>, !kgen.scalar<bool>, !kgen.simd<8, si8>, !kgen.scalar<bool>, !kgen.simd<8, si32>)
      -> !kgen.simd<8, si32>
    llvm.return
  }
}

// -----

// Case 9: E4M3 x E4M3 -> F32. FP8 operands lower to <8 x i8>, but the name
// token must carry the logical fp8 type (v8f8e4m3fn), recovered from the
// KGEN dtype.
module attributes {M.target_info = #M.target<
    triple = "air64-apple-macosx",
    arch = "apple-m5",
    features = "+metal4_0,+air2_8_0",
    data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32",
    simd_bit_width = 128>} {

// CHECK: llvm.func @air.simdgroup_matrix_16x16x16_multiply_accumulate.f.f.v8f32.v8f8e4m3fn.v8f8e4m3fn.v8f32(
// CHECK-LABEL: @test_mma_e4m3_e4m3
// CHECK: llvm.call @air.simdgroup_matrix_16x16x16_multiply_accumulate.f.f.v8f32.v8f8e4m3fn.v8f8e4m3fn.v8f32(
  kgen.func @test_mma_e4m3_e4m3(
      %a: !kgen.simd<8, f8e4m3fn>, %ta: !kgen.scalar<bool>, %b: !kgen.simd<8, f8e4m3fn>,
      %tb: !kgen.scalar<bool>, %c: !kgen.simd<8, f32>) {
    %r = pop.call_llvm_intrinsic
      "llvm.air.simdgroup_matrix_16x16x16_multiply_accumulate",
      (%a, %ta, %b, %tb, %c) :
      (!kgen.simd<8, f8e4m3fn>, !kgen.scalar<bool>, !kgen.simd<8, f8e4m3fn>, !kgen.scalar<bool>, !kgen.simd<8, f32>)
      -> !kgen.simd<8, f32>
    llvm.return
  }
}

// -----

// Case 10: E4M3 x E5M2 -> F32 (mixed fp8 input types).
module attributes {M.target_info = #M.target<
    triple = "air64-apple-macosx",
    arch = "apple-m5",
    features = "+metal4_0,+air2_8_0",
    data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32",
    simd_bit_width = 128>} {

// CHECK: llvm.func @air.simdgroup_matrix_16x16x16_multiply_accumulate.f.f.v8f32.v8f8e4m3fn.v8f8e5m2.v8f32(
// CHECK-LABEL: @test_mma_e4m3_e5m2
// CHECK: llvm.call @air.simdgroup_matrix_16x16x16_multiply_accumulate.f.f.v8f32.v8f8e4m3fn.v8f8e5m2.v8f32(
  kgen.func @test_mma_e4m3_e5m2(
      %a: !kgen.simd<8, f8e4m3fn>, %ta: !kgen.scalar<bool>, %b: !kgen.simd<8, f8e5m2>,
      %tb: !kgen.scalar<bool>, %c: !kgen.simd<8, f32>) {
    %r = pop.call_llvm_intrinsic
      "llvm.air.simdgroup_matrix_16x16x16_multiply_accumulate",
      (%a, %ta, %b, %tb, %c) :
      (!kgen.simd<8, f8e4m3fn>, !kgen.scalar<bool>, !kgen.simd<8, f8e5m2>, !kgen.scalar<bool>, !kgen.simd<8, f32>)
      -> !kgen.simd<8, f32>
    llvm.return
  }
}
