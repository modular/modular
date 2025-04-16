// RUN: kgen-opt -split-input-file -pass-pipeline='builtin.module(kgen.func(lower-pop-to-llvm),lower-kgen-to-llvm,canonicalize)' %s | FileCheck %s

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: @scalar_bitcast
// CHECK-SAME: %[[UI32:[a-z0-9]+]]:
// CHECK-SAME: %[[F32:[a-z0-9]+]]:
// CHECK-SAME: %[[F64:[a-z0-9]+]]:
kgen.func @scalar_bitcast(
    %ui32: !pop.simd<1, ui32>,
    %f32: !pop.simd<1, f32>,
    %f64: !pop.simd<1, f64>) -> (
      !pop.simd<1, f32>,
      !pop.simd<1, si32>,
      !pop.simd<1, ui64>,
      !pop.simd<32, bool>
    ) {
  // CHECK: llvm.bitcast %[[UI32]]
  %0 = pop.bitcast %ui32 : !pop.simd<1, ui32> to !pop.simd<1, f32>
  // CHECK: llvm.bitcast %[[F32]]
  %1 = pop.bitcast %f32 : !pop.simd<1, f32> to !pop.simd<1, si32>
  // CHECK: llvm.bitcast %[[F64]]
  %2 = pop.bitcast %f64 : !pop.simd<1, f64> to !pop.simd<1, ui64>
  // CHECK: llvm.bitcast %[[UI32]]
  %3 = pop.bitcast %ui32 : !pop.simd<1, ui32> to !pop.simd<32, bool>
  kgen.return %0, %1, %2, %3 :
      !pop.simd<1, f32>,
      !pop.simd<1, si32>,
      !pop.simd<1, ui64>,
      !pop.simd<32, bool>
}

// CHECK-LABEL: @simd_bitcast
// CHECK-SAME: %[[UI32:[a-z0-9]+]]:
// CHECK-SAME: %[[F32:[a-z0-9]+]]:
// CHECK-SAME: %[[F64:[a-z0-9]+]]:
kgen.func @simd_bitcast(
    %ui32:!pop.simd<4, ui32>,
    %f32:!pop.simd<4, f32>,
    %f64:!pop.simd<2, f64>) -> (
     !pop.simd<4, f32>,
     !pop.simd<4, si32>,
     !pop.simd<4, ui32>
    ) {
  // CHECK: lvm.bitcast %[[UI32]]
  %0 = pop.bitcast %ui32 :!pop.simd<4, ui32> to !pop.simd<4, f32>
  // CHECK: llvm.bitcast %[[F32]]
  %1 = pop.bitcast %f32 :!pop.simd<4, f32> to !pop.simd<4, si32>
  // CHECK: llvm.bitcast %[[F64]]
  %2 = pop.bitcast %f64 :!pop.simd<2, f64> to !pop.simd<4, ui32>
  kgen.return %0, %1, %2 :
     !pop.simd<4, f32>,
     !pop.simd<4, si32>,
     !pop.simd<4, ui32>
}

// CHECK-LABEL: @scalar_cast
// CHECK-SAME: %[[UI32:[a-z0-9]+]]:
// CHECK-SAME: %[[SI32:[a-z0-9]+]]:
// CHECK-SAME: %[[F32:[a-z0-9]+]]:
// CHECK-SAME: %[[F64:[a-z0-9]+]]:
kgen.func @scalar_cast(
    %ui32: !pop.simd<1, ui32>,
    %si32: !pop.simd<1, si32>,
    %f32: !pop.simd<1, f32>,
    %f64: !pop.simd<1, f64>) -> (
    !pop.simd<1, ui64>,
    !pop.simd<1, si64>,
    !pop.simd<1, ui16>,
    !pop.simd<1, si32>,
    !pop.simd<1, f64>,
    !pop.simd<1, f32>,
    !pop.simd<1, si64>,
    !pop.simd<1, ui32>,
    !pop.simd<1, f64>,
    !pop.simd<1, f32>,
    !pop.simd<1, f32>,
    !pop.simd<1, index>
    ) {
  // CHECK: %[[V0:.*]] = llvm.sext %[[SI32]]
  %0 = pop.cast %si32 : !pop.simd<1, si32> to !pop.simd<1, ui64>
  // CHECK: %[[V1:.*]] = llvm.zext %[[UI32]]
  %1 = pop.cast %ui32 : !pop.simd<1, ui32> to !pop.simd<1, si64>
  // CHECK: %[[V2:.*]] = llvm.trunc %[[SI32]]
  %2 = pop.cast %si32 : !pop.simd<1, si32> to !pop.simd<1, ui16>
  %3 = pop.cast %ui32 : !pop.simd<1, ui32> to !pop.simd<1, si32>
  // CHECK: %[[V4:.*]] = llvm.sitofp %[[SI32]]
  %4 = pop.cast %si32 : !pop.simd<1, si32> to !pop.simd<1, f64>
  // CHECK: %[[V5:.*]] = llvm.uitofp %[[UI32]]
  %5 = pop.cast %ui32 : !pop.simd<1, ui32> to !pop.simd<1, f32>
  // CHECK: %[[V6:.*]] = llvm.fptosi %[[F32]]
  %6 = pop.cast %f32 : !pop.simd<1, f32> to !pop.simd<1, si64>
  // CHECK: %[[V7:.*]] = llvm.fptoui %[[F64]]
  %7 = pop.cast %f64 : !pop.simd<1, f64> to !pop.simd<1, ui32>
  // CHECK: %[[V8:.*]] = llvm.fpext %[[F32]]
  %8 = pop.cast %f32 : !pop.simd<1, f32> to !pop.simd<1, f64>
  // CHECK: %[[V9:.*]] = llvm.fptrunc %[[F64]]
  %9 = pop.cast %f64 : !pop.simd<1, f64> to !pop.simd<1, f32>
  %10 = pop.cast %f32 : !pop.simd<1, f32> to !pop.simd<1, f32>
  // CHECK: %[[V11:.*]] = llvm.fptosi %[[F64]]
  %11 = pop.cast %f64 : !pop.simd<1, f64> to !pop.simd<1, index>
  // CHECK: insertvalue %[[V0]]
  // CHECK: insertvalue %[[V1]]
  // CHECK: insertvalue %[[V2]]
  // CHECK: insertvalue %[[UI32]]
  // CHECK: insertvalue %[[V4]]
  // CHECK: insertvalue %[[V5]]
  // CHECK: insertvalue %[[V6]]
  // CHECK: insertvalue %[[V7]]
  // CHECK: insertvalue %[[V8]]
  // CHECK: insertvalue %[[V9]]
  // CHECK: insertvalue %[[F32]]
  // CHECK: insertvalue %[[V11]]
  kgen.return %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11 :
    !pop.simd<1, ui64>,
    !pop.simd<1, si64>,
    !pop.simd<1, ui16>,
    !pop.simd<1, si32>,
    !pop.simd<1, f64>,
    !pop.simd<1, f32>,
    !pop.simd<1, si64>,
    !pop.simd<1, ui32>,
    !pop.simd<1, f64>,
    !pop.simd<1, f32>,
    !pop.simd<1, f32>,
    !pop.simd<1, index>
}

// CHECK-LABEL: @simd_cast
// CHECK-SAME: %[[UI32:[a-z0-9]+]]:
// CHECK-SAME: %[[SI32:[a-z0-9]+]]:
// CHECK-SAME: %[[F32:[a-z0-9]+]]:
// CHECK-SAME: %[[F64:[a-z0-9]+]]:
kgen.func @simd_cast(
    %ui32: !pop.simd<2, ui32>,
    %si32: !pop.simd<2, si32>,
    %f32: !pop.simd<2, f32>,
    %f64: !pop.simd<2, f64>) -> (
    !pop.simd<2, ui64>,
    !pop.simd<2, si64>,
    !pop.simd<2, ui16>,
    !pop.simd<2, si32>,
    !pop.simd<2, f64>,
    !pop.simd<2, f32>,
    !pop.simd<2, si64>,
    !pop.simd<2, ui32>,
    !pop.simd<2, f64>,
    !pop.simd<2, f32>,
    !pop.simd<2, f32>
    ) {
  // CHECK: %[[V0:.*]] = llvm.sext %[[SI32]]
  %0 = pop.cast %si32 : !pop.simd<2, si32> to !pop.simd<2, ui64>
  // CHECK: %[[V1:.*]] = llvm.zext %[[UI32]]
  %1 = pop.cast %ui32 : !pop.simd<2, ui32> to !pop.simd<2, si64>
  // CHECK: %[[V2:.*]] = llvm.trunc %[[SI32]]
  %2 = pop.cast %si32 : !pop.simd<2, si32> to !pop.simd<2, ui16>
  %3 = pop.cast %ui32 : !pop.simd<2, ui32> to !pop.simd<2, si32>
  // CHECK: %[[V4:.*]] = llvm.sitofp %[[SI32]]
  %4 = pop.cast %si32 : !pop.simd<2, si32> to !pop.simd<2, f64>
  // CHECK: %[[V5:.*]] = llvm.uitofp %[[UI32]]
  %5 = pop.cast %ui32 : !pop.simd<2, ui32> to !pop.simd<2, f32>
  // CHECK: %[[V6:.*]] = llvm.fptosi %[[F32]]
  %6 = pop.cast %f32 : !pop.simd<2, f32> to !pop.simd<2, si64>
  // CHECK: %[[V7:.*]] = llvm.fptoui %[[F64]]
  %7 = pop.cast %f64 : !pop.simd<2, f64> to !pop.simd<2, ui32>
  // CHECK: %[[V8:.*]] = llvm.fpext %[[F32]]
  %8 = pop.cast %f32 : !pop.simd<2, f32> to !pop.simd<2, f64>
  // CHECK: %[[V9:.*]] = llvm.fptrunc %[[F64]]
  %9 = pop.cast %f64 : !pop.simd<2, f64> to !pop.simd<2, f32>
  %10 = pop.cast %f32 : !pop.simd<2, f32> to !pop.simd<2, f32>
  // CHECK: insertvalue %[[V0]]
  // CHECK: insertvalue %[[V1]]
  // CHECK: insertvalue %[[V2]]
  // CHECK: insertvalue %[[UI32]]
  // CHECK: insertvalue %[[V4]]
  // CHECK: insertvalue %[[V5]]
  // CHECK: insertvalue %[[V6]]
  // CHECK: insertvalue %[[V7]]
  // CHECK: insertvalue %[[V8]]
  // CHECK: insertvalue %[[V9]]
  // CHECK: insertvalue %[[F32]]
  kgen.return %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10 :
    !pop.simd<2, ui64>,
    !pop.simd<2, si64>,
    !pop.simd<2, ui16>,
    !pop.simd<2, si32>,
    !pop.simd<2, f64>,
    !pop.simd<2, f32>,
    !pop.simd<2, si64>,
    !pop.simd<2, ui32>,
    !pop.simd<2, f64>,
    !pop.simd<2, f32>,
    !pop.simd<2, f32>
}
}

// -----

module attributes {M.target_info = #M.target<triple = "amdgcn-amd-amdhsa", arch="", data_layout="">} {
  // CHECK-LABEL: scalar_cast_f32_bf16
  kgen.func @scalar_cast_f32_bf16(%f32: !pop.scalar<f32>) -> (!pop.scalar<bf16>, !pop.scalar<bf16>) {
    // CHECK: %[[MANTISSA_DIFF:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK-NEXT: %[[NAN:.*]] = llvm.mlir.constant(0x7FC00000 : f32) : f32
    // CHECK-NEXT: %[[ROUNDED_BIAS:.*]] = llvm.mlir.constant(32767 : i32) : i32
    // CHECK-NEXT: %[[UNORDERED_MASK:.*]] = llvm.inline_asm asm_dialect = att "v_cmp_u_f32 $0, $1, $1", "=s,v" %arg0 : (f32) -> i64
    // CHECK-NEXT: %[[LSB:.*]] = llvm.inline_asm asm_dialect = att "v_bfe_u32 $0, $1, 16, 1", "=v,v" %arg0 : (f32) -> i32
    // CHECK-NEXT: %[[ROUNDED_VALUE:.*]] = llvm.inline_asm asm_dialect = att "v_add3_u32 $0, $1, $2, $3", "=v,v,v,v" %arg0, %[[LSB]], %[[ROUNDED_BIAS]] : (f32, i32, i32) -> i32
    // CHECK-NEXT: %[[FLOAT_BITS:.*]] = llvm.inline_asm asm_dialect = att "v_cndmask_b32 $0, $1, $2, $3", "=v,v,v,s" %[[ROUNDED_VALUE]], %[[NAN]], %[[UNORDERED_MASK]] : (i32, f32, i64) -> i32
    // CHECK-NEXT: %[[SHIFTED_I32:.*]] = llvm.lshr %[[FLOAT_BITS]], %[[MANTISSA_DIFF]] : i32
    // CHECK-NEXT: %[[SHIFTED_I16:.*]] = llvm.trunc %[[SHIFTED_I32]] : i32 to i16
    // CHECK-NEXT: llvm.bitcast %[[SHIFTED_I16]] : i16 to bf16
    %0 = pop.cast fast %f32 : !pop.scalar<f32> to !pop.scalar<bf16>
    // CHECK-NEXT: llvm.fptrunc %{{.*}} : f32 to bf16
    %1 = pop.cast %f32 : !pop.scalar<f32> to !pop.scalar<bf16>
    kgen.return %0, %1 :
      !pop.scalar<bf16>,
      !pop.scalar<bf16>
  }

  // CHECK-LABEL: simd_cast_f32_bf16
  kgen.func @simd_cast_f32_bf16(%f32: !pop.simd<2, f32>) -> (!pop.simd<2, bf16>, !pop.simd<2, bf16>) {
    // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: %[[MANTISSA_DIFF:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK-NEXT: %[[NAN:.*]] = llvm.mlir.constant(0x7FC00000 : f32) : f32
    // CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[UNDEF_BF16x2:.*]] = llvm.mlir.undef : vector<2xbf16>
    // CHECK-NEXT: %[[ROUNDED_BIAS:.*]] = llvm.mlir.constant(32767 : i32) : i32
    // CHECK-NEXT: %[[F32_0:.*]] = llvm.extractelement %arg0[%[[ZERO]] : i32] : vector<2xf32>
    // CHECK-NEXT: %[[UNORDERED_MASK_0:.*]] = llvm.inline_asm asm_dialect = att "v_cmp_u_f32 $0, $1, $1", "=s,v" %[[F32_0]] : (f32) -> i64
    // CHECK-NEXT: %[[LSB_0:.*]] = llvm.inline_asm asm_dialect = att "v_bfe_u32 $0, $1, 16, 1", "=v,v" %[[F32_0]] : (f32) -> i32
    // CHECK-NEXT: %[[ROUNDED_VALUE_0:.*]] = llvm.inline_asm asm_dialect = att "v_add3_u32 $0, $1, $2, $3", "=v,v,v,v" %[[F32_0]], %[[LSB_0]], %[[ROUNDED_BIAS]] : (f32, i32, i32) -> i32
    // CHECK-NEXT: %[[FLOAT_BITS_0:.*]] = llvm.inline_asm asm_dialect = att "v_cndmask_b32 $0, $1, $2, $3", "=v,v,v,s" %[[ROUNDED_VALUE_0]], %[[NAN]], %[[UNORDERED_MASK_0]] : (i32, f32, i64) -> i32
    // CHECK-NEXT: %[[SHIFTED_I32_0:.*]] = llvm.lshr %[[FLOAT_BITS_0]], %[[MANTISSA_DIFF]] : i32
    // CHECK-NEXT: %[[SHIFTED_I16_0:.*]] = llvm.trunc %[[SHIFTED_I32_0]] : i32 to i16
    // CHECK-NEXT: %[[BF16_0:.*]] = llvm.bitcast %[[SHIFTED_I16_0]] : i16 to bf16
    // CHECK-NEXT: %[[BF16x2_0:.*]] = llvm.insertelement %[[BF16_0]], %[[UNDEF_BF16x2]][%[[ZERO]] : i32] : vector<2xbf16>

    // CHECK-NEXT: %[[F32_1:.*]] = llvm.extractelement %arg0[%[[ONE]] : i32] : vector<2xf32>
    // CHECK-NEXT: %[[UNORDERED_MASK_1:.*]] = llvm.inline_asm asm_dialect = att "v_cmp_u_f32 $0, $1, $1", "=s,v" %[[F32_1]] : (f32) -> i64
    // CHECK-NEXT: %[[LSB_1:.*]] = llvm.inline_asm asm_dialect = att "v_bfe_u32 $0, $1, 16, 1", "=v,v" %[[F32_1]] : (f32) -> i32
    // CHECK-NEXT: %[[ROUNDED_VALUE_1:.*]] = llvm.inline_asm asm_dialect = att "v_add3_u32 $0, $1, $2, $3", "=v,v,v,v" %[[F32_1]], %[[LSB_1]], %[[ROUNDED_BIAS]] : (f32, i32, i32) -> i32
    // CHECK-NEXT: %[[FLOAT_BITS_1:.*]] = llvm.inline_asm asm_dialect = att "v_cndmask_b32 $0, $1, $2, $3", "=v,v,v,s" %[[ROUNDED_VALUE_1]], %[[NAN]], %[[UNORDERED_MASK_1]] : (i32, f32, i64) -> i32
    // CHECK-NEXT: %[[SHIFTED_I32_1:.*]] = llvm.lshr %[[FLOAT_BITS_1]], %[[MANTISSA_DIFF]] : i32
    // CHECK-NEXT: %[[SHIFTED_I16_1:.*]] = llvm.trunc %[[SHIFTED_I32_1]] : i32 to i16
    // CHECK-NEXT: %[[BF16_1:.*]] = llvm.bitcast %[[SHIFTED_I16_1]] : i16 to bf16
    // CHECK-NEXT: %[[BF16x2_1:.*]] = llvm.insertelement %[[BF16_1]], %[[BF16x2_0]][%[[ONE]] : i32] : vector<2xbf16>
    %0 = pop.cast fast %f32 : !pop.simd<2, f32> to !pop.simd<2, bf16>
    // CHECK: llvm.fptrunc %arg0 : vector<2xf32> to vector<2xbf16>
    %1 = pop.cast %f32 : !pop.simd<2, f32> to !pop.simd<2, bf16>
    kgen.return %0, %1 :
      !pop.simd<2, bf16>,
      !pop.simd<2, bf16>
  }
}

// -----

module attributes {M.target_info = #M.target<triple = "nvptx-nvidia-cuda", arch="sm_90", data_layout="">} {
  // CHECK-LABEL: simd_cast_f32_to_f8
  kgen.func @simd_cast_f32_to_f8(%f32: !pop.simd<4, f32>) -> (!pop.simd<4, f8e4m3fn>, !pop.simd<4, f8e5m2>) {
    // CHECK-DAG:    %[[TWO:.+]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-DAG:    %[[THREE:.+]] = llvm.mlir.constant(3 : i32) : i32
    // CHECK-DAG:    %[[ZERO:.+]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG:    %[[ONE:.+]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-DAG:    %[[UNDEF_F8:.+]] = llvm.mlir.undef : vector<2xi16>
    // CHECK-DAG:    %[[VAL_F8E4_1:.+]] = llvm.extractelement %arg0[%[[ONE]] : i32] : vector<4xf32>
    // CHECK-DAG:    %[[VAL_F8E4_0:.+]] = llvm.extractelement %arg0[%[[ZERO]] : i32] : vector<4xf32>
    // CHECK-DAG:    %[[F8x2_01:.+]] = llvm.inline_asm asm_dialect = att "cvt.rn.satfinite.e4m3x2.f32 $0, $1, $2;", "=h,f,f" %[[VAL_F8E4_1]], %[[VAL_F8E4_0]] : (f32, f32) -> i16
    // CHECK-DAG:    %[[VEC_F8E4_RES_01:.+]] = llvm.insertelement %[[F8x2_01]], %[[UNDEF_F8]][%[[ZERO]] : i32] : vector<2xi16>
    // CHECK-DAG:    %[[VAL_F8E4_3:.+]] = llvm.extractelement %arg0[%[[THREE]] : i32] : vector<4xf32>
    // CHECK-DAG:    %[[VAL_F8E4_2:.+]] = llvm.extractelement %arg0[%[[TWO]] : i32] : vector<4xf32>
    // CHECK-DAG:    %[[VEC_F8E4_RES_23:.+]] = llvm.inline_asm asm_dialect = att "cvt.rn.satfinite.e4m3x2.f32 $0, $1, $2;", "=h,f,f" %[[VAL_F8E4_3]], %[[VAL_F8E4_2]] : (f32, f32) -> i16
    // CHECK-DAG:    %[[VEC_F8E4_RES_0123:.+]] = llvm.insertelement %[[VEC_F8E4_RES_23]], %[[VEC_F8E4_RES_01]][%[[ONE]] : i32] : vector<2xi16>
    // CHECK-DAG:    %[[VEC_F8E4_RES:.+]] = llvm.bitcast %[[VEC_F8E4_RES_0123]] : vector<2xi16> to vector<4xi8>
    %0 = pop.cast %f32 : !pop.simd<4, f32> to !pop.simd<4, f8e4m3fn>
    // CHECK-DAG:    %[[VAL_F8E5_1:.+]] = llvm.extractelement %arg0[%[[ONE]] : i32] : vector<4xf32>
    // CHECK-DAG:    %[[VAL_F8E5_0:.+]] = llvm.extractelement %arg0[%[[ZERO]] : i32] : vector<4xf32>
    // CHECK-DAG:    %[[F8x2_01:.+]] = llvm.inline_asm asm_dialect = att "cvt.rn.satfinite.e5m2x2.f32 $0, $1, $2;", "=h,f,f" %[[VAL_F8E5_1]], %[[VAL_F8E5_0]] : (f32, f32) -> i16
    // CHECK-DAG:    %[[VEC_F8E5_RES_01:.+]] = llvm.insertelement %[[F8x2_01]], %[[UNDEF_F8]][%[[ZERO]] : i32] : vector<2xi16>
    // CHECK-DAG:    %[[VAL_F8E5_3:.+]] = llvm.extractelement %arg0[%[[THREE]] : i32] : vector<4xf32>
    // CHECK-DAG:    %[[VAL_F8E5_2:.+]] = llvm.extractelement %arg0[%[[TWO]] : i32] : vector<4xf32>
    // CHECK-DAG:    %[[VEC_F8E5_RES_23:.+]] = llvm.inline_asm asm_dialect = att "cvt.rn.satfinite.e5m2x2.f32 $0, $1, $2;", "=h,f,f" %[[VAL_F8E5_3]], %[[VAL_F8E5_2]] : (f32, f32) -> i16
    // CHECK-DAG:    %[[VEC_F8E5_RES_0123:.+]] = llvm.insertelement %[[VEC_F8E5_RES_23]], %[[VEC_F8E5_RES_01]][%[[ONE]] : i32] : vector<2xi16>
    // CHECK-DAG:    %[[VEC_F8E5_RES:.+]] = llvm.bitcast %[[VEC_F8E5_RES_0123]] : vector<2xi16> to vector<4xi8>
    %1 = pop.cast %f32 : !pop.simd<4, f32> to !pop.simd<4, f8e5m2>
    kgen.return %0, %1: !pop.simd<4, f8e4m3fn>, !pop.simd<4, f8e5m2>
  }

  // CHECK-LABEL: simd_cast_f8_to_f16
  kgen.func @simd_cast_f8_to_f16(%f8e4: !pop.simd<4, f8e4m3fn>, %f8e5: !pop.simd<4, f8e5m2>) -> (!pop.simd<4, f16>, !pop.simd<4, f16>) {
    // CHECK-DAG: %[[ZERO:.+]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG: %[[ONE:.+]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-DAG: %[[UNDEF_I32:.+]] = llvm.mlir.undef : vector<2xi32>
    // CHECK-NEXT: %[[ARG0_I16:.+]] = llvm.bitcast %arg0 : vector<4xi8> to vector<2xi16>
    // CHECK-NEXT: %[[ARG0_I16_0:.+]] = llvm.extractelement %[[ARG0_I16]][%2 : i32] : vector<2xi16>
    // CHECK-NEXT: %[[ARG0_I32_0:.+]] = llvm.inline_asm asm_dialect = att "cvt.rn.f16x2.e4m3x2 $0, $1;", "=r,h" %[[ARG0_I16_0]] : (i16) -> i32
    // CHECK-NEXT: %[[RES0_I32_0:.+]] = llvm.insertelement %[[ARG0_I32_0]], %3[%2 : i32] : vector<2xi32>
    // CHECK-NEXT: %[[ARG0_I16_1:.+]] = llvm.extractelement %[[ARG0_I16]][%1 : i32] : vector<2xi16>
    // CHECK-NEXT: %[[ARG0_I32_1:.+]] = llvm.inline_asm asm_dialect = att "cvt.rn.f16x2.e4m3x2 $0, $1;", "=r,h" %[[ARG0_I16_1]] : (i16) -> i32
    // CHECK-NEXT: %[[RES0_I32:.+]] = llvm.insertelement %[[ARG0_I32_1]], %[[RES0_I32_0]][%1 : i32] : vector<2xi32>
    // CHECK-NEXT: llvm.bitcast %[[RES0_I32]] : vector<2xi32> to vector<4xf16>
    %0 = pop.cast %f8e4 : !pop.simd<4, f8e4m3fn> to !pop.simd<4, f16>

    // CHECK-NEXT: %[[ARG1_I16:.+]] = llvm.bitcast %arg1 : vector<4xi8> to vector<2xi16>
    // CHECK-NEXT: %[[ARG1_I16_0:.+]] = llvm.extractelement %[[ARG1_I16]][%2 : i32] : vector<2xi16>
    // CHECK-NEXT: %[[ARG1_I32_0:.+]] = llvm.inline_asm asm_dialect = att "cvt.rn.f16x2.e5m2x2 $0, $1;", "=r,h" %[[ARG1_I16_0]] : (i16) -> i32
    // CHECK-NEXT: %[[RES1_I32_0:.+]] = llvm.insertelement %[[ARG1_I32_0]], %3[%2 : i32] : vector<2xi32>
    // CHECK-NEXT: %[[ARG1_I16_1:.+]] = llvm.extractelement %[[ARG1_I16]][%1 : i32] : vector<2xi16>
    // CHECK-NEXT: %[[ARG1_I32_1:.+]] = llvm.inline_asm asm_dialect = att "cvt.rn.f16x2.e5m2x2 $0, $1;", "=r,h" %[[ARG1_I16_1]] : (i16) -> i32
    // CHECK-NEXT: %[[RES1_I32:.+]] = llvm.insertelement %[[ARG1_I32_1]], %[[RES1_I32_0]][%1 : i32] : vector<2xi32>
    // CHECK-NEXT: %[[RES1_F16:.+]] = llvm.bitcast %[[RES1_I32]] : vector<2xi32> to vector<4xf16>
    %1 = pop.cast %f8e5  : !pop.simd<4, f8e5m2> to !pop.simd<4, f16>
    kgen.return %0, %1: !pop.simd<4, f16>, !pop.simd<4, f16>
  }

  // CHECK-LABEL: simd_cast_f8_to_f32
  kgen.func @simd_cast_f8_to_f32(%f8e4: !pop.simd<4, f8e4m3fn>, %f8e5: !pop.simd<4, f8e5m2>) -> (!pop.simd<4, f32>, !pop.simd<4, f32>) {
    // CHECK-DAG: %[[ZERO:.+]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG: %[[ONE:.+]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-DAG: %[[UNDEF_I32:.+]] = llvm.mlir.undef : vector<2xi32>
    // CHECK-NEXT: %[[ARG0_I16:.+]] = llvm.bitcast %arg0 : vector<4xi8> to vector<2xi16>
    // CHECK-NEXT: %[[ARG0_I16_0:.+]] = llvm.extractelement %[[ARG0_I16]][%2 : i32] : vector<2xi16>
    // CHECK-NEXT: %[[ARG0_I32_0:.+]] = llvm.inline_asm asm_dialect = att "cvt.rn.f16x2.e4m3x2 $0, $1;", "=r,h" %[[ARG0_I16_0]] : (i16) -> i32
    // CHECK-NEXT: %[[RES0_I32_0:.+]] = llvm.insertelement %[[ARG0_I32_0]], %3[%2 : i32] : vector<2xi32>
    // CHECK-NEXT: %[[ARG0_I16_1:.+]] = llvm.extractelement %[[ARG0_I16]][%1 : i32] : vector<2xi16>
    // CHECK-NEXT: %[[ARG0_I32_1:.+]] = llvm.inline_asm asm_dialect = att "cvt.rn.f16x2.e4m3x2 $0, $1;", "=r,h" %[[ARG0_I16_1]] : (i16) -> i32
    // CHECK-NEXT: %[[RES0_I32:.+]] = llvm.insertelement %[[ARG0_I32_1]], %[[RES0_I32_0]][%1 : i32] : vector<2xi32>
    // CHECK-NEXT: %[[RES0_F16:.+]] = llvm.bitcast %[[RES0_I32]] : vector<2xi32> to vector<4xf16>
    // CHECK-NEXT: llvm.fpext %[[RES0_F16]] : vector<4xf16> to vector<4xf32>
    %0 = pop.cast %f8e4 : !pop.simd<4, f8e4m3fn> to !pop.simd<4, f32>

    // CHECK-NEXT: %[[ARG1_I16:.+]] = llvm.bitcast %arg1 : vector<4xi8> to vector<2xi16>
    // CHECK-NEXT: %[[ARG1_I16_0:.+]] = llvm.extractelement %[[ARG1_I16]][%2 : i32] : vector<2xi16>
    // CHECK-NEXT: %[[ARG1_I32_0:.+]] = llvm.inline_asm asm_dialect = att "cvt.rn.f16x2.e5m2x2 $0, $1;", "=r,h" %[[ARG1_I16_0]] : (i16) -> i32
    // CHECK-NEXT: %[[RES1_I32_0:.+]] = llvm.insertelement %[[ARG1_I32_0]], %3[%2 : i32] : vector<2xi32>
    // CHECK-NEXT: %[[ARG1_I16_1:.+]] = llvm.extractelement %[[ARG1_I16]][%1 : i32] : vector<2xi16>
    // CHECK-NEXT: %[[ARG1_I32_1:.+]] = llvm.inline_asm asm_dialect = att "cvt.rn.f16x2.e5m2x2 $0, $1;", "=r,h" %[[ARG1_I16_1]] : (i16) -> i32
    // CHECK-NEXT: %[[RES1_I32:.+]] = llvm.insertelement %[[ARG1_I32_1]], %[[RES1_I32_0]][%1 : i32] : vector<2xi32>
    // CHECK-NEXT: %[[RES1_F16:.+]] = llvm.bitcast %[[RES1_I32]] : vector<2xi32> to vector<4xf16>
    // CHECK-NEXT: llvm.fpext %[[RES1_F16]] : vector<4xf16> to vector<4xf32>
    %1 = pop.cast %f8e5  : !pop.simd<4, f8e5m2> to !pop.simd<4, f32>
    kgen.return %0, %1: !pop.simd<4, f32>, !pop.simd<4, f32>
  }

  // CHECK-LABEL: scalar_cast_f8_to_f16
  kgen.func @scalar_cast_f8_to_f16(%f8e4: !pop.simd<1, f8e4m3fn>, %f8e5: !pop.simd<1, f8e5m2>) -> (!pop.simd<1, f16>, !pop.simd<1, f16>) {
    // CHECK: %[[ARG0_I16:.+]] = llvm.zext %arg0 : i8 to i16
    // CHECK-NEXT: %[[ARG0_I32:.+]] = llvm.inline_asm asm_dialect = att "cvt.rn.f16x2.e4m3x2 $0, $1;", "=r,h" %[[ARG0_I16]] : (i16) -> i32
    // CHECK-NEXT: %[[ARG0_I16:.+]] = llvm.trunc %[[ARG0_I32]] : i32 to i16
    // CHECK-NEXT: %[[ARG0_F16:.+]] = llvm.bitcast %[[ARG0_I16]] : i16 to f16
    %0 = pop.cast %f8e4 : !pop.simd<1, f8e4m3fn> to !pop.simd<1, f16>

    // CHECK: %[[ARG1_I16:.+]] = llvm.zext %arg1 : i8 to i16
    // CHECK-NEXT: %[[ARG1_I32:.+]] = llvm.inline_asm asm_dialect = att "cvt.rn.f16x2.e5m2x2 $0, $1;", "=r,h" %[[ARG1_I16]] : (i16) -> i32
    // CHECK-NEXT: %[[ARG1_I16:.+]] = llvm.trunc %[[ARG1_I32]] : i32 to i16
    // CHECK-NEXT: %[[ARG1_F16:.+]] = llvm.bitcast %[[ARG1_I16]] : i16 to f16
    %1 = pop.cast %f8e5  : !pop.simd<1, f8e5m2> to !pop.simd<1, f16>
    kgen.return %0, %1: !pop.simd<1, f16>, !pop.simd<1, f16>
  }

  // CHECK-LABEL: scalar_cast_f8_to_f32
  kgen.func @scalar_cast_f8_to_f32(%f8e4: !pop.simd<1, f8e4m3fn>, %f8e5: !pop.simd<1, f8e5m2>) -> (!pop.simd<1, f32>, !pop.simd<1, f32>) {
    // CHECK: %[[ARG0_I16:.+]] = llvm.zext %arg0 : i8 to i16
    // CHECK-NEXT: %[[ARG0_I32:.+]] = llvm.inline_asm asm_dialect = att "cvt.rn.f16x2.e4m3x2 $0, $1;", "=r,h" %[[ARG0_I16]] : (i16) -> i32
    // CHECK-NEXT: %[[ARG0_I16:.+]] = llvm.trunc %[[ARG0_I32]] : i32 to i16
    // CHECK-NEXT: %[[ARG0_F16:.+]] = llvm.bitcast %[[ARG0_I16]] : i16 to f16
    // CHECK-NEXT: %[[RES0_F32:.+]] = llvm.fpext %[[ARG0_F16]] : f16 to f32
    %0 = pop.cast %f8e4 : !pop.simd<1, f8e4m3fn> to !pop.simd<1, f32>

    // CHECK: %[[ARG1_I16:.+]] = llvm.zext %arg1 : i8 to i16
    // CHECK-NEXT: %[[ARG1_I32:.+]] = llvm.inline_asm asm_dialect = att "cvt.rn.f16x2.e5m2x2 $0, $1;", "=r,h" %[[ARG1_I16]] : (i16) -> i32
    // CHECK-NEXT: %[[ARG1_I16:.+]] = llvm.trunc %[[ARG1_I32]] : i32 to i16
    // CHECK-NEXT: %[[ARG1_F16:.+]] = llvm.bitcast %[[ARG1_I16]] : i16 to f16
    // CHECK-NEXT: %[[RES1_F32:.+]] = llvm.fpext %[[ARG1_F16]] : f16 to f32
    %1 = pop.cast %f8e5  : !pop.simd<1, f8e5m2> to !pop.simd<1, f32>
    kgen.return %0, %1: !pop.simd<1, f32>, !pop.simd<1, f32>
  }

  // CHECK-LABEL: scalar_cast_f32_to_f8
  kgen.func @scalar_cast_f32_to_f8(%f32: !pop.simd<1, f32>) -> (!pop.simd<1, f8e4m3fn>, !pop.simd<1, f8e5m2>) {
    // CHECK-DAG:    %[[FP32_ZERO:.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f3
    // CHECK-DAG:    %[[I16_RES0:.+]] = llvm.inline_asm asm_dialect = att "cvt.rn.satfinite.e4m3x2.f32 $0, $1, $2;", "=h,f,f" %[[FP32_ZERO]], %arg0 : (f32, f32) -> i16
    // CHECK-DAG:    %[[VEC_F8E4_RES:.+]] = llvm.bitcast %[[I16_RES0]] : i16 to i8
    %0 = pop.cast %f32 : !pop.simd<1, f32> to !pop.simd<1, f8e4m3fn>
    // CHECK-DAG:    %[[I16_RES1:.+]] = llvm.inline_asm asm_dialect = att "cvt.rn.satfinite.e5m2x2.f32 $0, $1, $2;", "=h,f,f" %[[FP32_ZERO]], %arg0 : (f32, f32) -> i16
    // CHECK-DAG:    %[[VEC_F8E5_RES:.+]] = llvm.bitcast %[[I16_RES1]] : i16 to i8
    %1 = pop.cast %f32 : !pop.simd<1, f32> to !pop.simd<1, f8e5m2>
    kgen.return %0, %1: !pop.simd<1, f8e4m3fn>, !pop.simd<1, f8e5m2>
  }
}
