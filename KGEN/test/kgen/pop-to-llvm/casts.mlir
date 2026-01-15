// RUN: kgen-opt -split-input-file -pass-pipeline='builtin.module(kgen.func(legalize-pop-operations,lower-pop-to-llvm),lower-kgen-to-llvm,canonicalize)' %s | FileCheck %s

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

module attributes {M.target_info = #M.target<triple = "amdgcn-amd-amdhsa", arch="gfx942", data_layout="">} {
  // CHECK-LABEL: scalar_cast_f32_bf16
  kgen.func @scalar_cast_f32_bf16(%f32: !pop.scalar<f32>) -> (!pop.scalar<bf16>, !pop.scalar<bf16>) {
    // CHECK: %[[MANTISSA_DIFF:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK-NEXT: %[[NAN:.*]] = llvm.mlir.constant(0x7FC00000 : f32) : f32
    // CHECK-NEXT: %[[ROUNDED_BIAS:.*]] = llvm.mlir.constant(32767 : i32) : i32
    // CHECK-NEXT: %[[UNORDERED_MASK:.*]] = llvm.inline_asm asm_dialect = att "v_cmp_u_f32 $0, $1, $1", "=s,v" %arg0 : (f32) -> i64
    // CHECK-NEXT: %[[LSB:.*]] = llvm.inline_asm asm_dialect = att "v_bfe_u32 $0, $1, 16, 1", "=v,v" %arg0 : (f32) -> i32
    // CHECK-NEXT: %[[ROUNDED_VALUE:.*]] = llvm.inline_asm asm_dialect = att "v_add3_u32 $0, $1, $2, $3", "=v,v,v,s" %arg0, %[[LSB]], %[[ROUNDED_BIAS]] : (f32, i32, i32) -> i32
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
    // CHECK-NEXT: %[[ROUNDED_VALUE_0:.*]] = llvm.inline_asm asm_dialect = att "v_add3_u32 $0, $1, $2, $3", "=v,v,v,s" %[[F32_0]], %[[LSB_0]], %[[ROUNDED_BIAS]] : (f32, i32, i32) -> i32
    // CHECK-NEXT: %[[FLOAT_BITS_0:.*]] = llvm.inline_asm asm_dialect = att "v_cndmask_b32 $0, $1, $2, $3", "=v,v,v,s" %[[ROUNDED_VALUE_0]], %[[NAN]], %[[UNORDERED_MASK_0]] : (i32, f32, i64) -> i32
    // CHECK-NEXT: %[[SHIFTED_I32_0:.*]] = llvm.lshr %[[FLOAT_BITS_0]], %[[MANTISSA_DIFF]] : i32
    // CHECK-NEXT: %[[SHIFTED_I16_0:.*]] = llvm.trunc %[[SHIFTED_I32_0]] : i32 to i16
    // CHECK-NEXT: %[[BF16_0:.*]] = llvm.bitcast %[[SHIFTED_I16_0]] : i16 to bf16
    // CHECK-NEXT: %[[BF16x2_0:.*]] = llvm.insertelement %[[BF16_0]], %[[UNDEF_BF16x2]][%[[ZERO]] : i32] : vector<2xbf16>

    // CHECK-NEXT: %[[F32_1:.*]] = llvm.extractelement %arg0[%[[ONE]] : i32] : vector<2xf32>
    // CHECK-NEXT: %[[UNORDERED_MASK_1:.*]] = llvm.inline_asm asm_dialect = att "v_cmp_u_f32 $0, $1, $1", "=s,v" %[[F32_1]] : (f32) -> i64
    // CHECK-NEXT: %[[LSB_1:.*]] = llvm.inline_asm asm_dialect = att "v_bfe_u32 $0, $1, 16, 1", "=v,v" %[[F32_1]] : (f32) -> i32
    // CHECK-NEXT: %[[ROUNDED_VALUE_1:.*]] = llvm.inline_asm asm_dialect = att "v_add3_u32 $0, $1, $2, $3", "=v,v,v,s" %[[F32_1]], %[[LSB_1]], %[[ROUNDED_BIAS]] : (f32, i32, i32) -> i32
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

  // CHECK-LABEL: scalar_cast_f32_f8
  kgen.func @scalar_cast_f32_f8(%f32: !pop.scalar<f32>) -> (!pop.scalar<f8e4m3fnuz>, !pop.scalar<f8e5m2fnuz>) {
    // CHECK-DAG: %[[MAXE5M2FNUZ:.+]] = llvm.mlir.constant(-5.734400e+04 : f32) : f32
    // CHECK-DAG: %[[MINE5M2FNUZ:.+]] = llvm.mlir.constant(5.734400e+04 : f32) : f32
    // CHECK-DAG: %[[FALSE:.+]] = llvm.mlir.constant(false) : i1
    // CHECK-DAG: %[[ZERO:.+]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG: %[[MINE4M3FNUZ:.+]] = llvm.mlir.constant(-2.400000e+02 : f32) : f32
    // CHECK-DAG: %[[MAXE4M3FNUZ:.+]] = llvm.mlir.constant(2.400000e+02 : f32) : f32

    // CHECK-DAG: %[[ARG0_CLAMPED0:.+]] = llvm.intr.maxnum(%arg0, %[[MINE4M3FNUZ]]) {fastmathFlags = #llvm.fastmath<contract>} : (f32, f32) -> f32
    // CHECK-DAG: %[[ARG0_CLAMPED1:.+]] = llvm.intr.minnum(%[[ARG0_CLAMPED0]], %[[MAXE4M3FNUZ]]) {fastmathFlags = #llvm.fastmath<contract>} : (f32, f32) -> f32
    // CHECK-DAG: %[[ISNAN0:.+]] = llvm.fcmp "uno" %arg0, %arg0 : f32
    // CHECK-DAG: %[[SEL0:.+]] = llvm.select %[[ISNAN0]], %arg0, %[[ARG0_CLAMPED1]] {fastmathFlags = #llvm.fastmath<contract>} : i1, f32
    // CHECK-DAG: %[[F8_0:.+]] = llvm.call_intrinsic "llvm.amdgcn.cvt.pk.fp8.f32"(%[[SEL0]], %[[SEL0]], %[[ZERO]], %[[FALSE]]) : (f32, f32, i32, i1) -> i32
    // CHECK-DAG: %[[F8_1:.+]] = llvm.trunc %[[F8_0]] : i32 to i8
    %0 = pop.cast %f32 : !pop.scalar<f32> to !pop.scalar<f8e4m3fnuz>

    // CHECK-DAG: %[[ARG0_CLAMPED2:.+]] = llvm.intr.maxnum(%arg0, %[[MAXE5M2FNUZ]]) {fastmathFlags = #llvm.fastmath<contract>} : (f32, f32) -> f32
    // CHECK-DAG: %[[ARG0_CLAMPED3:.+]] = llvm.intr.minnum(%[[ARG0_CLAMPED2]], %[[MINE5M2FNUZ]]) {fastmathFlags = #llvm.fastmath<contract>} : (f32, f32) -> f32
    // CHECK-DAG: %[[ISNAN1:.+]] = llvm.fcmp "uno" %arg0, %arg0 : f32
    // CHECK-DAG: %[[SEL1:.+]] = llvm.select %[[ISNAN1]], %arg0, %[[ARG0_CLAMPED3]] {fastmathFlags = #llvm.fastmath<contract>} : i1, f32
    // CHECK-DAG: %[[F8_2:.+]] = llvm.call_intrinsic "llvm.amdgcn.cvt.pk.bf8.f32"(%[[SEL1]], %[[SEL1]], %[[ZERO]], %[[FALSE]]) : (f32, f32, i32, i1) -> i32
    // CHECK-DAG: %[[F8_3:.+]] = llvm.trunc %[[F8_2]] : i32 to i8
    %1 = pop.cast %f32 : !pop.scalar<f32> to !pop.scalar<f8e5m2fnuz>

    kgen.return %0, %1 :
      !pop.scalar<f8e4m3fnuz>,
      !pop.scalar<f8e5m2fnuz>
  }

  // CHECK-LABEL: simd2_cast_f32_f8
  kgen.func @simd2_cast_f32_f8(%f32: !pop.simd<2, f32>) -> (!pop.simd<2, f8e4m3fnuz>, !pop.simd<2, f8e5m2fnuz>) {
    // CHECK-DAG: %[[MINE5M2FNUZ:.+]] = llvm.mlir.constant(#M.dense_array<-5.734400e+04, -5.734400e+04> : vector<2xf32>) : vector<2xf32>
    // CHECK-DAG: %[[MAXE5M2FNUZ:.+]] = llvm.mlir.constant(#M.dense_array<5.734400e+04, 5.734400e+04> : vector<2xf32>) : vector<2xf32>
    // CHECK-DAG: %[[FALSE:.+]] = llvm.mlir.constant(false) : i1
    // CHECK-DAG: %[[ONE:.+]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-DAG: %[[ZERO:.+]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG: %[[MINE4M3FNUZ:.+]] = llvm.mlir.constant(#M.dense_array<-2.400000e+02, -2.400000e+02> : vector<2xf32>) : vector<2xf32>
    // CHECK-DAG: %[[MAXE4M3FNUZ:.+]] = llvm.mlir.constant(#M.dense_array<2.400000e+02, 2.400000e+02> : vector<2xf32>) : vector<2xf32>

    // CHECK-DAG: %[[ARG0_CLAMPED0:.+]] = llvm.intr.maxnum(%arg0, %[[MINE4M3FNUZ]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<2xf32>, vector<2xf32>) -> vector<2xf32>
    // CHECK-DAG: %[[ARG0_CLAMPED1:.+]] = llvm.intr.minnum(%[[ARG0_CLAMPED0]], %[[MAXE4M3FNUZ]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<2xf32>, vector<2xf32>) -> vector<2xf32>
    // CHECK-DAG: %[[ISNAN0:.+]] = llvm.fcmp "uno" %arg0, %arg0 : vector<2xf32>
    // CHECK-DAG: %[[SEL0:.+]] = llvm.select %[[ISNAN0]], %arg0, %[[ARG0_CLAMPED1]] {fastmathFlags = #llvm.fastmath<contract>} : vector<2xi1>, vector<2xf32>
    // CHECK-DAG: %[[FP32_0:.+]] = llvm.extractelement %[[SEL0]][%[[ZERO]] : i32] : vector<2xf32>
    // CHECK-DAG: %[[FP32_1:.+]] = llvm.extractelement %[[SEL0]][%[[ONE]] : i32] : vector<2xf32>
    // CHECK-DAG: %[[FP8_2_0:.+]] = llvm.call_intrinsic "llvm.amdgcn.cvt.pk.fp8.f32"(%[[FP32_0]], %[[FP32_1]], %[[ZERO]], %[[FALSE]]) : (f32, f32, i32, i1) -> i32
    // CHECK-DAG: %[[FP8_2_1:.+]] = llvm.trunc %[[FP8_2_0]] : i32 to i16
    // CHECK-DAG: %[[FP8_2_2:.+]] = llvm.bitcast %[[FP8_2_1]] : i16 to vector<2xi8>
    %0 = pop.cast %f32 : !pop.simd<2, f32> to !pop.simd<2, f8e4m3fnuz>

    // CHECK-DAG: %[[ARG0_CLAMPED2:.+]] = llvm.intr.maxnum(%arg0, %[[MINE5M2FNUZ]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<2xf32>, vector<2xf32>) -> vector<2xf32>
    // CHECK-DAG: %[[ARG0_CLAMPED3:.+]] = llvm.intr.minnum(%[[ARG0_CLAMPED2]], %[[MAXE5M2FNUZ]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<2xf32>, vector<2xf32>) -> vector<2xf32>
    // CHECK-DAG: %[[ISNAN1:.+]] = llvm.fcmp "uno" %arg0, %arg0 : vector<2xf32>
    // CHECK-DAG: %[[SEL1:.+]] = llvm.select %[[ISNAN1]], %arg0, %[[ARG0_CLAMPED3]] {fastmathFlags = #llvm.fastmath<contract>} : vector<2xi1>, vector<2xf32>
    // CHECK-DAG: %[[FP32_2:.+]] = llvm.extractelement %[[SEL1]][%[[ZERO]] : i32] : vector<2xf32>
    // CHECK-DAG: %[[FP32_3:.+]] = llvm.extractelement %[[SEL1]][%[[ONE]] : i32] : vector<2xf32>
    // CHECK-DAG: %[[FP8_2_3:.+]] = llvm.call_intrinsic "llvm.amdgcn.cvt.pk.bf8.f32"(%[[FP32_2]], %[[FP32_3]], %[[ZERO]], %[[FALSE]]) : (f32, f32, i32, i1) -> i32
    // CHECK-DAG: %[[FP8_2_4:.+]] = llvm.trunc %[[FP8_2_3]] : i32 to i16
    // CHECK-DAG: %[[FP8_2_5:.+]] = llvm.bitcast %[[FP8_2_4]] : i16 to vector<2xi8>
    %1 = pop.cast %f32 : !pop.simd<2, f32> to !pop.simd<2, f8e5m2fnuz>

    kgen.return %0, %1 :
      !pop.simd<2, f8e4m3fnuz>,
      !pop.simd<2, f8e5m2fnuz>
  }

  // CHECK-LABEL: simd4_cast_f32_f8
  kgen.func @simd4_cast_f32_f8(%f32: !pop.simd<4, f32>) -> (!pop.simd<4, f8e4m3fnuz>, !pop.simd<4, f8e5m2fnuz>) {
    // CHECK-DAG: %[[MINE5M2FNUZ:.+]] = llvm.mlir.constant(#M.dense_array<-5.734400e+04, -5.734400e+04, -5.734400e+04, -5.734400e+04> : vector<4xf32>) : vector<4xf32>
    // CHECK-DAG: %[[MAXE5M2FNUZ:.+]] = llvm.mlir.constant(#M.dense_array<5.734400e+04, 5.734400e+04, 5.734400e+04, 5.734400e+04> : vector<4xf32>) : vector<4xf32>
    // CHECK-DAG: %[[TRUE:.+]] = llvm.mlir.constant(true) : i1
    // CHECK-DAG: %[[TWO:.+]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-DAG: %[[FALSE:.+]] = llvm.mlir.constant(false) : i1
    // CHECK-DAG: %[[ONE:.+]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-DAG: %[[ZERO:.+]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG: %[[OUT:.+]] = llvm.mlir.undef : vector<1xi32>
    // CHECK-DAG: %[[THREE:.+]] = llvm.mlir.constant(3 : i32) : i32
    // CHECK-DAG: %[[MINE4M3FNUZ:.+]] = llvm.mlir.constant(#M.dense_array<-2.400000e+02, -2.400000e+02, -2.400000e+02, -2.400000e+02> : vector<4xf32>) : vector<4xf32>
    // CHECK-DAG: %[[MAXE4M3FNUZ:.+]] = llvm.mlir.constant(#M.dense_array<2.400000e+02, 2.400000e+02, 2.400000e+02, 2.400000e+02> : vector<4xf32>) : vector<4xf32>

    // CHECK-DAG: %[[ARG0_CLAMPED0:.+]] = llvm.intr.maxnum(%arg0, %[[MINE4M3FNUZ]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    // CHECK-DAG: %[[ARG0_CLAMPED1:.+]] = llvm.intr.minnum(%[[ARG0_CLAMPED0]], %[[MAXE4M3FNUZ]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    // CHECK-DAG: %[[ISNAN0:.+]] = llvm.fcmp "uno" %arg0, %arg0 : vector<4xf32>
    // CHECK-DAG: %[[SEL0:.+]] = llvm.select %[[ISNAN0]], %arg0, %[[ARG0_CLAMPED1]] {fastmathFlags = #llvm.fastmath<contract>} : vector<4xi1>, vector<4xf32>
    // CHECK-DAG: %[[FP32_0:.+]] = llvm.extractelement %[[SEL0]][%[[ZERO]] : i32] : vector<4xf32>
    // CHECK-DAG: %[[FP32_1:.+]] = llvm.extractelement %[[SEL0]][%[[ONE]] : i32] : vector<4xf32>
    // CHECK-DAG: %[[WORD0:.+]] = llvm.call_intrinsic "llvm.amdgcn.cvt.pk.fp8.f32"(%[[FP32_0]], %[[FP32_1]], %[[ZERO]], %[[FALSE]]) : (f32, f32, i32, i1) -> i32
    // CHECK-DAG: %[[FP32_2:.+]] = llvm.extractelement %[[SEL0]][%[[TWO]] : i32] : vector<4xf32>
    // CHECK-DAG: %[[FP32_3:.+]] = llvm.extractelement %[[SEL0]][%[[THREE]] : i32] : vector<4xf32>
    // CHECK-DAG: %[[WORD1:.+]] = llvm.call_intrinsic "llvm.amdgcn.cvt.pk.fp8.f32"(%[[FP32_2]], %[[FP32_3]], %[[WORD0]], %[[TRUE]]) : (f32, f32, i32, i1) -> i32
    // CHECK-DAG: %[[FP8_4_0:.+]] = llvm.insertelement %[[WORD1]], %[[OUT]][%[[ZERO]] : i32] : vector<1xi32>
    // CHECK-DAG: %[[FP8_4_1:.+]] = llvm.bitcast %[[FP8_4_0]] : vector<1xi32> to vector<4xi8>
    %0 = pop.cast %f32 : !pop.simd<4, f32> to !pop.simd<4, f8e4m3fnuz>

    // CHECK-DAG: %[[ARG0_CLAMPED2:.+]] = llvm.intr.maxnum(%arg0, %[[MINE5M2FNUZ]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    // CHECK-DAG: %[[ARG0_CLAMPED3:.+]] = llvm.intr.minnum(%[[ARG0_CLAMPED2]], %[[MAXE5M2FNUZ]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    // CHECK-DAG: %[[ISNAN1:.+]] = llvm.fcmp "uno" %arg0, %arg0 : vector<4xf32>
    // CHECK-DAG: %[[SEL1:.+]] = llvm.select %[[ISNAN1]], %arg0, %[[ARG0_CLAMPED3]] {fastmathFlags = #llvm.fastmath<contract>} : vector<4xi1>, vector<4xf32>
    // CHECK-DAG: %[[FP32_4:.+]] = llvm.extractelement %[[SEL1]][%[[ZERO]] : i32] : vector<4xf32>
    // CHECK-DAG: %[[FP32_5:.+]] = llvm.extractelement %[[SEL1]][%[[ONE]] : i32] : vector<4xf32>
    // CHECK-DAG: %[[WORD2:.+]] = llvm.call_intrinsic "llvm.amdgcn.cvt.pk.bf8.f32"(%[[FP32_4]], %[[FP32_5]], %[[ZERO]], %[[FALSE]]) : (f32, f32, i32, i1) -> i32
    // CHECK-DAG: %[[FP32_6:.+]] = llvm.extractelement %[[SEL1]][%[[TWO]] : i32] : vector<4xf32>
    // CHECK-DAG: %[[FP32_7:.+]] = llvm.extractelement %[[SEL1]][%[[THREE]] : i32] : vector<4xf32>
    // CHECK-DAG: %[[WORD3:.+]] = llvm.call_intrinsic "llvm.amdgcn.cvt.pk.bf8.f32"(%[[FP32_6]], %[[FP32_7]], %[[WORD2]], %[[TRUE]]) : (f32, f32, i32, i1) -> i32
    // CHECK-DAG: %[[FP8_4_2:.+]] = llvm.insertelement %[[WORD3]], %[[OUT]][%[[ZERO]] : i32] : vector<1xi32>
    // CHECK-DAG: %[[FP8_4_3:.+]] = llvm.bitcast %[[FP8_4_2]] : vector<1xi32> to vector<4xi8>
    %1 = pop.cast %f32 : !pop.simd<4, f32> to !pop.simd<4, f8e5m2fnuz>

    kgen.return %0, %1 :
      !pop.simd<4, f8e4m3fnuz>,
      !pop.simd<4, f8e5m2fnuz>
  }

  // CHECK-LABEL: scalar_cast_f8_f32
  kgen.func @scalar_cast_f8_f32(%f8: !pop.scalar<f8e4m3fnuz>) -> !pop.scalar<f32> {
    // CHECK-DAG: %[[ZERO:.+]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[INPUT0:.+]] = llvm.mlir.undef : vector<4xi8>
    // CHECK: %[[INPUT1:.+]] = llvm.insertelement %arg0, %[[INPUT0]][%[[ZERO]] : i32] : vector<4xi8>
    // CHECK: %[[INPUT2:.+]] = llvm.bitcast %[[INPUT1]] : vector<4xi8> to i32
    // CHECK: %[[RES:.+]] = llvm.call_intrinsic "llvm.amdgcn.cvt.f32.fp8"(%[[INPUT2]], %[[ZERO]]) : (i32, i32) -> f32
    // CHECK: llvm.return %[[RES]] : f32
    %0 = pop.cast %f8: !pop.scalar<f8e4m3fnuz> to !pop.scalar<f32>
    kgen.return %0 : !pop.scalar<f32>
  }

  // CHECK-LABEL: simd2_cast_f8_f32
  kgen.func @simd2_cast_f8_f32(%f8: !pop.simd<2, f8e4m3fnuz>) -> !pop.simd<2, f32> {
    // CHECK-DAG: %[[FALSE:.+]] = llvm.mlir.constant(false) : i1
    // CHECK-DAG: %[[ONE:.+]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-DAG: %[[ZERO:.+]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG: %[[INPUT0:.+]] = llvm.mlir.undef : vector<4xi8>
    // CHECK: %[[FP8_0:.+]] = llvm.extractelement %arg0[%[[ZERO]] : i32] : vector<2xi8>
    // CHECK: %[[INPUT1:.+]] = llvm.insertelement %[[FP8_0]], %[[INPUT0]][%[[ZERO]] : i32] : vector<4xi8>
    // CHECK: %[[FP8_1:.+]] = llvm.extractelement %arg0[%[[ONE]] : i32] : vector<2xi8>
    // CHECK: %[[INPUT2:.+]] = llvm.insertelement %[[FP8_1]], %[[INPUT1]][%[[ONE]] : i32] : vector<4xi8>
    // CHECK: %[[INPUT3:.+]] = llvm.bitcast %[[INPUT2]] : vector<4xi8> to i32
    // CHECK: %[[RES:.+]] = llvm.call_intrinsic "llvm.amdgcn.cvt.pk.f32.fp8"(%[[INPUT3]], %[[FALSE]]) : (i32, i1) -> vector<2xf32>
    // CHECK: llvm.return %[[RES]] : vector<2xf32>
    %0 = pop.cast %f8 : !pop.simd<2, f8e4m3fnuz> to !pop.simd<2, f32>
    kgen.return %0 : !pop.simd<2, f32>
  }

  // CHECK-LABEL: simd4_cast_f8_f32
  kgen.func @simd4_cast_f8_f32(%f8: !pop.simd<4, f8e4m3fnuz>) -> !pop.simd<4, f32> {
    // CHECK-DAG: %[[TRUE:.+]] = llvm.mlir.constant(true) : i1
    // CHECK-DAG: %[[FALSE:.+]] = llvm.mlir.constant(false) : i1
    // CHECK-DAG: %[[THREE:.+]] = llvm.mlir.constant(3 : i32) : i32
    // CHECK-DAG: %[[TWO:.+]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-DAG: %[[ONE:.+]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-DAG: %[[ZERO:.+]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG: %[[RES0:.+]] = llvm.mlir.undef : vector<4xf32>
    // CHECK-DAG: %[[INPUT0:.+]] = llvm.mlir.undef : vector<4xi8>
    // CHECK: %[[FP8_0:.+]] = llvm.extractelement %arg0[%[[ZERO]] : i32] : vector<4xi8>
    // CHECK: %[[INPUT1:.+]] = llvm.insertelement %[[FP8_0]], %[[INPUT0]][%[[ZERO]] : i32] : vector<4xi8>
    // CHECK: %[[FP8_1:.+]] = llvm.extractelement %arg0[%[[ONE]] : i32] : vector<4xi8>
    // CHECK: %[[INPUT2:.+]] = llvm.insertelement %[[FP8_1]], %[[INPUT1]][%[[ONE]] : i32] : vector<4xi8>
    // CHECK: %[[FP8_2:.+]] = llvm.extractelement %arg0[%[[TWO]] : i32] : vector<4xi8>
    // CHECK: %[[INPUT3:.+]] = llvm.insertelement %[[FP8_2]], %[[INPUT2]][%[[TWO]] : i32] : vector<4xi8>
    // CHECK: %[[FP8_3:.+]] = llvm.extractelement %arg0[%[[THREE]] : i32] : vector<4xi8>
    // CHECK: %[[INPUT4:.+]] = llvm.insertelement %[[FP8_3]], %[[INPUT3]][%[[THREE]] : i32] : vector<4xi8>
    // CHECK: %[[INPUT5:.+]] = llvm.bitcast %[[INPUT4]] : vector<4xi8> to i32
    // CHECK: %[[RES1:.+]] = llvm.call_intrinsic "llvm.amdgcn.cvt.pk.f32.fp8"(%[[INPUT5]], %[[FALSE]]) : (i32, i1) -> vector<2xf32>
    // CHECK: %[[RES2:.+]] = llvm.call_intrinsic "llvm.amdgcn.cvt.pk.f32.fp8"(%[[INPUT5]], %[[TRUE]]) : (i32, i1) -> vector<2xf32>
    // CHECK: %[[FP32_0:.+]] = llvm.extractelement %[[RES1]][%[[ZERO]] : i32] : vector<2xf32>
    // CHECK: %[[RES3:.+]] = llvm.insertelement %[[FP32_0]], %[[RES0]][%[[ZERO]] : i32] : vector<4xf32>
    // CHECK: %[[FP32_1:.+]] = llvm.extractelement %[[RES1]][%[[ONE]] : i32] : vector<2xf32>
    // CHECK: %[[RES4:.+]] = llvm.insertelement %[[FP32_1]], %[[RES3]][%[[ONE]] : i32] : vector<4xf32>
    // CHECK: %[[FP32_2:.+]] = llvm.extractelement %[[RES2]][%[[ZERO]] : i32] : vector<2xf32>
    // CHECK: %[[RES5:.+]] = llvm.insertelement %[[FP32_2]], %[[RES4]][%[[TWO]] : i32] : vector<4xf32>
    // CHECK: %[[FP32_3:.+]] = llvm.extractelement %[[RES2]][%[[ONE]] : i32] : vector<2xf32>
    // CHECK: %[[RES6:.+]] = llvm.insertelement %[[FP32_3]], %[[RES5]][%[[THREE]] : i32] : vector<4xf32>
    // CHECK: llvm.return %[[RES6]] : vector<4xf32>
    %0 = pop.cast %f8 : !pop.simd<4, f8e4m3fnuz> to !pop.simd<4, f32>
    kgen.return %0 : !pop.simd<4, f32>
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
    // CHECK-DAG:    %[[VEC_F8E4_RES:.+]] = llvm.trunc %[[I16_RES0]] : i16 to i8
    %0 = pop.cast %f32 : !pop.simd<1, f32> to !pop.simd<1, f8e4m3fn>
    // CHECK-DAG:    %[[I16_RES1:.+]] = llvm.inline_asm asm_dialect = att "cvt.rn.satfinite.e5m2x2.f32 $0, $1, $2;", "=h,f,f" %[[FP32_ZERO]], %arg0 : (f32, f32) -> i16
    // CHECK-DAG:    %[[VEC_F8E5_RES:.+]] = llvm.trunc %[[I16_RES1]] : i16 to i8
    %1 = pop.cast %f32 : !pop.simd<1, f32> to !pop.simd<1, f8e5m2>
    kgen.return %0, %1: !pop.simd<1, f8e4m3fn>, !pop.simd<1, f8e5m2>
  }

  // CHECK-LABEL: simd_cast_f8_to_bf16
  kgen.func @simd_cast_f8_to_bf16(%f8e4: !pop.simd<4, f8e4m3fn>, %f8e5: !pop.simd<4, f8e5m2>) -> (!pop.simd<4, bf16>, !pop.simd<4, bf16>) {
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
    // CHECK-NEXT: %[[RES0_F32:.+]] = llvm.fpext %[[RES0_F16]] : vector<4xf16> to vector<4xf32>
    // CHECK-NEXT: %[[RES0_BF16:.+]] = llvm.fptrunc %[[RES0_F32]] : vector<4xf32> to vector<4xbf16>
    %0 = pop.cast %f8e4 : !pop.simd<4, f8e4m3fn> to !pop.simd<4, bf16>

    // CHECK-NEXT: %[[ARG1_I16:.+]] = llvm.bitcast %arg1 : vector<4xi8> to vector<2xi16>
    // CHECK-NEXT: %[[ARG1_I16_0:.+]] = llvm.extractelement %[[ARG1_I16]][%2 : i32] : vector<2xi16>
    // CHECK-NEXT: %[[ARG1_I32_0:.+]] = llvm.inline_asm asm_dialect = att "cvt.rn.f16x2.e5m2x2 $0, $1;", "=r,h" %[[ARG1_I16_0]] : (i16) -> i32
    // CHECK-NEXT: %[[RES1_I32_0:.+]] = llvm.insertelement %[[ARG1_I32_0]], %3[%2 : i32] : vector<2xi32>
    // CHECK-NEXT: %[[ARG1_I16_1:.+]] = llvm.extractelement %[[ARG1_I16]][%1 : i32] : vector<2xi16>
    // CHECK-NEXT: %[[ARG1_I32_1:.+]] = llvm.inline_asm asm_dialect = att "cvt.rn.f16x2.e5m2x2 $0, $1;", "=r,h" %[[ARG1_I16_1]] : (i16) -> i32
    // CHECK-NEXT: %[[RES1_I32:.+]] = llvm.insertelement %[[ARG1_I32_1]], %[[RES1_I32_0]][%1 : i32] : vector<2xi32>
    // CHECK-NEXT: %[[RES1_F16:.+]] = llvm.bitcast %[[RES1_I32]] : vector<2xi32> to vector<4xf16>
    // CHECK-NEXT: %[[RES1_F32:.+]] = llvm.fpext %[[RES1_F16]] : vector<4xf16> to vector<4xf32>
    // CHECK-NEXT: %[[RES1_BF16:.+]] = llvm.fptrunc %[[RES1_F32]] : vector<4xf32> to vector<4xbf16>
    %1 = pop.cast %f8e5  : !pop.simd<4, f8e5m2> to !pop.simd<4, bf16>
    kgen.return %0, %1: !pop.simd<4, bf16>, !pop.simd<4, bf16>
  }

  // CHECK-LABEL: scalar_cast_f8_to_bf16
  kgen.func @scalar_cast_f8_to_bf16(%f8e4: !pop.simd<1, f8e4m3fn>, %f8e5: !pop.simd<1, f8e5m2>) -> (!pop.simd<1, bf16>, !pop.simd<1, bf16>) {
    // CHECK: %[[ARG0_I16:.+]] = llvm.zext %arg0 : i8 to i16
    // CHECK-NEXT: %[[ARG0_I32:.+]] = llvm.inline_asm asm_dialect = att "cvt.rn.f16x2.e4m3x2 $0, $1;", "=r,h" %[[ARG0_I16]] : (i16) -> i32
    // CHECK-NEXT: %[[ARG0_I16:.+]] = llvm.trunc %[[ARG0_I32]] : i32 to i16
    // CHECK-NEXT: %[[ARG0_F16:.+]] = llvm.bitcast %[[ARG0_I16]] : i16 to f16
    // CHECK-NEXT: %[[RES0_F32:.+]] = llvm.fpext %[[ARG0_F16]] : f16 to f32
    // CHECK-NEXT: %[[RES0_BF16:.+]] = llvm.fptrunc %[[RES0_F32]] : f32 to bf16
    %0 = pop.cast %f8e4 : !pop.simd<1, f8e4m3fn> to !pop.simd<1, bf16>

    // CHECK: %[[ARG1_I16:.+]] = llvm.zext %arg1 : i8 to i16
    // CHECK-NEXT: %[[ARG1_I32:.+]] = llvm.inline_asm asm_dialect = att "cvt.rn.f16x2.e5m2x2 $0, $1;", "=r,h" %[[ARG1_I16]] : (i16) -> i32
    // CHECK-NEXT: %[[ARG1_I16:.+]] = llvm.trunc %[[ARG1_I32]] : i32 to i16
    // CHECK-NEXT: %[[ARG1_F16:.+]] = llvm.bitcast %[[ARG1_I16]] : i16 to f16
    // CHECK-NEXT: %[[RES1_F32:.+]] = llvm.fpext %[[ARG1_F16]] : f16 to f32
    // CHECK-NEXT: %[[RES1_BF16:.+]] = llvm.fptrunc %[[RES1_F32]] : f32 to bf16
    %1 = pop.cast %f8e5  : !pop.simd<1, f8e5m2> to !pop.simd<1, bf16>
    kgen.return %0, %1: !pop.simd<1, bf16>, !pop.simd<1, bf16>
  }

  // CHECK-LABEL: simd_cast_bf16_to_f8
  kgen.func @simd_cast_bf16_to_f8(%bf16: !pop.simd<4, bf16>) -> (!pop.simd<4, f8e4m3fn>, !pop.simd<4, f8e5m2>) {
    // CHECK-DAG:    %[[TWO:.+]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-DAG:    %[[THREE:.+]] = llvm.mlir.constant(3 : i32) : i32
    // CHECK-DAG:    %[[ZERO:.+]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG:    %[[ONE:.+]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-DAG:    %[[UNDEF_F8:.+]] = llvm.mlir.undef : vector<2xi16>
    // CHECK-DAG:    %[[FP32x2_0:.+]] = llvm.fpext %arg0 : vector<4xbf16> to vector<4xf32>
    // CHECK-DAG:    %[[VAL_F8E4_1:.+]] = llvm.extractelement %[[FP32x2_0]][%[[ONE]] : i32] : vector<4xf32>
    // CHECK-DAG:    %[[VAL_F8E4_0:.+]] = llvm.extractelement %[[FP32x2_0]][%[[ZERO]] : i32] : vector<4xf32>
    // CHECK-DAG:    %[[F8x2_01:.+]] = llvm.inline_asm asm_dialect = att "cvt.rn.satfinite.e4m3x2.f32 $0, $1, $2;", "=h,f,f" %[[VAL_F8E4_1]], %[[VAL_F8E4_0]] : (f32, f32) -> i16
    // CHECK-DAG:    %[[VEC_F8E4_RES_01:.+]] = llvm.insertelement %[[F8x2_01]], %[[UNDEF_F8]][%[[ZERO]] : i32] : vector<2xi16>
    // CHECK-DAG:    %[[VAL_F8E4_3:.+]] = llvm.extractelement %[[FP32x2_0]][%[[THREE]] : i32] : vector<4xf32>
    // CHECK-DAG:    %[[VAL_F8E4_2:.+]] = llvm.extractelement %[[FP32x2_0]][%[[TWO]] : i32] : vector<4xf32>
    // CHECK-DAG:    %[[VEC_F8E4_RES_23:.+]] = llvm.inline_asm asm_dialect = att "cvt.rn.satfinite.e4m3x2.f32 $0, $1, $2;", "=h,f,f" %[[VAL_F8E4_3]], %[[VAL_F8E4_2]] : (f32, f32) -> i16
    // CHECK-DAG:    %[[VEC_F8E4_RES_0123:.+]] = llvm.insertelement %[[VEC_F8E4_RES_23]], %[[VEC_F8E4_RES_01]][%[[ONE]] : i32] : vector<2xi16>
    // CHECK-DAG:    %[[VEC_F8E4_RES:.+]] = llvm.bitcast %[[VEC_F8E4_RES_0123]] : vector<2xi16> to vector<4xi8>
    %0 = pop.cast %bf16 : !pop.simd<4, bf16> to !pop.simd<4, f8e4m3fn>
    // CHECK-DAG:    %[[FP32x2_1:.+]] = llvm.fpext %arg0 : vector<4xbf16> to vector<4xf32>
    // CHECK-DAG:    %[[VAL_F8E5_1:.+]] = llvm.extractelement %[[FP32x2_1]][%[[ONE]] : i32] : vector<4xf32>
    // CHECK-DAG:    %[[VAL_F8E5_0:.+]] = llvm.extractelement %[[FP32x2_1]][%[[ZERO]] : i32] : vector<4xf32>
    // CHECK-DAG:    %[[F8x2_01:.+]] = llvm.inline_asm asm_dialect = att "cvt.rn.satfinite.e5m2x2.f32 $0, $1, $2;", "=h,f,f" %[[VAL_F8E5_1]], %[[VAL_F8E5_0]] : (f32, f32) -> i16
    // CHECK-DAG:    %[[VEC_F8E5_RES_01:.+]] = llvm.insertelement %[[F8x2_01]], %[[UNDEF_F8]][%[[ZERO]] : i32] : vector<2xi16>
    // CHECK-DAG:    %[[VAL_F8E5_3:.+]] = llvm.extractelement %[[FP32x2_1]][%[[THREE]] : i32] : vector<4xf32>
    // CHECK-DAG:    %[[VAL_F8E5_2:.+]] = llvm.extractelement %[[FP32x2_1]][%[[TWO]] : i32] : vector<4xf32>
    // CHECK-DAG:    %[[VEC_F8E5_RES_23:.+]] = llvm.inline_asm asm_dialect = att "cvt.rn.satfinite.e5m2x2.f32 $0, $1, $2;", "=h,f,f" %[[VAL_F8E5_3]], %[[VAL_F8E5_2]] : (f32, f32) -> i16
    // CHECK-DAG:    %[[VEC_F8E5_RES_0123:.+]] = llvm.insertelement %[[VEC_F8E5_RES_23]], %[[VEC_F8E5_RES_01]][%[[ONE]] : i32] : vector<2xi16>
    // CHECK-DAG:    %[[VEC_F8E5_RES:.+]] = llvm.bitcast %[[VEC_F8E5_RES_0123]] : vector<2xi16> to vector<4xi8>
    %1 = pop.cast %bf16 : !pop.simd<4, bf16> to !pop.simd<4, f8e5m2>
    kgen.return %0, %1: !pop.simd<4, f8e4m3fn>, !pop.simd<4, f8e5m2>
  }

  // CHECK-LABEL: scalar_cast_bf16_to_f8
  kgen.func @scalar_cast_bf16_to_f8(%bf16: !pop.simd<1, bf16>) -> (!pop.simd<1, f8e4m3fn>, !pop.simd<1, f8e5m2>) {
    // CHECK-DAG:    %[[FP32_ZERO:.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f3
    // CHECK-DAG:    %[[FP32_0:.+]] = llvm.fpext %arg0 : bf16 to f32
    // CHECK-DAG:    %[[I16_RES0:.+]] = llvm.inline_asm asm_dialect = att "cvt.rn.satfinite.e4m3x2.f32 $0, $1, $2;", "=h,f,f" %[[FP32_ZERO]], %[[FP32_0]] : (f32, f32) -> i16
    // CHECK-DAG:    %[[VEC_F8E4_RES:.+]] = llvm.trunc %[[I16_RES0]] : i16 to i8
    %0 = pop.cast %bf16 : !pop.simd<1, bf16> to !pop.simd<1, f8e4m3fn>
    // CHECK-DAG:    %[[FP32_1:.+]] = llvm.fpext %arg0 : bf16 to f32
    // CHECK-DAG:    %[[I16_RES1:.+]] = llvm.inline_asm asm_dialect = att "cvt.rn.satfinite.e5m2x2.f32 $0, $1, $2;", "=h,f,f" %[[FP32_ZERO]], %[[FP32_1]] : (f32, f32) -> i16
    // CHECK-DAG:    %[[VEC_F8E5_RES:.+]] = llvm.trunc %[[I16_RES1]] : i16 to i8
    %1 = pop.cast %bf16 : !pop.simd<1, bf16> to !pop.simd<1, f8e5m2>
    kgen.return %0, %1: !pop.simd<1, f8e4m3fn>, !pop.simd<1, f8e5m2>
  }
}

// -----

module attributes {M.target_info = #M.target<triple = "amdgcn-amd-amdhsa", arch="gfx1100", data_layout="">} {
  // CHECK-LABEL: scalar_cast_f32_bf16
  kgen.func @scalar_cast_f32_bf16(%f32: !pop.scalar<f32>) -> (!pop.scalar<bf16>, !pop.scalar<bf16>) {
    // CHECK-DAG: %[[UNDEF:.+]] = llvm.mlir.undef : !llvm.struct<(bf16, bf16)>
    // CHECK-DAG: %[[VAL0:.+]] = llvm.fptrunc %arg0 : f32 to bf16
    // CHECK-DAG: %[[VAL1:.+]] = llvm.fptrunc %arg0 : f32 to bf16
    // CHECK-DAG: %[[VAL2:.+]] = llvm.insertvalue %[[VAL0]], %[[UNDEF]][0] : !llvm.struct<(bf16, bf16)>
    // CHECK-DAG: %[[VAL3:.+]] = llvm.insertvalue %[[VAL1]], %[[VAL2]][1] : !llvm.struct<(bf16, bf16)>
    %0 = pop.cast fast %f32 : !pop.scalar<f32> to !pop.scalar<bf16>
    %1 = pop.cast %f32 : !pop.scalar<f32> to !pop.scalar<bf16>
    // CHECK-DAG: llvm.return %[[VAL3]] : !llvm.struct<(bf16, bf16)>
    kgen.return %0, %1 :
      !pop.scalar<bf16>,
      !pop.scalar<bf16>
  }

  // CHECK-LABEL: simd_cast_f32_bf16
  kgen.func @simd_cast_f32_bf16(%f32: !pop.simd<2, f32>) -> (!pop.simd<2, bf16>, !pop.simd<2, bf16>) {
    // CHECK-DAG: %[[UNDEF:.+]] = llvm.mlir.undef : !llvm.struct<(vector<2xbf16>, vector<2xbf16>)>
    // CHECK-DAG: %[[VAL0:.+]] = llvm.fptrunc %arg0 : vector<2xf32> to vector<2xbf16>
    // CHECK-DAG: %[[VAL1:.+]] = llvm.fptrunc %arg0 : vector<2xf32> to vector<2xbf16>
    // CHECK-DAG: %[[VAL2:.+]] = llvm.insertvalue %[[VAL0]], %[[UNDEF]][0] : !llvm.struct<(vector<2xbf16>, vector<2xbf16>)>
    // CHECK-DAG: %[[VAL3:.+]] = llvm.insertvalue %[[VAL1]], %[[VAL2]][1] : !llvm.struct<(vector<2xbf16>, vector<2xbf16>)>
    %0 = pop.cast fast %f32 : !pop.simd<2, f32> to !pop.simd<2, bf16>
    %1 = pop.cast %f32 : !pop.simd<2, f32> to !pop.simd<2, bf16>
    // CHECK-DAG: llvm.return %[[VAL3]] : !llvm.struct<(vector<2xbf16>, vector<2xbf16>)>
    kgen.return %0, %1 :
      !pop.simd<2, bf16>,
      !pop.simd<2, bf16>
  }
}

// -----

module attributes {M.target_info = #M.target<triple = "amdgcn-amd-amdhsa", arch="gfx950", data_layout="">} {
  // CHECK-LABEL: scalar_cast_f32_bf16
  kgen.func @scalar_cast_f32_bf16(%f32: !pop.scalar<f32>) -> (!pop.scalar<bf16>, !pop.scalar<bf16>) {
    // CHECK-DAG: %[[UNDEF:.+]] = llvm.mlir.undef : !llvm.struct<(bf16, bf16)>
    // CHECK-DAG: %[[VAL0:.+]] = llvm.fptrunc %arg0 : f32 to bf16
    // CHECK-DAG: %[[VAL1:.+]] = llvm.fptrunc %arg0 : f32 to bf16
    // CHECK-DAG: %[[VAL2:.+]] = llvm.insertvalue %[[VAL0]], %[[UNDEF]][0] : !llvm.struct<(bf16, bf16)>
    // CHECK-DAG: %[[VAL3:.+]] = llvm.insertvalue %[[VAL1]], %[[VAL2]][1] : !llvm.struct<(bf16, bf16)>
    %0 = pop.cast fast %f32 : !pop.scalar<f32> to !pop.scalar<bf16>
    %1 = pop.cast %f32 : !pop.scalar<f32> to !pop.scalar<bf16>
    // CHECK-DAG: llvm.return %[[VAL3]] : !llvm.struct<(bf16, bf16)>
    kgen.return %0, %1 :
      !pop.scalar<bf16>,
      !pop.scalar<bf16>
  }

  // CHECK-LABEL: simd_cast_f32_bf16
  kgen.func @simd_cast_f32_bf16(%f32: !pop.simd<2, f32>) -> (!pop.simd<2, bf16>, !pop.simd<2, bf16>) {
    // CHECK-DAG: %[[UNDEF:.+]] = llvm.mlir.undef : !llvm.struct<(vector<2xbf16>, vector<2xbf16>)>
    // CHECK-DAG: %[[VAL0:.+]] = llvm.fptrunc %arg0 : vector<2xf32> to vector<2xbf16>
    // CHECK-DAG: %[[VAL1:.+]] = llvm.fptrunc %arg0 : vector<2xf32> to vector<2xbf16>
    // CHECK-DAG: %[[VAL2:.+]] = llvm.insertvalue %[[VAL0]], %[[UNDEF]][0] : !llvm.struct<(vector<2xbf16>, vector<2xbf16>)>
    // CHECK-DAG: %[[VAL3:.+]] = llvm.insertvalue %[[VAL1]], %[[VAL2]][1] : !llvm.struct<(vector<2xbf16>, vector<2xbf16>)>
    %0 = pop.cast fast %f32 : !pop.simd<2, f32> to !pop.simd<2, bf16>
    %1 = pop.cast %f32 : !pop.simd<2, f32> to !pop.simd<2, bf16>
    // CHECK-DAG: llvm.return %[[VAL3]] : !llvm.struct<(vector<2xbf16>, vector<2xbf16>)>
    kgen.return %0, %1 :
      !pop.simd<2, bf16>,
      !pop.simd<2, bf16>
  }

  // CHECK-LABEL: simd_cast_f8_to_bf16_amd
  kgen.func @simd_cast_f8_to_bf16_amd(%f8e4: !pop.simd<4, f8e4m3fn>, %f8e5: !pop.simd<4, f8e5m2>) -> (!pop.simd<4, bf16>, !pop.simd<4, bf16>) {
    // For e4m3fn: First convert FP8 -> F32 using AMD intrinsics
    // CHECK-DAG: %[[ZERO:.+]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG: %[[ONE:.+]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-DAG: %[[FALSE:.+]] = llvm.mlir.constant(false) : i1
    // CHECK-DAG: %[[TRUE:.+]] = llvm.mlir.constant(true) : i1

    // Extract and convert FP8 elements to F32
    // CHECK: llvm.call_intrinsic "llvm.amdgcn.cvt.pk.f32.fp8"
    // Then convert F32 to BF16 using fptrunc
    // CHECK: llvm.fptrunc {{.*}} : vector<4xf32> to vector<4xbf16>
    %0 = pop.cast %f8e4 : !pop.simd<4, f8e4m3fn> to !pop.simd<4, bf16>

    // For e5m2: First convert FP8 -> F32 using AMD intrinsics
    // CHECK: llvm.call_intrinsic "llvm.amdgcn.cvt.pk.f32.bf8"
    // Then convert F32 to BF16 using fptrunc
    // CHECK: llvm.fptrunc {{.*}} : vector<4xf32> to vector<4xbf16>
    %1 = pop.cast %f8e5 : !pop.simd<4, f8e5m2> to !pop.simd<4, bf16>

    kgen.return %0, %1 : !pop.simd<4, bf16>, !pop.simd<4, bf16>
  }
}

// -----
module attributes {M.target_info = #M.target<triple = "air64-apple-macosx", arch = "", data_layout = "", simd_bit_width = 128>} {
  // CHECK-LABEL: @test_pop_cast_no_air(
  kgen.func @test_pop_cast_no_air(
    %si8_val: !pop.scalar<si8>,
    %ui8_val: !pop.scalar<ui8>,
    %si16_val: !pop.scalar<si16>,
    %ui16_val: !pop.scalar<ui16>,
    %si32_val: !pop.scalar<si32>,
    %ui32_val: !pop.scalar<ui32>,
    %f16_val: !pop.scalar<f16>,
    %bf16_val: !pop.scalar<bf16>,
    %f32_val: !pop.scalar<f32>
  ) -> (
    !pop.scalar<bf16>, !pop.scalar<f32>,
    !pop.scalar<si16>, !pop.scalar<ui16>, !pop.scalar<si32>, !pop.scalar<ui32>,
    !pop.scalar<si8>, !pop.scalar<ui8>, !pop.scalar<si32>, !pop.scalar<ui32>,
    !pop.scalar<si8>, !pop.scalar<ui8>, !pop.scalar<si32>, !pop.scalar<ui32>,
    !pop.scalar<si8>, !pop.scalar<ui8>, !pop.scalar<si16>, !pop.scalar<ui16>,
    !pop.scalar<si8>, !pop.scalar<ui8>, !pop.scalar<si16>, !pop.scalar<ui16>,
    !pop.scalar<f32>,
    !pop.scalar<f32>,
    !pop.scalar<f16>, !pop.scalar<bf16>
  ){
    // ========================================================================
    // from si8
    // ========================================================================
    // CHECK: %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %arg0 : i8 to !pop.scalar<si8>
    // CHECK: %[[CAST_0:.*]] = pop.cast %[[UNREALIZED_CONVERSION_CAST_0]] : !pop.scalar<si8> to !pop.scalar<bf16>
    %si8_to_bf16 = pop.cast %si8_val : !pop.scalar<si8> to !pop.scalar<bf16>

    // CHECK: %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[CAST_0]] : !pop.scalar<bf16> to bf16
    // CHECK: %[[CAST_1:.*]] = pop.cast %[[UNREALIZED_CONVERSION_CAST_0]] : !pop.scalar<si8> to !pop.scalar<f32>
    %si8_to_f32 = pop.cast %si8_val : !pop.scalar<si8> to !pop.scalar<f32>

    // ========================================================================
    // from ui8
    // ========================================================================
    // CHECK: %[[ZEXT_0:.*]] = llvm.zext %arg1 : i8 to i16
    %ui8_to_si16 = pop.cast %ui8_val : !pop.scalar<ui8> to !pop.scalar<si16>

    // CHECK: %[[ZEXT_1:.*]] = llvm.zext %arg1 : i8 to i16
    %ui8_to_ui16 = pop.cast %ui8_val : !pop.scalar<ui8> to !pop.scalar<ui16>

    // CHECK: %[[ZEXT_2:.*]] = llvm.zext %arg1 : i8 to i32
    %ui8_to_si32 = pop.cast %ui8_val : !pop.scalar<ui8> to !pop.scalar<si32>

    // CHECK: %[[ZEXT_3:.*]] = llvm.zext %arg1 : i8 to i32
    %ui8_to_ui32 = pop.cast %ui8_val : !pop.scalar<ui8> to !pop.scalar<ui32>

    // ========================================================================
    // from si16
    // ========================================================================
    // CHECK: %[[TRUNC_0:.*]] = llvm.trunc %arg2 : i16 to i8
    %si16_to_si8 = pop.cast %si16_val : !pop.scalar<si16> to !pop.scalar<si8>

    // CHECK: %[[TRUNC_1:.*]] = llvm.trunc %arg2 : i16 to i8
    %si16_to_ui8 = pop.cast %si16_val : !pop.scalar<si16> to !pop.scalar<ui8>

    // CHECK: %[[SEXT_0:.*]] = llvm.sext %arg2 : i16 to i32
    %si16_to_si32 = pop.cast %si16_val : !pop.scalar<si16> to !pop.scalar<si32>

    // CHECK: %[[SEXT_1:.*]] = llvm.sext %arg2 : i16 to i32
    %si16_to_ui32 = pop.cast %si16_val : !pop.scalar<si16> to !pop.scalar<ui32>

    // ========================================================================
    // from ui16
    // ========================================================================
    // CHECK: %[[TRUNC_2:.*]] = llvm.trunc %arg3 : i16 to i8
    %ui16_to_si8 = pop.cast %ui16_val : !pop.scalar<ui16> to !pop.scalar<si8>

    // CHECK: %[[TRUNC_3:.*]] = llvm.trunc %arg3 : i16 to i8
    %ui16_to_ui8 = pop.cast %ui16_val : !pop.scalar<ui16> to !pop.scalar<ui8>

    // CHECK: %[[ZEXT_4:.*]] = llvm.zext %arg3 : i16 to i32
    %ui16_to_si32 = pop.cast %ui16_val : !pop.scalar<ui16> to !pop.scalar<si32>

    // CHECK: %[[ZEXT_5:.*]] = llvm.zext %arg3 : i16 to i32
    %ui16_to_ui32 = pop.cast %ui16_val : !pop.scalar<ui16> to !pop.scalar<ui32>

    // ========================================================================
    // from si32
    // ========================================================================
    // CHECK: %[[TRUNC_4:.*]] = llvm.trunc %arg4 : i32 to i8
    %si32_to_si8 = pop.cast %si32_val : !pop.scalar<si32> to !pop.scalar<si8>

    // CHECK: %[[TRUNC_5:.*]] = llvm.trunc %arg4 : i32 to i8
    %si32_to_ui8 = pop.cast %si32_val : !pop.scalar<si32> to !pop.scalar<ui8>

    // CHECK: %[[TRUNC_6:.*]] = llvm.trunc %arg4 : i32 to i16
    %si32_to_si16 = pop.cast %si32_val : !pop.scalar<si32> to !pop.scalar<si16>

    // CHECK: %[[TRUNC_7:.*]] = llvm.trunc %arg4 : i32 to i16
    %si32_to_ui16 = pop.cast %si32_val : !pop.scalar<si32> to !pop.scalar<ui16>

    // ========================================================================
    // from ui32
    // ========================================================================
    // CHECK: %[[TRUNC_8:.*]] = llvm.trunc %arg5 : i32 to i8
    %ui32_to_si8 = pop.cast %ui32_val : !pop.scalar<ui32> to !pop.scalar<si8>

    // CHECK: %[[TRUNC_9:.*]] = llvm.trunc %arg5 : i32 to i8
    %ui32_to_ui8 = pop.cast %ui32_val : !pop.scalar<ui32> to !pop.scalar<ui8>

    // CHECK: %[[TRUNC_10:.*]] = llvm.trunc %arg5 : i32 to i16
    %ui32_to_si16 = pop.cast %ui32_val : !pop.scalar<ui32> to !pop.scalar<si16>

    // CHECK: %[[TRUNC_11:.*]] = llvm.trunc %arg5 : i32 to i16
    %ui32_to_ui16 = pop.cast %ui32_val : !pop.scalar<ui32> to !pop.scalar<ui16>

    // ========================================================================
    // from f16
    // ========================================================================
    // CHECK: %[[FPEXT_0:.*]] = llvm.fpext %arg6 : f16 to f32
    %f16_to_f32 = pop.cast %f16_val : !pop.scalar<f16> to !pop.scalar<f32>

    // ========================================================================
    // from bf16
    // ========================================================================
    // CHECK: %[[FPEXT_1:.*]] = llvm.fpext %arg7 : bf16 to f32
    %bf16_to_f32 = pop.cast %bf16_val : !pop.scalar<bf16> to !pop.scalar<f32>

    // ========================================================================
    // from f32
    // ========================================================================
    // CHECK: %[[FPTRUNC_0:.*]] = llvm.fptrunc %arg8 : f32 to f16
    %f32_to_f16 = pop.cast %f32_val : !pop.scalar<f32> to !pop.scalar<f16>

    // CHECK: %[[FPTRUNC_1:.*]] = llvm.fptrunc %arg8 : f32 to bf16
    %f32_to_bf16 = pop.cast %f32_val : !pop.scalar<f32> to !pop.scalar<bf16>

    kgen.return %si8_to_bf16, %si8_to_f32,
      %ui8_to_si16, %ui8_to_ui16, %ui8_to_si32, %ui8_to_ui32,
      %si16_to_si8, %si16_to_ui8, %si16_to_si32, %si16_to_ui32,
      %ui16_to_si8, %ui16_to_ui8, %ui16_to_si32, %ui16_to_ui32,
      %si32_to_si8, %si32_to_ui8, %si32_to_si16, %si32_to_ui16,
      %ui32_to_si8, %ui32_to_ui8, %ui32_to_si16, %ui32_to_ui16,
      %f16_to_f32,
      %bf16_to_f32,
      %f32_to_f16, %f32_to_bf16
      : !pop.scalar<bf16>, !pop.scalar<f32>,
        !pop.scalar<si16>, !pop.scalar<ui16>, !pop.scalar<si32>, !pop.scalar<ui32>,
        !pop.scalar<si8>, !pop.scalar<ui8>, !pop.scalar<si32>, !pop.scalar<ui32>,
        !pop.scalar<si8>, !pop.scalar<ui8>, !pop.scalar<si32>, !pop.scalar<ui32>,
        !pop.scalar<si8>, !pop.scalar<ui8>, !pop.scalar<si16>, !pop.scalar<ui16>,
        !pop.scalar<si8>, !pop.scalar<ui8>, !pop.scalar<si16>, !pop.scalar<ui16>,
        !pop.scalar<f32>,
        !pop.scalar<f32>,
        !pop.scalar<f16>, !pop.scalar<bf16>
  }
}
