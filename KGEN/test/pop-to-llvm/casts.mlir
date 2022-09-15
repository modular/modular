// RUN: kgen-opt -split-input-file -convert-pop-to-llvm -convert-kgen-to-llvm -canonicalize %s | FileCheck %s

// CHECK-LABEL: @scalar_bitcast
// CHECK-SAME: %[[UI32:[a-z0-9]+]]:
// CHECK-SAME: %[[F32:[a-z0-9]+]]:
// CHECK-SAME: %[[F64:[a-z0-9]+]]:
kgen.func @scalar_bitcast(
    %ui32: !meta.scalar<ui32>,
    %f32: !meta.scalar<f32>,
    %f64: !meta.scalar<f64>) -> (
      !meta.scalar<f32>,
      !meta.scalar<si32>,
      !meta.scalar<ui64>
    ) {
  // CHECK: llvm.bitcast %[[UI32]]
  %0 = pop.bitcast %ui32 : !meta.scalar<ui32> to !meta.scalar<f32>
  // CHECK: llvm.bitcast %[[F32]]
  %1 = pop.bitcast %f32 : !meta.scalar<f32> to !meta.scalar<si32>
  // CHECK: llvm.bitcast %[[F64]]
  %2 = pop.bitcast %f64 : !meta.scalar<f64> to !meta.scalar<ui64>
  kgen.return %0, %1, %2 :
      !meta.scalar<f32>,
      !meta.scalar<si32>,
      !meta.scalar<ui64>
}

// CHECK-LABEL: @simd_bitcast
// CHECK-SAME: %[[UI32:[a-z0-9]+]]:
// CHECK-SAME: %[[F32:[a-z0-9]+]]:
// CHECK-SAME: %[[F64:[a-z0-9]+]]:
kgen.func @simd_bitcast(
    %ui32:!meta.simd<4, ui32>,
    %f32:!meta.simd<4, f32>,
    %f64:!meta.simd<2, f64>) -> (
     !meta.simd<4, f32>,
     !meta.simd<4, si32>,
     !meta.simd<4, ui32>
    ) {
  // CHECK: lvm.bitcast %[[UI32]]
  %0 = pop.bitcast %ui32 :!meta.simd<4, ui32> to !meta.simd<4, f32>
  // CHECK: llvm.bitcast %[[F32]]
  %1 = pop.bitcast %f32 :!meta.simd<4, f32> to !meta.simd<4, si32>
  // CHECK: llvm.bitcast %[[F64]]
  %2 = pop.bitcast %f64 :!meta.simd<2, f64> to !meta.simd<4, ui32>
  kgen.return %0, %1, %2 :
     !meta.simd<4, f32>,
     !meta.simd<4, si32>,
     !meta.simd<4, ui32>
}

// CHECK-LABEL: @pointer_bitcast
// CHECK-SAME: %[[UI32:[a-z0-9]+]]:
// CHECK-SAME: %[[F32:[a-z0-9]+]]:
// CHECK-SAME: %[[F64:[a-z0-9]+]]:
kgen.func @pointer_bitcast(
    %ui32:!meta.pointer<!meta.scalar<ui32>>,
    %simd_f32:!meta.pointer<!meta.simd<4, f32>>,
    %simd_f64:!meta.pointer<!meta.simd<2, f64>>) -> (
     !meta.pointer<!meta.simd<4, f32>>,
     !meta.pointer<!meta.scalar<si32>>,
     !meta.pointer<!meta.scalar<ui32>>
    ) {
  // CHECK: lvm.bitcast %[[UI32]]
  %0 = pop.bitcast %ui32 : !meta.pointer<!meta.scalar<ui32>> to !meta.pointer<!meta.simd<4, f32>>
  // CHECK: llvm.bitcast %[[F32]]
  %1 = pop.bitcast %simd_f32 : !meta.pointer<!meta.simd<4, f32>> to !meta.pointer<!meta.scalar<si32>>
  // CHECK: llvm.bitcast %[[F64]]
  %2 = pop.bitcast %simd_f64 : !meta.pointer<!meta.simd<2, f64>> to !meta.pointer<!meta.scalar<ui32>>
  kgen.return %0, %1, %2 :
     !meta.pointer<!meta.simd<4, f32>>,
     !meta.pointer<!meta.scalar<si32>>,
     !meta.pointer<!meta.scalar<ui32>>
}

// CHECK-LABEL: @scalar_cast
// CHECK-SAME: %[[UI32:[a-z0-9]+]]:
// CHECK-SAME: %[[SI32:[a-z0-9]+]]:
// CHECK-SAME: %[[F32:[a-z0-9]+]]:
// CHECK-SAME: %[[F64:[a-z0-9]+]]:
kgen.func @scalar_cast(
    %ui32: !meta.scalar<ui32>,
    %si32: !meta.scalar<si32>,
    %f32: !meta.scalar<f32>,
    %f64: !meta.scalar<f64>) -> (
    !meta.scalar<ui64>,
    !meta.scalar<si64>,
    !meta.scalar<ui16>,
    !meta.scalar<si32>,
    !meta.scalar<f64>,
    !meta.scalar<f32>,
    !meta.scalar<si64>,
    !meta.scalar<ui32>,
    !meta.scalar<f64>,
    !meta.scalar<f32>,
    !meta.scalar<f32>
    ) {
  // CHECK: %[[V0:.*]] = llvm.sext %[[SI32]]
  %0 = pop.cast %si32 : !meta.scalar<si32> to !meta.scalar<ui64>
  // CHECK: %[[V1:.*]] = llvm.zext %[[UI32]]
  %1 = pop.cast %ui32 : !meta.scalar<ui32> to !meta.scalar<si64>
  // CHECK: %[[V2:.*]] = llvm.trunc %[[SI32]]
  %2 = pop.cast %si32 : !meta.scalar<si32> to !meta.scalar<ui16>
  %3 = pop.cast %ui32 : !meta.scalar<ui32> to !meta.scalar<si32>
  // CHECK: %[[V4:.*]] = llvm.sitofp %[[SI32]]
  %4 = pop.cast %si32 : !meta.scalar<si32> to !meta.scalar<f64>
  // CHECK: %[[V5:.*]] = llvm.uitofp %[[UI32]]
  %5 = pop.cast %ui32 : !meta.scalar<ui32> to !meta.scalar<f32>
  // CHECK: %[[V6:.*]] = llvm.fptosi %[[F32]]
  %6 = pop.cast %f32 : !meta.scalar<f32> to !meta.scalar<si64>
  // CHECK: %[[V7:.*]] = llvm.fptoui %[[F64]]
  %7 = pop.cast %f64 : !meta.scalar<f64> to !meta.scalar<ui32>
  // CHECK: %[[V8:.*]] = llvm.fpext %[[F32]]
  %8 = pop.cast %f32 : !meta.scalar<f32> to !meta.scalar<f64>
  // CHECK: %[[V9:.*]] = llvm.fptrunc %[[F64]]
  %9 = pop.cast %f64 : !meta.scalar<f64> to !meta.scalar<f32>
  %10 = pop.cast %f32 : !meta.scalar<f32> to !meta.scalar<f32>
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
    !meta.scalar<ui64>,
    !meta.scalar<si64>,
    !meta.scalar<ui16>,
    !meta.scalar<si32>,
    !meta.scalar<f64>,
    !meta.scalar<f32>,
    !meta.scalar<si64>,
    !meta.scalar<ui32>,
    !meta.scalar<f64>,
    !meta.scalar<f32>,
    !meta.scalar<f32>
}

// CHECK-LABEL: @simd_cast
// CHECK-SAME: %[[UI32:[a-z0-9]+]]:
// CHECK-SAME: %[[SI32:[a-z0-9]+]]:
// CHECK-SAME: %[[F32:[a-z0-9]+]]:
// CHECK-SAME: %[[F64:[a-z0-9]+]]:
kgen.func @simd_cast(
    %ui32: !meta.simd<2, ui32>,
    %si32: !meta.simd<2, si32>,
    %f32: !meta.simd<2, f32>,
    %f64: !meta.simd<2, f64>) -> (
    !meta.simd<2, ui64>,
    !meta.simd<2, si64>,
    !meta.simd<2, ui16>,
    !meta.simd<2, si32>,
    !meta.simd<2, f64>,
    !meta.simd<2, f32>,
    !meta.simd<2, si64>,
    !meta.simd<2, ui32>,
    !meta.simd<2, f64>,
    !meta.simd<2, f32>,
    !meta.simd<2, f32>
    ) {
  // CHECK: %[[V0:.*]] = llvm.sext %[[SI32]]
  %0 = pop.cast %si32 : !meta.simd<2, si32> to !meta.simd<2, ui64>
  // CHECK: %[[V1:.*]] = llvm.zext %[[UI32]]
  %1 = pop.cast %ui32 : !meta.simd<2, ui32> to !meta.simd<2, si64>
  // CHECK: %[[V2:.*]] = llvm.trunc %[[SI32]]
  %2 = pop.cast %si32 : !meta.simd<2, si32> to !meta.simd<2, ui16>
  %3 = pop.cast %ui32 : !meta.simd<2, ui32> to !meta.simd<2, si32>
  // CHECK: %[[V4:.*]] = llvm.sitofp %[[SI32]]
  %4 = pop.cast %si32 : !meta.simd<2, si32> to !meta.simd<2, f64>
  // CHECK: %[[V5:.*]] = llvm.uitofp %[[UI32]]
  %5 = pop.cast %ui32 : !meta.simd<2, ui32> to !meta.simd<2, f32>
  // CHECK: %[[V6:.*]] = llvm.fptosi %[[F32]]
  %6 = pop.cast %f32 : !meta.simd<2, f32> to !meta.simd<2, si64>
  // CHECK: %[[V7:.*]] = llvm.fptoui %[[F64]]
  %7 = pop.cast %f64 : !meta.simd<2, f64> to !meta.simd<2, ui32>
  // CHECK: %[[V8:.*]] = llvm.fpext %[[F32]]
  %8 = pop.cast %f32 : !meta.simd<2, f32> to !meta.simd<2, f64>
  // CHECK: %[[V9:.*]] = llvm.fptrunc %[[F64]]
  %9 = pop.cast %f64 : !meta.simd<2, f64> to !meta.simd<2, f32>
  %10 = pop.cast %f32 : !meta.simd<2, f32> to !meta.simd<2, f32>
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
    !meta.simd<2, ui64>,
    !meta.simd<2, si64>,
    !meta.simd<2, ui16>,
    !meta.simd<2, si32>,
    !meta.simd<2, f64>,
    !meta.simd<2, f32>,
    !meta.simd<2, si64>,
    !meta.simd<2, ui32>,
    !meta.simd<2, f64>,
    !meta.simd<2, f32>,
    !meta.simd<2, f32>
}
