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

// CHECK-LABEL: address_space_cast
//  CHECK-SAME: %[[ARG0:[a-z0-9]+]]:
//  CHECK-SAME: %[[ARG1:[a-z0-9]+]]:
kgen.func @address_space_cast(%arg0: !kgen.pointer<scalar<f32>>, %arg1: !kgen.pointer<f32>)
    -> (!kgen.pointer<scalar<f32>, 1>, !kgen.pointer<f32, 3>) {
  // CHECK: llvm.addrspacecast %[[ARG0]] : !llvm.ptr to !llvm.ptr<1>
  %0 = pop.pointer.addrspacecast %arg0 : !kgen.pointer<scalar<f32>> to !kgen.pointer<scalar<f32>, 1>
  // CHECK: llvm.addrspacecast %[[ARG1]] : !llvm.ptr to !llvm.ptr<3>
  %1 = pop.pointer.addrspacecast %arg1 : !kgen.pointer<f32> to !kgen.pointer<f32, 3>
  kgen.return %0, %1 : !kgen.pointer<scalar<f32>, 1>, !kgen.pointer<f32, 3>
}

}
