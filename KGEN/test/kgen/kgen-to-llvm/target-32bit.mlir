// RUN: kgen-opt %s -lower-kgen-to-llvm | FileCheck %s

module attributes {M.target_info = #M.target<triple = "aarch64-unknown-macosx", arch = "generic", features = "", data_layout = "e-m:o-p:32:32-Fi8-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32", simd_bit_width = 128>} {

// CHECK-LABEL: @index_32bit() -> i32
kgen.func @index_32bit() -> index {
  // CHECK-NEXT: %0 = llvm.mlir.constant(4 : i32) : i32
  %0 = kgen.param.constant = <4>
  // CHECK-NEXT: llvm.return %0 : i32
  kgen.return %0 : index
}

// CHECK-LABEL: @string_32bit
kgen.func @string_32bit() -> !kgen.string {
  // CHECK: [[SIZE:%.*]] = llvm.mlir.constant(6 : i32) : i32
  // CHECK: llvm.insertvalue [[SIZE]], {{.*}}[1] : !llvm.struct<(ptr, i32)>
  %0 = kgen.param.constant: string = <"abduld">
  kgen.return %0 : !kgen.string
}

kgen.func @variadic_32bit() -> !kgen.variadic<pointer<struct<(pointer<none>, index, index) memoryOnly>>> {
  // CHECK: [[V0:%.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: [[V1:%.*]] = llvm.alloca [[V0]] x !llvm.ptr : (i32) -> !llvm.ptr
  // CHECK: [[V2:%.*]] = llvm.mlir.undef : !llvm.struct<(ptr, i32)>
  // CHECK: [[V3:%.*]] = llvm.insertvalue [[V1]], [[V2]][0] : !llvm.struct<(ptr, i32)>
  // CHECK: [[V4:%.*]] = llvm.insertvalue [[V0]], [[V3]][1] : !llvm.struct<(ptr, i32)>
  %0 = kgen.param.constant: variadic<pointer<struct<(pointer<none>, index, index) memoryOnly>>> = <[]>
  kgen.return %0 : !kgen.variadic<pointer<struct<(pointer<none>, index, index) memoryOnly>>>
}

}
