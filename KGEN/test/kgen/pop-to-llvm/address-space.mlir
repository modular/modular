// RUN: kgen-opt -split-input-file -pass-pipeline='builtin.module(lower-global-pop-to-llvm,kgen.func(lower-pop-to-llvm))'  %s | FileCheck %s

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // CHECK-LABEL: llvm.mlir.global internal @global_load_global_alloc() {addr_space = 3 : i32, alignment = 4 : i64} : !llvm.array<2 x f32>
  // CHECK-LABEL: kgen.func @global_load
  kgen.func @global_load() -> !pop.scalar<f32> {
    // CHECK-NEXT: %0 = llvm.mlir.addressof @global_load_global_alloc : !llvm.ptr<3>
    // CHECK-NEXT: %1 = llvm.bitcast %0 : !llvm.ptr<3> to !llvm.ptr<3>
    // CHECK-NEXT: %2 = builtin.unrealized_conversion_cast %1 : !llvm.ptr<3> to !kgen.pointer<scalar<f32>, 3>
    // CHECK-NEXT: %3 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr
    %0 = pop.global_alloc 2 x !pop.scalar<f32> address_space 3 align 4
    %1 = pop.load %0 :!kgen.pointer<scalar<f32>, 3>
    kgen.return %1 : !pop.scalar<f32>
  }
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // CHECK-LABEL: llvm.mlir.global internal @global_store_global_alloc() {addr_space = 3 : i32, alignment = 4 : i64} : !llvm.array<2 x f32>
  // CHECK-LABEL: kgen.func @global_store
  kgen.func @global_store(%arg0: !pop.scalar<f32>) {
    // CHECK-NEXT: %0 = builtin.unrealized_conversion_cast %arg0 : !pop.scalar<f32> to f32
    // CHECK-NEXT: %1 = llvm.mlir.addressof @global_store_global_alloc : !llvm.ptr<3>
    // CHECK-NEXT: %2 = llvm.bitcast %1 : !llvm.ptr<3> to !llvm.ptr<3>
    // CHECK-NEXT: %3 = builtin.unrealized_conversion_cast %2 : !llvm.ptr<3> to !kgen.pointer<scalar<f32>, 3>
    // CHECK-NEXT: llvm.store %0, %2 {alignment = 4 : i64} : f32, !llvm.ptr
    %0 = pop.global_alloc 2 x !pop.scalar<f32> address_space 3 align 4
    pop.store %arg0, %0 :!kgen.pointer<scalar<f32>, 3>
    kgen.return
  }
}
