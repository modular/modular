// RUN: kgen-opt -split-input-file -allow-unregistered-dialect -lower-global-pop-to-llvm %s | FileCheck %s

module attributes {M.target_info = #M.target<triple="", cpu="", features="", pointer_bit_width=64, simd_bit_width=128>} {
  // CHECK-LABEL: @external_call
  kgen.func @external_call(%a: !pop.simd<1, ui32>) -> !pop.simd<4, f64> {
    // CHECK: llvm.call @foo
    %0 = pop.external_call @foo(%a) : (!pop.simd<1, ui32>) -> !pop.simd<4, f64>
    kgen.return %0 : !pop.simd<4, f64>
  }
  // CHECK: llvm.func @foo(i32) -> vector<4xf64>
}


// -----

module attributes {M.target_info = #M.target<triple="", cpu="", features="", pointer_bit_width=64, simd_bit_width=128>} {
  // CHECK-LABEL: @external_call_variadic
  kgen.func @external_call_variadic(%a: !pop.simd<1, ui32>) {
    // CHECK: llvm.call @foo
    pop.external_call @foo (%a) (!pop.simd<1, ui32>) -> () : (!pop.simd<1, ui32>) -> ()
    // CHECK: llvm.call @foo
    pop.external_call @foo (%a, %a) (!pop.simd<1, ui32>) -> () : (!pop.simd<1, ui32>, !pop.simd<1, ui32>) -> ()
    kgen.return
  }
  // CHECK: llvm.func @foo(i32, ...)
}

// -----

module attributes {M.target_info = #M.target<triple="", cpu="", features="", pointer_bit_width=64, simd_bit_width=128>} {
  // CHECK-LABEL: @global_constant
  kgen.func @global_constant() {
    // CHECK: llvm.mlir.addressof @global_constant_0
    %0 = pop.global_constant: ui32 = <5>
    // CHECK: llvm.mlir.addressof @global_constant_0
    %1 = pop.global_constant: ui32 = <5>
    // CHECK: llvm.mlir.addressof @global_constant_1
    %2 = pop.global_constant: simd<2, si32> = <<2, 5>>
    kgen.return
  }
}

// -----

module attributes {M.target_info = #M.target<triple="", cpu="", features="", pointer_bit_width=64, simd_bit_width=128>} {
  // CHECK-LABEL: @global_array_constant
  kgen.func @global_array_constant() {
    // CHECK: llvm.mlir.addressof @global_constant
    %0 = pop.global_constant: array<4, ui32> = <[1, 2, 3, 4]>
    kgen.return
  }
  // CHECK: llvm.mlir.global internal constant @global_constant() {
  // CHECK: %0 = llvm.mlir.undef : !llvm.array<4 x i32>
  // CHECK: llvm.return %{{.*}} : !llvm.array<4 x i32>
}
