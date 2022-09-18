// RUN: kgen-opt -split-input-file -allow-unregistered-dialect -lower-global-pop-to-llvm %s | FileCheck %s

// CHECK-LABEL: @external_call
kgen.func @external_call(%a: !meta.scalar<ui32>) -> !meta.simd<4, f64> {
  // CHECK: llvm.call @foo
  %0 = pop.external_call @foo(%a) : (!meta.scalar<ui32>) -> !meta.simd<4, f64>
  kgen.return %0 : !meta.simd<4, f64>
}

// CHECK: llvm.func @foo(i32) -> vector<4xf64>

// -----

// CHECK-LABEL: @external_call_variadic
kgen.func @external_call_variadic(%a: !meta.scalar<ui32>) {
  // CHECK: llvm.call @foo
  pop.external_call @foo (%a) (!meta.scalar<ui32>) -> () : (!meta.scalar<ui32>) -> ()
  // CHECK: llvm.call @foo
  pop.external_call @foo (%a, %a) (!meta.scalar<ui32>) -> () : (!meta.scalar<ui32>, !meta.scalar<ui32>) -> ()
  kgen.return
}

// CHECK: llvm.func @foo(i32, ...)

// -----

// CHECK-LABEL: @global_constant
kgen.func @global_constant() {
  // CHECK: llvm.mlir.addressof @global_constant_0
  %0 = pop.global_constant(5 : ui32) : !meta.scalar<ui32>
  // CHECK: llvm.mlir.addressof @global_constant_0
  %1 = pop.global_constant(5 : ui32) : !meta.scalar<ui32>
  // CHECK: llvm.mlir.addressof @global_constant_1
  %2 = pop.global_constant(6 : ui32) : !meta.scalar<ui32>
  kgen.return
}

// -----

// CHECK-LABEL: @global_array_constant
kgen.func @global_array_constant() {
  // CHECK: llvm.mlir.addressof @global_constant
  %0 = pop.global_constant(dense<[0, 1, 2, 3]> : tensor<4xui32>) : !pop.array<4, !meta.scalar<ui32>>
  kgen.return
}

// CHECK: llvm.mlir.global internal constant @global_constant(dense<[0, 1, 2, 3]> : tensor<4xui32>) {{.*}} : !llvm.array<4 x i32>
