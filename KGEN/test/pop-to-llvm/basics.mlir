// RUN: kgen-opt -split-input-file -convert-pop-to-llvm %s | FileCheck %s

// CHECK-LABEL: @constant
kgen.func @constant() -> !meta.scalar<f32> {
  // CHECK: llvm.mlir.constant(1.{{0+}}e+00 : f32) : f32
  %cst0 = pop.constant(1.0 : f32) : !meta.scalar<f32>
  kgen.return %cst0 : !meta.scalar<f32>
}

// -----

// CHECK-LABEL: @constant
kgen.func @constant() -> !meta.scalar<si64> {
  // CHECK: llvm.mlir.constant(1 : si64) : i64
  %cst0 = pop.constant(1 : si64) : !meta.scalar<si64>
  kgen.return %cst0 : !meta.scalar<si64>
}

// -----

// CHECK-LABEL: @abs
kgen.func @abs(%arg0: !meta.scalar<f32>) -> !meta.scalar<f32> {
  // CHECK: llvm.intr.fabs
  %0 = pop.abs %arg0 : !meta.scalar<f32>
  kgen.return %0 : !meta.scalar<f32>
}

// -----

// CHECK-LABEL: @abs
kgen.func @abs(%arg0: !meta.scalar<si32>) -> !meta.scalar<si32> {
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(false
  // CHECK: "llvm.intr.abs"(%{{.*}}, %[[ZERO]]
  %0 = pop.abs %arg0 : !meta.scalar<si32>
  kgen.return %0 : !meta.scalar<si32>
}

// -----

// CHECK-LABEL: @neg
kgen.func @neg(%arg0: !meta.scalar<f32>) -> !meta.scalar<f32> {
  // CHECK: llvm.fneg
  %0 = pop.neg %arg0 : !meta.scalar<f32>
  kgen.return %0 : !meta.scalar<f32>
}

// -----

// CHECK-LABEL: @neg
// CHECK-SAME: %[[ARG0:.*]]:
kgen.func @neg(%arg0: !meta.scalar<si32>) -> !meta.scalar<si32> {
  // CHECK: %[[RHS:.*]] = builtin.unrealized_conversion_cast %[[ARG0]]
  // CHECK: %[[LHS:.*]] = llvm.mlir.constant(0 :
  // CHECK: llvm.sub %[[LHS]], %[[RHS]]
  %0 = pop.neg %arg0 : !meta.scalar<si32>
  kgen.return %0 : !meta.scalar<si32>
}

// -----

// CHECK-LABEL: @add
kgen.func @add(%arg0: !meta.scalar<si32>) -> !meta.scalar<si32> {
  // CHECK: llvm.add
  %0 = pop.add %arg0, %arg0 : !meta.scalar<si32>
  kgen.return %0 : !meta.scalar<si32>
}

// -----

// CHECK-LABEL: @add
kgen.func @add(%arg0: !meta.scalar<f32>) -> !meta.scalar<f32> {
  // CHECK: llvm.fadd
  %0 = pop.add %arg0, %arg0 : !meta.scalar<f32>
  kgen.return %0 : !meta.scalar<f32>
}

// -----

// CHECK-LABEL: @sub
kgen.func @sub(%arg0: !meta.scalar<si32>, %arg1: !meta.scalar<si32>) -> !meta.scalar<si32> {
  // CHECK: llvm.sub
  %0 = pop.sub %arg0, %arg1 : !meta.scalar<si32>
  kgen.return %0 : !meta.scalar<si32>
}

// -----

// CHECK-LABEL: @sub
kgen.func @sub(%arg0: !meta.scalar<f32>, %arg1: !meta.scalar<f32>) -> !meta.scalar<f32> {
  // CHECK: llvm.fsub
  %0 = pop.sub %arg0, %arg1 : !meta.scalar<f32>
  kgen.return %0 : !meta.scalar<f32>
}

// -----

// CHECK-LABEL: @mul
kgen.func @mul(%arg0: !meta.scalar<si32>) -> !meta.scalar<si32> {
  // CHECK: llvm.mul
  %0 = pop.mul %arg0, %arg0 : !meta.scalar<si32>
  kgen.return %0 : !meta.scalar<si32>
}

// -----

// CHECK-LABEL: @mul
kgen.func @mul(%arg0: !meta.scalar<f32>) -> !meta.scalar<f32> {
  // CHECK: llvm.fmul
  %0 = pop.mul %arg0, %arg0 : !meta.scalar<f32>
  kgen.return %0 : !meta.scalar<f32>
}

// -----

// CHECK-LABEL: @div
kgen.func @div(%arg0: !meta.scalar<si32>,
                 %arg1: !meta.scalar<ui32>,
                 %arg2: !meta.scalar<f32>) -> (
                  !meta.scalar<si32>,
                  !meta.scalar<ui32>,
                  !meta.scalar<f32>) {
  // CHECK: llvm.sdiv
  %0 = pop.div %arg0, %arg0 : !meta.scalar<si32>
  // CHECK: llvm.udiv
  %1 = pop.div %arg1, %arg1 : !meta.scalar<ui32>
  // CHECK: llvm.fdiv
  %2 = pop.div %arg2, %arg2 : !meta.scalar<f32>
  kgen.return %0, %1, %2 : !meta.scalar<si32>,!meta.scalar<ui32>,!meta.scalar<f32>
}

// -----

// CHECK-LABEL: @copysign
kgen.func @copysign(%arg0: !meta.scalar<f32>, %arg1: !meta.scalar<f32>) -> !meta.scalar<f32> {
  // CHECK: llvm.intr.copysign
  %0 = pop.copysign %arg0, %arg1 : !meta.scalar<f32>
  kgen.return %0 : !meta.scalar<f32>
}

// -----

// CHECK-LABEL: @fma
kgen.func @fma(%arg0: !meta.scalar<f32>) -> !meta.scalar<f32> {
  // CHECK: llvm.intr.fma
  %0 = pop.fma %arg0, %arg0, %arg0 : !meta.scalar<f32>
  kgen.return %0 : !meta.scalar<f32>
}

// -----

// CHECK-LABEL: @fma
// CHECK-SAME: %[[ARG0:[a-z0-9]*]]:
// CHECK-SAME: %[[ARG1:[a-z0-9]*]]:
kgen.func @fma(%arg0: !meta.scalar<si32>, %arg1: !meta.scalar<si32>) -> !meta.scalar<si32> {
  // CHECK: %[[LHS:.*]] = builtin.unrealized_conversion_cast %[[ARG0]]
  // CHECK: %[[RHS:.*]] = builtin.unrealized_conversion_cast %[[ARG1]]
  // CHECK: %[[MUL:.*]] = llvm.mul %[[LHS]], %[[LHS]]
  // CHECK: %[[FMA:.*]] = llvm.add %[[MUL]], %[[RHS]]
  // CHECK: builtin.unrealized_conversion_cast %[[FMA]]
  %0 = pop.fma %arg0, %arg0, %arg1 : !meta.scalar<si32>
  kgen.return %0 : !meta.scalar<si32>
}

// -----

// CHECK-LABEL: @select
kgen.func @select(%arg0: !meta.scalar<bool>, %arg1: !meta.scalar<f32>, %arg2: !meta.scalar<f32>) -> !meta.scalar<f32> {
  // CHECK: llvm.select
  %0 = pop.select %arg0, %arg1, %arg2 : !meta.scalar<f32>
  kgen.return %0 : !meta.scalar<f32>
}

// -----

// CHECK-LABEL: @load
kgen.func @load(%p: !meta.pointer<f32>) -> !meta.scalar<f32> {
  // CHECK: llvm.load
  %0 = pop.load %p : !meta.pointer<f32>
  kgen.return %0 : !meta.scalar<f32>
}

// -----

// CHECK-LABEL: @store
kgen.func @store(%p: !meta.pointer<si32>, %v: !meta.scalar<si32>) {
  // CHECK: llvm.store
  pop.store %v, %p : !meta.pointer<si32>
  kgen.return
}

// -----

// CHECK-LABEL: @offset
kgen.func @offset(%p: !meta.pointer<f32>, %i: index) -> !meta.pointer<f32> {
  // CHECK: llvm.getelementptr %{{.*}}[{{.*}}]
  %0 = pop.offset %p[%i] : !meta.pointer<f32>
  kgen.return %0 : !meta.pointer<f32>
}

// -----

// CHECK-LABEL: @shifts
kgen.func @shifts(%arg0: !meta.scalar<si32>, %arg1: !meta.scalar<si32>, %arg2: !meta.scalar<ui32>, %arg3: !meta.scalar<ui32>) {
  // CHECK: llvm.shl
  %0 = pop.shl %arg0, %arg1 : !meta.scalar<si32>
  // CHECK: llvm.ashr
  %1 = pop.shr %arg0, %arg1 : !meta.scalar<si32>
  // CHECKL llvm.lshr
  %2 = pop.shr %arg2, %arg3 : !meta.scalar<ui32>
  kgen.return
}

// -----

// CHECK-LABEL: @simd_shift
kgen.func @simd_shift(%arg0: !meta.simd<4, si32>, %arg1: !meta.simd<4, si32>, %arg2: !meta.simd<4, ui32>, %arg3: !meta.simd<4, ui32>) {
  // CHECK: llvm.shl
  %0 = pop.shl %arg0, %arg1 : !meta.simd<4, si32>
  // CHECK: llvm.ashr
  %1 = pop.shr %arg0, %arg1 : !meta.simd<4, si32>
  // CHECKL llvm.lshr
  %2 = pop.shr %arg2, %arg3 : !meta.simd<4, ui32>
  kgen.return
}
