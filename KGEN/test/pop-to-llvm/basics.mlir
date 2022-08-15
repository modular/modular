// RUN: kgen-opt -split-input-file -convert-pop-to-llvm %s | FileCheck %s

// CHECK-LABEL: @constant
kgen.kernel @constant() -> !meta.scalar<f32> {
  // CHECK: llvm.mlir.constant(1.{{0+}}e+00 : f32) : f32
  %cst0 = pop.constant(1.0 : f32) : !meta.scalar<f32>
  kgen.return %cst0 : !meta.scalar<f32>
}

// -----

// CHECK-LABEL: @constant
kgen.kernel @constant() -> !meta.scalar<si64> {
  // CHECK: llvm.mlir.constant(1 : i64) : i64
  %cst0 = pop.constant(1 : i64) : !meta.scalar<si64>
  kgen.return %cst0 : !meta.scalar<si64>
}

// -----

// CHECK-LABEL: @abs
kgen.kernel @abs(%arg0: !meta.scalar<f32>) -> !meta.scalar<f32> {
  // CHECK: llvm.intr.fabs
  %0 = pop.abs %arg0 : !meta.scalar<f32>
  kgen.return %0 : !meta.scalar<f32>
}

// -----

// CHECK-LABEL: @abs
kgen.kernel @abs(%arg0: !meta.scalar<si32>) -> !meta.scalar<si32> {
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(false
  // CHECK: "llvm.intr.abs"(%{{.*}}, %[[ZERO]]
  %0 = pop.abs %arg0 : !meta.scalar<si32>
  kgen.return %0 : !meta.scalar<si32>
}

// -----

// CHECK-LABEL: @neg
kgen.kernel @neg(%arg0: !meta.scalar<f32>) -> !meta.scalar<f32> {
  // CHECK: llvm.fneg
  %0 = pop.neg %arg0 : !meta.scalar<f32>
  kgen.return %0 : !meta.scalar<f32>
}

// -----

// CHECK-LABEL: @neg
// CHECK-SAME: %[[ARG0:.*]]:
kgen.kernel @neg(%arg0: !meta.scalar<si32>) -> !meta.scalar<si32> {
  // CHECK: %[[RHS:.*]] = builtin.unrealized_conversion_cast %[[ARG0]]
  // CHECK: %[[LHS:.*]] = llvm.mlir.constant(0 :
  // CHECK: llvm.sub %[[LHS]], %[[RHS]]
  %0 = pop.neg %arg0 : !meta.scalar<si32>
  kgen.return %0 : !meta.scalar<si32>
}

// -----

// CHECK-LABEL: @add
kgen.kernel @add(%arg0: !meta.scalar<si32>) -> !meta.scalar<si32> {
  // CHECK: llvm.add
  %0 = pop.add %arg0, %arg0 : !meta.scalar<si32>
  kgen.return %0 : !meta.scalar<si32>
}

// -----

// CHECK-LABEL: @add
kgen.kernel @add(%arg0: !meta.scalar<f32>) -> !meta.scalar<f32> {
  // CHECK: llvm.fadd
  %0 = pop.add %arg0, %arg0 : !meta.scalar<f32>
  kgen.return %0 : !meta.scalar<f32>
}

// -----

// CHECK-LABEL: @sub
kgen.kernel @sub(%arg0: !meta.scalar<si32>, %arg1: !meta.scalar<si32>) -> !meta.scalar<si32> {
  // CHECK: llvm.sub
  %0 = pop.sub %arg0, %arg1 : !meta.scalar<si32>
  kgen.return %0 : !meta.scalar<si32>
}

// -----

// CHECK-LABEL: @sub
kgen.kernel @sub(%arg0: !meta.scalar<f32>, %arg1: !meta.scalar<f32>) -> !meta.scalar<f32> {
  // CHECK: llvm.fsub
  %0 = pop.sub %arg0, %arg1 : !meta.scalar<f32>
  kgen.return %0 : !meta.scalar<f32>
}

// -----

// CHECK-LABEL: @mul
kgen.kernel @mul(%arg0: !meta.scalar<si32>) -> !meta.scalar<si32> {
  // CHECK: llvm.mul
  %0 = pop.mul %arg0, %arg0 : !meta.scalar<si32>
  kgen.return %0 : !meta.scalar<si32>
}

// -----

// CHECK-LABEL: @mul
kgen.kernel @mul(%arg0: !meta.scalar<f32>) -> !meta.scalar<f32> {
  // CHECK: llvm.fmul
  %0 = pop.mul %arg0, %arg0 : !meta.scalar<f32>
  kgen.return %0 : !meta.scalar<f32>
}

// -----

// CHECK-LABEL: @copysign
kgen.kernel @copysign(%arg0: !meta.scalar<f32>, %arg1: !meta.scalar<f32>) -> !meta.scalar<f32> {
  // CHECK: llvm.intr.copysign
  %0 = pop.copysign %arg0, %arg1 : !meta.scalar<f32>
  kgen.return %0 : !meta.scalar<f32>
}

// -----

// CHECK-LABEL: @fma
kgen.kernel @fma(%arg0: !meta.scalar<f32>) -> !meta.scalar<f32> {
  // CHECK: llvm.intr.fma
  %0 = pop.fma %arg0, %arg0, %arg0 : !meta.scalar<f32>
  kgen.return %0 : !meta.scalar<f32>
}

// -----

// CHECK-LABEL: @fma
// CHECK-SAME: %[[ARG0:[a-z0-9]*]]:
// CHECK-SAME: %[[ARG1:[a-z0-9]*]]:
kgen.kernel @fma(%arg0: !meta.scalar<si32>, %arg1: !meta.scalar<si32>) -> !meta.scalar<si32> {
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
kgen.kernel @select(%arg0 : !meta.scalar<bool>, %arg1: !meta.scalar<f32>, %arg2: !meta.scalar<f32>) -> !meta.scalar<f32> {
  // CHECK: llvm.select
  %0 = pop.select %arg0, %arg1, %arg2 : !meta.scalar<f32>
  kgen.return %0 : !meta.scalar<f32>
}
