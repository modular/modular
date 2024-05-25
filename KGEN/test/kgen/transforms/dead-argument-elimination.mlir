// RUN: kgen-opt -dead-argument-elimination -split-input-file %s | FileCheck %s

// COM: simple local dead argument.
// CHECK-LABEL: kgen.func @simple(%arg0: index) -> index
kgen.func @simple(%dead_arg: !pop.scalar<f32> , %live_arg: index) -> index {
  kgen.return %live_arg: index
}

// CHECK-LABEL: kgen.func export @simple_ret_use(%arg0: index) -> index
kgen.func export @simple_ret_use(%arg: index) -> index {
  // CHECK:  %0 = kgen.call @simple(%arg0) : (index) -> index
  %0 = kgen.call @simple(%arg, %arg) : (index, index) -> index
  kgen.return %0: index
}

// -----

// COM: simple dead argument due to callee's argument being dead.
// CHECK-LABEL: kgen.func @f_dead_arg(%arg0: index) -> index
kgen.func @f_dead_arg(%dead_arg: !pop.scalar<f32>, %live_arg: index) -> index {
  // CHEKC: [[V0:%.*]] = kgen.call @g_dead_arg(): () -> index
  %0 = kgen.call @g_dead_arg(%dead_arg): (!pop.scalar<f32>) -> index
  %1 = index.add %0, %live_arg
  kgen.return %1: index
}

// CHECK-LABEL: kgen.func @g_dead_arg() -> index {
kgen.func @g_dead_arg(%dead_arg: !pop.scalar<f32>) -> index {
  %0 = index.constant 0
  kgen.return %0: index
}

// CHECK-LABEL: kgen.func export @h_dead_arg(%arg0: index) -> index {
kgen.func export @h_dead_arg(%arg: index) -> index {
  // CHECK: [[V0:%.*]] = kgen.call @g_dead_arg() : () -> index
  %0 = kgen.call @g_dead_arg(%arg) : (index) -> index
  // CHECK: [[V1:%.*]] = kgen.call @f_dead_arg(%arg0) : (index) -> index
  %1 = kgen.call @f_dead_arg(%arg, %arg) : (index, index) -> index
  %2 = index.add %0, %1
  kgen.return %2: index
}

// -----

// COM: f has dead_arg that reaches return/call through operations.
// CHECK-LABEL: kgen.func @f_dead_arg1() -> index {
kgen.func @f_dead_arg1(%dead_arg: index) -> index {
  %0 = index.constant 0
  %1 = index.add %dead_arg, %0
  // CHECK: [[V0:%.*]] = kgen.call @g_dead_arg1() : () -> index
  %2 = kgen.call @g_dead_arg1(%1): (index) -> index
  kgen.return %2: index
}

// CHECK-LABEL: kgen.func @g_dead_arg1() -> index {
kgen.func @g_dead_arg1(%dead_arg: index) -> index {
  %0 = index.constant 0
  kgen.return %0: index
}

// CHECK-LABEL: kgen.func export @h_dead1(%arg0: index) -> index {
kgen.func export @h_dead1(%arg: index) -> index {
  // CHECK: [[V0:%.*]] = kgen.call @g_dead_arg1() : () -> index
  %0 = kgen.call @g_dead_arg1(%arg) : (index) -> index
  kgen.return %0: index
}
