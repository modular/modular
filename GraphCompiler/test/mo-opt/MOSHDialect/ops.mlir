// RUN: mo-opt %s | mo-opt | FileCheck %s

// CHECK-LABEL: kgen.func @broadcast(
// CHECK-SAME: %[[SH0:.*]]: !mosh.ape, %[[SH1:.*]]: !mosh.ape) -> !mosh.ape {
kgen.func @broadcast(%sh0: !mosh.ape, %sh1: !mosh.ape) -> !mosh.ape {
  // CHECK: %[[RES:.*]] = mosh.broadcast(%[[SH0]], %[[SH1]])
  %0 = mosh.broadcast(%sh0, %sh1)
  hlcf.return %0 : !mosh.ape
}

// CHECK-LABEL: kgen.func @concat(
// CHECK-SAME: %[[ARG0:.*]]: !mosh.ape, %[[ARG1:.*]]: !mosh.ape) -> !mosh.ape {
kgen.func @concat(%sh0: !mosh.ape, %sh1: !mosh.ape) -> !mosh.ape {
  // CHECK: %[[VAL0:.*]] = mosh.concat(%[[ARG0]], %[[ARG1]])
  %0 = mosh.concat(%sh0, %sh1)
  hlcf.return %0 : !mosh.ape
}

// CHECK-LABEL: kgen.func @equal(
// CHECK-SAME: %[[ARG0:.*]]: !mosh.ape, %[[ARG1:.*]]: !mosh.ape) -> !kgen.scalar<bool> {
kgen.func @equal(%sh0: !mosh.ape, %sh1: !mosh.ape) -> !kgen.scalar<bool> {
  // CHECK: %[[VAL0:.*]] = mosh.eq(%[[ARG0]], %[[ARG1]])
  %0 = mosh.eq(%sh0, %sh1)
  hlcf.return %0 : !kgen.scalar<bool>
}

// CHECK-LABEL: kgen.func @get_dim(
// CHECK-SAME: %[[SH0:.*]]: !mosh.ape, %[[DIM:.*]]: !kgen.scalar<si64>) -> (!kgen.scalar<si64>, !kgen.scalar<si64>) {
kgen.func @get_dim(%sh0: !mosh.ape, %dim: !kgen.scalar<si64>) -> (!kgen.scalar<si64>, !kgen.scalar<si64>) {
  // CHECK: %[[VAL0:.*]] = mosh.get_dim(%[[SH0]])[%[[DIM]]]
  %0 = mosh.get_dim(%sh0)[%dim]
  hlcf.return %0, %0 : !kgen.scalar<si64>, !kgen.scalar<si64>
}

// CHECK-LABEL: kgen.func @get_rank(
// CHECK-SAME: %[[SH0:.*]]: !mosh.ape) -> !kgen.scalar<si64> {
kgen.func @get_rank(%sh0: !mosh.ape) -> !kgen.scalar<si64> {
  // CHECK: %[[VAL0:.*]] = mosh.get_rank(%[[SH0]])
  %0 = mosh.get_rank(%sh0)
  hlcf.return %0 : !kgen.scalar<si64>
}

// CHECK-LABEL: kgen.func @new_dim(
// CHECK-SAME: %[[D0:.*]]: !kgen.scalar<si64>, %[[D1:.*]]: !kgen.scalar<si64>) -> (!mosh.ape, !mosh.ape, !mosh.ape) {
kgen.func @new_dim(%d0: !kgen.scalar<si64>, %d1: !kgen.scalar<si64>)
    -> (!mosh.ape, !mosh.ape, !mosh.ape) {
  // CHECK: %[[SH0:.*]] = mosh.new()
  %sh0 = mosh.new()
  // CHECK: %[[SH1:.*]] = mosh.new(%[[D0]])
  %sh1 = mosh.new(%d0)
  // CHECK: %[[SH2:.*]] = mosh.new(%[[D0]], %[[D1]], %[[D0]])
  %sh2 = mosh.new(%d0, %d1, %d0)
  hlcf.return %sh0, %sh1, %sh2 : !mosh.ape, !mosh.ape, !mosh.ape
}

// CHECK-LABEL: kgen.func @num_elements(
// CHECK-SAME: %[[SH0:.*]]: !mosh.ape) -> !kgen.scalar<si64> {
kgen.func @num_elements(%sh0: !mosh.ape) -> !kgen.scalar<si64> {
  // CHECK: %[[VAL0:.*]] = mosh.num_elements(%[[SH0]])
  %0 = mosh.num_elements(%sh0)
  hlcf.return %0 : !kgen.scalar<si64>
}

// CHECK-LABEL: kgen.func @slice(
// CHECK-SAME: %[[SH0:.*]]: !mosh.ape, %[[START:.*]]: !kgen.scalar<si64>, %[[END:.*]]: !kgen.scalar<si64>) -> !mosh.ape {
kgen.func @slice(%sh0: !mosh.ape, %start: !kgen.scalar<si64>, %end: !kgen.scalar<si64>) -> !mosh.ape {
  // CHECK: %[[VAL0:.*]] = mosh.slice(%[[SH0]])[%[[START]], %[[END]]]
  %sh1 = mosh.slice(%sh0)[%start, %end]
  hlcf.return %sh1 : !mosh.ape
}

// CHECK-LABEL: kgen.generator @param_to_value<
// CHECK-SAME: [[D:.*]]>() -> (!kgen.scalar<si64>, !mosh.ape) {
kgen.generator @param_to_value<D>() -> (!kgen.scalar<si64>, !mosh.ape) {

  // CHECK-NEXT: %[[IDX:.*]]   = mosh.param.to_value = <[[D]]>
  // CHECK-NEXT: %[[SHAPE:.*]] = mosh.param.to_value: !mosh.ape = <[[[D]], 42]>
  %idx   = mosh.param.to_value = <D>
  %shape = mosh.param.to_value: !mosh.ape = <[D, 42]>

  // CHECK: hlcf.return %[[IDX]], %[[SHAPE]] : !kgen.scalar<si64>, !mosh.ape
  hlcf.return %idx, %shape : !kgen.scalar<si64>, !mosh.ape
}

// CHECK-LABEL: kgen.generator @value_to_param<
// CHECK-SAME: [[D:.*]]>(%[[ARG0:.*]]: !kgen.scalar<si64>, %[[ARG1:.*]]: f32, %[[ARG2:.*]]: !mosh.ape) {
kgen.generator @value_to_param<D>(%arg0: !kgen.scalar<si64>, %arg1: f32, %arg2: !mosh.ape) -> () {
  // CHECK: mosh.param.from_value  Q : !kgen.scalar<si64> = %[[ARG0]]
  mosh.param.from_value Q : !kgen.scalar<si64> = %arg0

  // CHECK: mosh.param.from_value  P : f32 = %[[ARG1]]
  mosh.param.from_value P : f32 = %arg1

  // CHECK: mosh.param.from_value  Sh : !mosh.ape = %[[ARG2]]
  mosh.param.from_value Sh : !mosh.ape = %arg2

  // CHECK: hlcf.return
  hlcf.return
}
