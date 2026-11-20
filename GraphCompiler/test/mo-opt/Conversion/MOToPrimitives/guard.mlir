// RUN: mo-opt --mo-shape-inference --mo-infer-layouts --mogg-import-packages --mo-to-map --map-outline-structured-kernel-bodies --map-to-mogg --mogg-to-mgp --mo-topological-sort %s | FileCheck %s

// using the rather laborious specification of passes above because
// predicate-asserts-deps pass doesn't currently handle mo.guard ops and I don't
// want any interactions with that pass in this test

// CHECK-LABEL: mgp.model @add_guard
// CHECK-DAG  : [[CHAIN:%.*]] = mgp.assert
// CHECK-DAG  : [[RES:%.*]] = mgp.guard [[CHAIN]], [[ARG0:%.*]], [[ARG1:%.*]]
// CHECK-DAG  : [[BUF0:%.*]] = mgp.tensor.extract.buffer{{.*}},
// CHECK-SAME : [[RES]]#0
// CHECK-DAG  : [[BUF1:%.*]] = mgp.tensor.extract.buffer{{.*}},
// CHECK-SAME : [[RES]]#1
// CHECK-SAME : in [[BUF0]]{{.*}}in [[BUF1]]
mo.graph @add_guard<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>, %arg1: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
  kgen.param.declare B: scalar<bool> = <identical(:scalar<si64> D0, D1)>
  kgen.param.declare CH0: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "not square matrix" : !kgen.string}, B)>

  %chain = mosh.param.to_value: !mo.chain = <CH0>

  %a0, %a1 = mo.guard[%chain](%arg0, %arg1) : !mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], f32>
  %1 = mo.add(%a0, %a1) : !mo.tensor<[D0, D1], f32>

  mo.output %1 : !mo.tensor<[D0, D1], f32>
}
