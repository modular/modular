// RUN: mo-opt %s --mo-pattern-fusion | FileCheck %s


// Negative: the device is not SM100
// CHECK-LABEL: mo.graph @neg_matmul_add_no_sm100
// CHECK: mo.matmul
// CHECK-NOT: mo.composite.matmul_add
// CHECK: mo.add
mo.graph @neg_matmul_add_no_sm100(%arg0: !mo.tensor<[65536, 1536], bf16, gpu:0>, %arg1: !mo.tensor<[1536, 1536], bf16, gpu:0, {layout = #mo.layout<KN>}>, %arg2: !mo.tensor<[65536, 1536], bf16, gpu:0>) -> !mo.tensor<[65536, 1536], bf16, gpu:0> {
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[65536, 1536], bf16, gpu:0>, !mo.tensor<[1536, 1536], bf16, gpu:0, {layout = #mo.layout<KN>}>) -> !mo.tensor<[65536, 1536], bf16, gpu:0>
  %1 = mo.add(%0, %arg2) : !mo.tensor<[65536, 1536], bf16, gpu:0>
  mo.output %1 : !mo.tensor<[65536, 1536], bf16, gpu:0>
}
