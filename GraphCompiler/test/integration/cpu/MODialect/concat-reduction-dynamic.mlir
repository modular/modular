// RUN: mo-opt %s --mo-shape-inference --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --input-shapes=2x3 --input-shapes=2x5 -model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

// A concat with run-time extents fused into a sum over its axis: the select
// reads the extents at run time, and the launch limits the width to one
// dividing both (here 1).

// CHECK: --- Running 'sum_of_dynamic_concat':
// CHECK-NEXT: 'sum_of_dynamic_concat' returned tensor<2x1xsi32> [8, 8]
mo.graph @sum_of_dynamic_concat<D0, D1, D2>(%a: !mo.tensor<[D0, D1], si32>, %b: !mo.tensor<[D0, D2], si32>) -> !mo.tensor<[D0, 1], si32> {
  %c = mo.concat<() -> D3>[1](%a, %b) : (!mo.tensor<[D0, D1], si32>, !mo.tensor<[D0, D2], si32>) -> !mo.tensor<[D0, D3], si32>
  %r = mo.reduce.add(%c) : (!mo.tensor<[D0, D3], si32>) -> !mo.tensor<[D0, 1], si32>
  mo.output %r : !mo.tensor<[D0, 1], si32>
}
