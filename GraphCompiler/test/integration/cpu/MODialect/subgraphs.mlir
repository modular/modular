// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s

mo.graph @add(%arg0: !mo.tensor<[5], f32>, %arg1: !mo.tensor<[5], f32>) -> (!mo.tensor<[5], f32>) isSubgraph {
  %add = mo.add(%arg0, %arg1) : !mo.tensor<[5], f32>
  mo.output %add : !mo.tensor<[5], f32>
}

mo.graph @relu(%arg0: !mo.tensor<[5], f32>) -> (!mo.tensor<[5], f32>) isSubgraph {
  %relu = mo.relu(%arg0) : !mo.tensor<[5], f32>
  mo.output %relu : !mo.tensor<[5], f32>
}

mo.graph @combined() -> (!mo.tensor<[5], f32>) {
  %0 = mo.constant <#M.dense_array<-1.0, 2.0, -3.0, -4.0, 5.0> : tensor<5xf32>> : !mo.tensor<[5], f32>
  %1 = mo.constant <#M.dense_array<3.0, 1.0, -6.0, 2.0, 5.0> : tensor<5xf32>> : !mo.tensor<[5], f32>
  %add = mo.call @add(%0, %1) : (!mo.tensor<[5], f32>, !mo.tensor<[5], f32>) -> !mo.tensor<[5], f32>
  %relu = mo.call @relu(%add) : (!mo.tensor<[5], f32>) -> !mo.tensor<[5], f32>
  mo.output %relu : !mo.tensor<[5], f32>
}
// CHECK{LITERAL}: 'combined' returned tensor<5xf32> [2.000000e+00, 3.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+01]
