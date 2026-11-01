// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

mo.graph @broadcast_shape_dynamic_inputs(
  %lhs_shape: !mo.tensor<[2], si32>,
  %rhs_shape: !mo.tensor<[4], si64>
) -> (!mo.tensor<[4], si32>) {
  %out_shape = mo.broadcast_shape(%lhs_shape, %rhs_shape) : (
    !mo.tensor<[2], si32>, !mo.tensor<[4], si64>
  ) -> !mo.tensor<[4], si32>
  mo.output %out_shape : !mo.tensor<[4], si32>
}
// CHECK{LITERAL}: 'broadcast_shape_dynamic_inputs' returned tensor<4xsi32> [1, 1, 1, 1]
