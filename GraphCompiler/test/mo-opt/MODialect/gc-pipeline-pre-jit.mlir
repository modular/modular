// RUN: mo-opt %s --gc-pipeline-pre-jit=constant-fold=false -o %t.mlir
// RUN: FileCheck %s < %t.mlir

// CHECK: mgp.model
mo.graph @conv_transpose_default_hyperparameters() -> (!mo.tensor<[1, 3, 3, 1], si64>) {
  %input = mo.constant <#M.dense_array<0, 1, 2, 3> : tensor<4xsi64>> : !mo.tensor<[1, 2, 2, 1], si64>
  %filter = mo.constant <#M.dense_array<0, 1, 2, 3> : tensor<4xsi64>> : !mo.tensor<[2, 2, 1, 1], si64>
  %st = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %di = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %pa = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %op = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %output = mo.conv_transpose(%input, %filter)
    [strides = %st, dilations = %di, paddings = %pa, output_paddings = %op, filter_layout = ""] : (
    !mo.tensor<[1, 2, 2, 1], si64>, !mo.tensor<[2, 2, 1, 1], si64>,
    !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[1, 3, 3, 1], si64>
  mo.output %output : !mo.tensor<[1, 3, 3, 1], si64>
}
