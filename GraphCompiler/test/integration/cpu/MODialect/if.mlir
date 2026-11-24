// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s

// CHECK: --- Running 'if_op_scalar_tensor':
// CHECK-NEXT: 'if_op_scalar_tensor' returned tensor<2xf32> [4.000000e+00, 6.000000e+00]
mo.graph @if_op_scalar_tensor() -> (!mo.tensor<[2], f32>) {
  %cond = mo.constant <#M.dense_array<1> : tensor<1xi1>> : !mo.tensor<[], bool>
  %val1 = mo.constant <#M.dense_array<1.0, 2.0> : tensor<2xf32>> : !mo.tensor<[2], f32>
  %val2 = mo.constant <#M.dense_array<3.0, 4.0> : tensor<2xf32>> : !mo.tensor<[2], f32>
  %0 = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.tensor<[2], f32> {
    %1 = mo.add(%val1, %val2) : !mo.tensor<[2], f32>
    mo.yield %1 : !mo.tensor<[2], f32>
  } else {
    %2= mo.mul(%val1, %val2) : !mo.tensor<[2], f32>
    mo.yield %2 : !mo.tensor<[2], f32>
  }
  mo.output %0 : !mo.tensor<[2], f32>
}

// CHECK: --- Running 'if_op_scalar_tensor_false':
// CHECK-NEXT: 'if_op_scalar_tensor_false' returned tensor<2xf32> [3.000000e+00, 8.000000e+00]
mo.graph @if_op_scalar_tensor_false() -> (!mo.tensor<[2], f32>) {
  %cond = mo.constant <#M.dense_array<0> : tensor<1xi1>> : !mo.tensor<[], bool>
  %val1 = mo.constant <#M.dense_array<1.0, 2.0> : tensor<2xf32>> : !mo.tensor<[2], f32>
  %val2 = mo.constant <#M.dense_array<3.0, 4.0> : tensor<2xf32>> : !mo.tensor<[2], f32>
  %0 = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.tensor<[2], f32> {
    %1 = mo.add(%val1, %val2) : !mo.tensor<[2], f32>
    mo.yield %1 : !mo.tensor<[2], f32>
  } else {
    %2= mo.mul(%val1, %val2) : !mo.tensor<[2], f32>
    mo.yield %2 : !mo.tensor<[2], f32>
  }
  mo.output %0 : !mo.tensor<[2], f32>
}

// CHECK: --- Running 'if_op_scalar':
// CHECK-NEXT: 'if_op_scalar' returned tensor<2xf32> [4.000000e+00, 6.000000e+00]
mo.graph @if_op_scalar() -> (!mo.tensor<[2], f32>) {
  %cond = mo.constant.scalar <1 : i1> : !mo.scalar<bool>
  %val1 = mo.constant <#M.dense_array<1.0, 2.0> : tensor<2xf32>> : !mo.tensor<[2], f32>
  %val2 = mo.constant <#M.dense_array<3.0, 4.0> : tensor<2xf32>> : !mo.tensor<[2], f32>
  %0 = mo.if(%cond : !mo.scalar<bool>) -> !mo.tensor<[2], f32> {
    %1 = mo.add(%val1, %val2) : !mo.tensor<[2], f32>
    mo.yield %1 : !mo.tensor<[2], f32>
  } else {
    %2= mo.mul(%val1, %val2) : !mo.tensor<[2], f32>
    mo.yield %2 : !mo.tensor<[2], f32>
  }
  mo.output %0 : !mo.tensor<[2], f32>
}

// CHECK: --- Running 'if_op_scalar_false':
// CHECK-NEXT: 'if_op_scalar_false' returned tensor<2xf32> [3.000000e+00, 8.000000e+00]
mo.graph @if_op_scalar_false() -> (!mo.tensor<[2], f32>) {
  %cond = mo.constant.scalar <0 : i1> : !mo.scalar<bool>
  %val1 = mo.constant <#M.dense_array<1.0, 2.0> : tensor<2xf32>> : !mo.tensor<[2], f32>
  %val2 = mo.constant <#M.dense_array<3.0, 4.0> : tensor<2xf32>> : !mo.tensor<[2], f32>
  %0 = mo.if(%cond : !mo.scalar<bool>) -> !mo.tensor<[2], f32> {
    %1 = mo.add(%val1, %val2) : !mo.tensor<[2], f32>
    mo.yield %1 : !mo.tensor<[2], f32>
  } else {
    %2= mo.mul(%val1, %val2) : !mo.tensor<[2], f32>
    mo.yield %2 : !mo.tensor<[2], f32>
  }
  mo.output %0 : !mo.tensor<[2], f32>
}
