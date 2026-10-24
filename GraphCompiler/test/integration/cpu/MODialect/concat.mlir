// RUN: mo-opt %s --mo-shape-inference --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

// RUN: mo-opt %s --mo-shape-inference --gc-pipeline="trivial-kernel-opt=false constant-fold=false" -o %t_no_opt.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t_no_opt.mlir | FileCheck %s

// CHECK: --- Running 'concat_static_output':
// CHECK-NEXT: 'concat_static_output' returned tensor<6xsi32> [2, 1, 1, 3, 3, 3]
mo.graph @concat_static_output(%arg2: !mo.tensor<[2], si32>) -> (!mo.tensor<[6], si32>) {
  %ax = mo.constant <#M.dense_array<0> : tensor<si32>> : !mo.tensor<[], si32>
  %arg1 = mo.constant <#M.dense_array<2> : tensor<1xsi32>> : !mo.tensor<[1], si32>
  %arg3 = mo.constant <#M.dense_array<3, 3, 3> : tensor<3xsi32>> : !mo.tensor<[3], si32>
  %out = mo.concat[0](%arg1, %arg2, %arg3) : (
    !mo.tensor<[1], si32>, !mo.tensor<[2], si32>, !mo.tensor<[3], si32>
  ) -> !mo.tensor<[6], si32>
  mo.output %out : !mo.tensor<[6], si32>
}

// CHECK: --- Running 'concat_dyn_output':
// CHECK-NEXT: 'concat_dyn_output' returned tensor<6xsi32> [2, 1, 1, 3, 3, 3]
mo.graph @concat_dyn_output<() -> D0>(%arg2: !mo.tensor<[2], si32>) -> !mo.tensor<[D0], si32> {
  %ax = mo.constant <#M.dense_array<0> : tensor<si32>> : !mo.tensor<[], si32>
  %arg1 = mo.constant <#M.dense_array<2> : tensor<1xsi32>> : !mo.tensor<[1], si32>
  %arg3 = mo.constant <#M.dense_array<3, 3, 3> : tensor<3xsi32>> : !mo.tensor<[3], si32>
  %out = mo.concat<() -> D1>[0](%arg1, %arg2, %arg3) : (
    !mo.tensor<[1], si32>, !mo.tensor<[2], si32>, !mo.tensor<[3], si32>
  ) -> !mo.tensor<[D1], si32>
  mo.output<D1> %out : !mo.tensor<[D1], si32>
}

// CHECK: --- Running 'concat_negative_axis':
// CHECK-NEXT: 'concat_negative_axis' returned tensor<6xsi32> [2, 1, 1, 3, 3, 3]
mo.graph @concat_negative_axis(%arg2: !mo.tensor<[2], si32>) -> (!mo.tensor<[6], si32>) {
  %ax = mo.constant <#M.dense_array<-1> : tensor<si32>> : !mo.tensor<[], si32>
  %arg1 = mo.constant <#M.dense_array<2> : tensor<1xsi32>> : !mo.tensor<[1], si32>
  %arg3 = mo.constant <#M.dense_array<3, 3, 3> : tensor<3xsi32>> : !mo.tensor<[3], si32>
  %out = mo.concat[-1](%arg1, %arg2, %arg3) : (
    !mo.tensor<[1], si32>, !mo.tensor<[2], si32>, !mo.tensor<[3], si32>
  ) -> !mo.tensor<[6], si32>
  mo.output %out : !mo.tensor<[6], si32>
}

// CHECK-LABEL: --- Running 'concat_fallback_shape_func':
// CHECK-NEXT: 'concat_fallback_shape_func' returned tensor<10x50xsi32>
mo.graph @concat_fallback_shape_func<() -> D0, D1>(
  %input1 : !mo.tensor<[10, 20], f32>,
  %input2 : !mo.tensor<[10, 30], f32>
) -> !mo.tensor<[D0, D1], si32> {

  mo.invoke_shape_func["mo.concat"]<() -> D2, D3>(%input1, %input2)
    [kgenParams = {input_type = f32, input_rank = 2, axis = 1}, dataDeptTensors = [0 : i32]]
    : (!mo.tensor<[10, 20], f32>, !mo.tensor<[10, 30], f32>)

  %one = mo.constant <#M.dense_array<1> : tensor<si32>> : !mo.tensor<[], si32>
  %val = mo.static.broadcast_to(%one): !mo.tensor<[], si32> -> !mo.tensor<[D2, D3], si32>

  mo.output<D2, D3> %val : !mo.tensor<[D2, D3], si32>
}
