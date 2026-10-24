// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

// Test case from test_limit_outputs from test_non_max_suppression.mojo.
mo.graph @non_maximum_suppression<() -> D0, D1>(%one : !mo.tensor<[], si64>) -> !mo.tensor<[D0, D1], si64> {
    %boxes = mo.constant <#M.dense_array<0.0,0.0,1.0,1.0,0.0,0.10000000149011612,1.0,1.1000000238418579,0.0,-0.10000000149011612,1.0,0.89999997615814209,0.0,10.0,1.0,11.0,0.0,10.100000381469727,1.0,11.100000381469727,0.0,100.0,1.0,101.0> : tensor<1x6x4xf32>> : !mo.tensor<[1, 6, 4], f32>
    %scores = mo.constant <#M.dense_array<0.9, 0.75, 0.6, 0.95, 0.5, 0.3> : tensor<1x1x6xf32>> : !mo.tensor<[1, 1, 6], f32>
    %two = mo.constant <#M.dense_array<2> : tensor<1xsi64>> : !mo.tensor<[], si64>
    %maxOutputBoxesPerClass = mo.mul(%one, %two) : !mo.tensor<[], si64>
    %iouThreshold = mo.constant <#M.dense_array<0.5> : tensor<1xf32>> : !mo.tensor<[], f32>
    %scoreThreshold = mo.constant <#M.dense_array<0.0> : tensor<1xf32>> : !mo.tensor<[], f32>
    %output = mo.non_maximum_suppression<() -> D2, D3>(%boxes, %scores, %maxOutputBoxesPerClass, %iouThreshold, %scoreThreshold) : (!mo.tensor<[1, 6, 4], f32>, !mo.tensor<[1, 1, 6], f32>, !mo.tensor<[], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[D2, D3], si64>
    mo.output<D2, D3> %output : !mo.tensor<[D2, D3], si64>
}
// CHECK{LITERAL}: 'non_maximum_suppression' returned tensor<2x3xsi64> [0, 0, 3, 0, 0, 0]

mo.graph @non_maximum_suppression_fallback_shape_func<() -> D0, D1>(%one : !mo.tensor<[], si64>) -> !mo.tensor<[D0, D1], si64> {
    %boxes = mo.constant <#M.dense_array<0.0,0.0,1.0,1.0,0.0,0.10000000149011612,1.0,1.1000000238418579,0.0,-0.10000000149011612,1.0,0.89999997615814209,0.0,10.0,1.0,11.0,0.0,10.100000381469727,1.0,11.100000381469727,0.0,100.0,1.0,101.0> : tensor<1x6x4xf32>> : !mo.tensor<[1, 6, 4], f32>
    %scores = mo.constant <#M.dense_array<0.9, 0.75, 0.6, 0.95, 0.5, 0.3> : tensor<1x1x6xf32>> : !mo.tensor<[1, 1, 6], f32>
    %two = mo.constant <#M.dense_array<2> : tensor<1xsi64>> : !mo.tensor<[], si64>
    %maxOutputBoxesPerClass = mo.mul(%one, %two) : !mo.tensor<[], si64>
    %iouThreshold = mo.constant <#M.dense_array<0.5> : tensor<1xf32>> : !mo.tensor<[], f32>
    %scoreThreshold = mo.constant <#M.dense_array<0.0> : tensor<1xf32>> : !mo.tensor<[], f32>
    mo.invoke_shape_func["mo.non_maximum_suppression"]<() -> D2, D3>(%boxes, %scores, %maxOutputBoxesPerClass, %iouThreshold, %scoreThreshold) [dataDeptTensors = [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32]] : (!mo.tensor<[1, 6, 4], f32>, !mo.tensor<[1, 1, 6], f32>, !mo.tensor<[], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>)
    %val = mo.static.broadcast_to(%one) : !mo.tensor<[], si64> -> !mo.tensor<[D2, D3], si64>
    mo.output<D2, D3> %val : !mo.tensor<[D2, D3], si64>
}
// CHECK-LABEL: --- Running 'non_maximum_suppression_fallback_shape_func':
// CHECK-NEXT: 'non_maximum_suppression_fallback_shape_func' returned tensor<2x3xsi64>
