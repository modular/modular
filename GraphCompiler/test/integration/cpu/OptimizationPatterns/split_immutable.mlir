// RUN: mo-opt %s --mo-shape-inference --gc-pipeline=constant-fold=false -o %t
// RUN: mt --execute %t --input-shapes=1x1024x1024x3 | FileCheck %s

mo.graph @split_immutable<D0, D1>(%arg0: !mo.tensor<[1, D0, D1, 3], f32>) -> !mo.tensor<[1, D0, D1, 3], f32> {
    %one = mo.constant <#M.dense_array<1> : tensor<1xsi64>> : !mo.tensor<[1], si64>
    %split_out = mo.split(%arg0, %one) [axis = 0] : (!mo.tensor<[1, D0, D1, 3], f32>, !mo.tensor<[1], si64>) -> (!mo.tensor<[1, D0, D1, 3], f32>)

    %0 = mo.add(%split_out, %split_out) : !mo.tensor<[1, D0, D1, 3], f32>
    mo.output %0 : !mo.tensor<[1, D0, D1, 3], f32>
}

// CHECK-LABEL: --- Running 'split_immutable':
// CHECK{LITERAL}: 'split_immutable' returned tensor([[[[0.0000, 0.0000, 0.0000],
// CHECK{LITERAL}: [0.0000, 0.0000, 0.0000]]]], dtype=f32, shape=[1,1024,1024,3])
