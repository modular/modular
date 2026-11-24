// RUN: mo-opt %s --gc-pipeline="constant-fold=false" -o %t.mlir
// RUN: mt --model-inputs=ones --execute --result-output-style=full %t.mlir | FileCheck %s

// CHECK-LABEL: Running 'transfer_concat':
// CHECK-NEXT: tensor<6x3xf32> [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00]

mo.graph @transfer_concat(%arg0: !mo.tensor<[6, 3],f32>) -> !mo.tensor<[6, 3],f32> {
    %c = mo.chain.create()
    %0 = mo.constant <#M.dense_array<1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00> : tensor<3x3xf32>> : !mo.tensor<[3, 3],f32>
    %1 = mo.constant <#M.dense_array<2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00> : tensor<3x3xf32>> : !mo.tensor<[3, 3],f32>
    %g0, %outChain_0 = mo.transfer[%c] %0 : !mo.tensor<[3, 3],f32> to <"gpu", 0>
    %g1, %outChain_1 = mo.transfer[%c] %1 : !mo.tensor<[3, 3],f32> to <"gpu", 0>
    %g2, %outChain_2 = mo.transfer[%c] %arg0 : !mo.tensor<[6, 3],f32> to <"gpu", 0>
    %12 = mogg.concat(%g0, %g1) : (!mo.tensor<[3, 3], f32, gpu:0>, !mo.tensor<[3, 3], f32, gpu:0>) -> !mo.tensor<[6, 3], f32, gpu:0>
    %13 = mo.add(%g2, %12) : !mo.tensor<[6, 3], f32, gpu:0>
    %out, %outChain_3 = mo.transfer[%c] %13 : !mo.tensor<[6, 3], f32, gpu:0> to <"cpu", 0>
    mo.output %out : !mo.tensor<[6, 3],f32>
}
