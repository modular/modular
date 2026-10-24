// RUN: mo-opt %s --mo-shape-inference --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s

mo.graph @range1<() -> D0>(%limit: !mo.tensor<[], f32>) -> !mo.tensor<[D0], f32> {
  %start = mo.constant <#M.dense_array<-4.5> : tensor<f32>> : !mo.tensor<[], f32>
  %step = mo.constant <#M.dense_array<0.5> : tensor<f32>> : !mo.tensor<[], f32>
  %output = mo.range<() -> D1>(%start, %limit, %step) : (!mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[D1], f32>
  mo.output<D1> %output : !mo.tensor<[D1], f32>
}

// print(np.range(-4.5, 0, 0.5))
// CHECK{LITERAL}: 'range1' returned tensor<9xf32> [-4.500000e+00, -4.000000e+00, -3.500000e+00, -3.000000e+00, -2.500000e+00, -2.000000e+00, -1.500000e+00, -1.000000e+00, -5.000000e-01]
