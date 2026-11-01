// RUN: mo-opt %s --mo-shape-inference --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --input-shapes=5 -model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

mo.graph @relu<D0>(%arg0 : !mo.tensor<[D0], f32>) -> !mo.tensor<[D0], f32> {
  %0 = mo.relu(%arg0) : !mo.tensor<[D0], f32>
  mo.output %0 : !mo.tensor<[D0], f32>
}
// CHECK{LITERAL}: 'relu' returned tensor<5xf32> [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]

mo.graph @cast<D0>(%arg0 : !mo.tensor<[D0], f32>) -> !mo.tensor<[D0], si32> {
  %1 = mo.cast(%arg0) : (!mo.tensor<[D0], f32>) -> !mo.tensor<[D0], si32>
  mo.output %1 : !mo.tensor<[D0], si32>
}
// CHECK: 'cast' returned tensor<5xsi32> [1, 1, 1, 1, 1]

mo.graph @log<D0>(%arg0 : !mo.tensor<[D0], f32>) -> !mo.tensor<[D0], f32> {
  %1 = mo.log(%arg0) : !mo.tensor<[D0], f32>
  mo.output %1 : !mo.tensor<[D0], f32>
}

// CHECK: 'log' returned tensor<5xf32> [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]

mo.graph @not<D0>(%arg0 : !mo.tensor<[D0], bool>) -> !mo.tensor<[D0], bool> {
  %1 = mo.not(%arg0) : !mo.tensor<[D0], bool>
  mo.output %1 : !mo.tensor<[D0], bool>
}

// CHECK: 'not' returned tensor<5xbool> [False, False, False, False, False]

mo.graph @negative<D0>(%arg0 : !mo.tensor<[D0], f32>) -> !mo.tensor<[D0], f32> {
  %1 = mo.negative(%arg0) : !mo.tensor<[D0], f32>
  mo.output %1 : !mo.tensor<[D0], f32>
}

// CHECK: 'negative' returned tensor<5xf32> [-1.000000e+00, -1.000000e+00, -1.000000e+00, -1.000000e+00, -1.000000e+00]

mo.graph @sin<D0>(%arg0 : !mo.tensor<[D0], f32>) -> !mo.tensor<[D0], f32> {
  %1 = mo.sin(%arg0) : !mo.tensor<[D0], f32>
  mo.output %1 : !mo.tensor<[D0], f32>
}

// CHECK: 'sin' returned tensor<5xf32> [8.414710e-01, 8.414710e-01, 8.414710e-01, 8.414710e-01, 8.414710e-01]

mo.graph @cos<D0>(%arg0 : !mo.tensor<[D0], f32>) -> !mo.tensor<[D0], f32> {
  %1 = mo.cos(%arg0) : !mo.tensor<[D0], f32>
  mo.output %1 : !mo.tensor<[D0], f32>
}

// CHECK: 'cos' returned tensor<5xf32> [5.403023e-01, 5.403023e-01, 5.403023e-01, 5.403023e-01, 5.403023e-01]
