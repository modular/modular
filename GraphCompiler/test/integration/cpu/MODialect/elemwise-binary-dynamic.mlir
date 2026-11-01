// RUN: mo-opt %s --mo-shape-inference --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --input-shapes=5 --input-shapes=5 -model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

mo.graph @add<D0>(%arg0 : !mo.tensor<[D0], f32>, %arg1 : !mo.tensor<[D0], f32>) -> !mo.tensor<[D0], f32> {
  %0 = mo.add(%arg0, %arg1) : !mo.tensor<[D0], f32>
  mo.output %0 : !mo.tensor<[D0], f32>
}
// CHECK{LITERAL}: 'add' returned tensor<5xf32> [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]

mo.graph @max<D0>(%arg0 : !mo.tensor<[D0], f32>, %arg1 : !mo.tensor<[D0], f32>) -> !mo.tensor<[D0], f32> {
  %0 = mo.max(%arg0, %arg1) : !mo.tensor<[D0], f32>
  mo.output %0 : !mo.tensor<[D0], f32>
}
// CHECK{LITERAL}: 'max' returned tensor<5xf32> [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]

mo.graph @mul<D0>(%arg0 : !mo.tensor<[D0], f32>, %arg1 : !mo.tensor<[D0], f32>) -> !mo.tensor<[D0], f32> {
  %0 = mo.mul(%arg0, %arg1) : !mo.tensor<[D0], f32>
  mo.output %0 : !mo.tensor<[D0], f32>
}
// CHECK{LITERAL}: 'mul' returned tensor<5xf32> [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]

mo.graph @min<D0>(%arg0 : !mo.tensor<[D0], f32>, %arg1 : !mo.tensor<[D0], f32>) -> !mo.tensor<[D0], f32> {
  %0 = mo.min(%arg0, %arg1) : !mo.tensor<[D0], f32>
  mo.output %0 : !mo.tensor<[D0], f32>
}
// CHECK{LITERAL}: 'min' returned tensor<5xf32> [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]

mo.graph @mod<D0>(%arg0 : !mo.tensor<[D0], si32>, %arg1 : !mo.tensor<[D0], si32>) -> !mo.tensor<[D0], si32> {
  %0 = mo.mod(%arg0, %arg1) : !mo.tensor<[D0], si32>
  mo.output %0 : !mo.tensor<[D0], si32>
}
// CHECK{LITERAL}: 'mod' returned tensor<5xsi32> [0, 0, 0, 0, 0]

mo.graph @sub<D0>(%arg0 : !mo.tensor<[D0], f32>, %arg1 : !mo.tensor<[D0], f32>) -> !mo.tensor<[D0], f32> {
  %0 = mo.sub(%arg0, %arg1) : !mo.tensor<[D0], f32>
  mo.output %0 : !mo.tensor<[D0], f32>
}
// CHECK{LITERAL}: 'sub' returned tensor<5xf32> [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]


mo.graph @div<D0>(%arg0 : !mo.tensor<[D0], f32>, %arg1 : !mo.tensor<[D0], f32>) -> !mo.tensor<[D0], f32> {
  %0 = mo.div(%arg0, %arg1) : !mo.tensor<[D0], f32>
  mo.output %0 : !mo.tensor<[D0], f32>
}
// CHECK{LITERAL}: 'div' returned tensor<5xf32> [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]


mo.graph @and<D0>(%arg0 : !mo.tensor<[D0], bool>, %arg1 : !mo.tensor<[D0], bool>) -> !mo.tensor<[D0], bool> {
  %0 = mo.and(%arg0, %arg1) : !mo.tensor<[D0], bool>
  mo.output %0 : !mo.tensor<[D0], bool>
}
// CHECK{LITERAL}: 'and' returned tensor<5xbool> [True, True, True, True, True]


mo.graph @xor<D0>(%arg0 : !mo.tensor<[D0], bool>, %arg1 : !mo.tensor<[D0], bool>) -> !mo.tensor<[D0], bool> {
  %0 = mo.xor(%arg0, %arg1) : !mo.tensor<[D0], bool>
  mo.output %0 : !mo.tensor<[D0], bool>
}
// CHECK{LITERAL}: 'xor' returned tensor<5xbool> [False, False, False, False, False]
