// RUN: mo-opt %s --gc-pipeline="constant-fold=false" -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s

// softmax reduces with the target's native SIMD width, so a
// row of 5 tiles differently on ARM (4-wide) vs x86 (8-wide); float
// addition isn't associative, so only the leading digits of each value
// are architecture-stable (same convention as layer_norm.mlir).

mo.graph @softmax() -> (!mo.tensor<[1, 5], f32>) {
  %0 = "mo.constant"() {value = #M.dense_array<-1.0, 2.0, -3.0, -4.0, 5.0> : tensor<1x5xf32>} : () -> !mo.tensor<[1, 5], f32>
  %1 = mo.composite.reduce.softmax(%0) [axis = 1] : (!mo.tensor<[1, 5], f32>) -> !mo.tensor<[1, 5], f32>
  mo.output %1 : !mo.tensor<[1, 5], f32>
}
// CHECK: 'softmax' returned tensor<1x5xf32> [2.3546{{[0-9]*}}e-03, 4.7293{{[0-9]*}}e-02, 3.1866{{[0-9]*}}e-04, 1.1722{{[0-9]*}}e-04, 9.4991{{[0-9]*}}e-01]

mo.graph @softmax_3D() -> (!mo.tensor<[1, 2, 3], f32>) {
  %0 = "mo.constant"() {value = #M.dense_array<-1.0, 2.0, -3.0, -4.0, 5.0, 2.0> : tensor<1x2x3xf32>} : () -> !mo.tensor<[1, 2, 3], f32>
  %1 = mo.composite.reduce.softmax(%0) [axis = 2] : (!mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[1, 2, 3], f32>
  mo.output %1 : !mo.tensor<[1, 2, 3], f32>
}
// CHECK: 'softmax_3D' returned tensor<1x2x3xf32> [4.7123{{[0-9]*}}e-02, 9.4649{{[0-9]*}}e-01, 6.3774{{[0-9]*}}e-03, 1.1754{{[0-9]*}}e-04, 9.5246{{[0-9]*}}e-01, 4.7420{{[0-9]*}}e-02]
