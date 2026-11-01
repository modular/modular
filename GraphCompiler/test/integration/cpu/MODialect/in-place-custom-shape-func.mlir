// RUN: mo-opt --gc-pipeline="kernel-library-paths=%kernel_api_test_pkg" %s -o %t.mlir
// RUN: mt --execute --input-shapes=1x1 --result-output-style=full %t.mlir | FileCheck %s

mo.graph @dyn_shape_func<D0, D1 -> D4, D5>(%arg0: !mo.buffer<[D0, D1], f32>) -> (!mo.buffer<[D0, D1], f32>, !mo.tensor<[D4, D5], f32>, !mo.chain) {
  %0 = mo.chain.create()
  %1:2 = mo.custom<() -> D2, D3> [symbol = "shape_func_with_mutable"](%arg0, %0) : (!mo.buffer<[D0, D1], f32>, !mo.chain) -> (!mo.tensor<[D2, D3], f32>, !mo.chain)
  mo.output<D2, D3> %arg0, %1#0, %1#1 : !mo.buffer<[D0, D1], f32>, !mo.tensor<[D2, D3], f32>, !mo.chain
}

// CHECK-LABEL: --- Running 'dyn_shape_func':
// CHECK{LITERAL}: 'dyn_shape_func' returned tensor<1x1xf32> [2.000000e+00]
// CHECK{LITERAL}: tensor<1x1xf32> [1.000000e+00]
