// RUN: mo-opt %s --mo-shape-inference --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: python3 %mo_vs_numpy_py -n Range -f %t.mlir | FileCheck %s

// RUN: mo-opt %s --mo-shape-inference --gc-pipeline="constant-fold=false fuse-jit-kernels=false" -o %t_no_fuse.mlir
// RUN: python3 %mo_vs_numpy_py -n Range -f %t_no_fuse.mlir | FileCheck %s

// CHECK: All MO values match numpy

mo.graph @serving_default<() -> D0>(%arg0: !mo.tensor<[3], si32>) -> !mo.tensor<[D0], si32> attributes {argument_names = ["input_1"], result_names = ["lambda"]} {
  %0 = mo.constant <#M.dense_array<1, 1, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %2:3 = mo.split(%arg0, %0) [axis = 0] : (!mo.tensor<[3], si32>, !mo.tensor<[3], si64>) -> (!mo.tensor<[1], si32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>)
  %3 = mo.static.reshape(%2#0) : !mo.tensor<[1], si32> -> !mo.tensor<[], si32>
  %4 = mo.static.reshape(%2#1) : !mo.tensor<[1], si32> -> !mo.tensor<[], si32>
  %5 = mo.static.reshape(%2#2) : !mo.tensor<[1], si32> -> !mo.tensor<[], si32>
  %6 = mo.range<() -> D1>(%3, %4, %5) : (!mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[D1], si32>
  mo.output<D1> %6 : !mo.tensor<[D1], si32>
}
