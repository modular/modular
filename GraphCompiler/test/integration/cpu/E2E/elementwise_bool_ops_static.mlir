// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: python3 %mo_vs_numpy_py -n ElementwiseBoolOpsStatic -f %t.mlir | FileCheck %s

// RUN: mo-opt %s --gc-pipeline="constant-fold=false fuse-jit-kernels=false" -o %t_no_fuse.mlir
// RUN: python3 %mo_vs_numpy_py -n ElementwiseBoolOpsStatic -f %t_no_fuse.mlir | FileCheck %s

// CHECK: All MO values match numpy

module {
  mo.graph @serving_default(%arg0: !mo.tensor<[20, 5, 1, 6], bool>, %arg1: !mo.tensor<[20, 5, 1, 6], f32>) -> !mo.tensor<[20, 5, 1, 6], bool> attributes {argument_names = ["input_1", "input_2"], result_names = ["lambda"]} {
    %0 = mo.is_inf(%arg1) : (!mo.tensor<[20, 5, 1, 6], f32>) -> !mo.tensor<[20, 5, 1, 6], bool>
    %1 = mo.is_nan(%arg1) : (!mo.tensor<[20, 5, 1, 6], f32>) -> !mo.tensor<[20, 5, 1, 6], bool>
    %2 = mo.and(%1, %arg0) : !mo.tensor<[20, 5, 1, 6], bool>
    %3 = mo.equal(%2, %0) : (!mo.tensor<[20, 5, 1, 6], bool>, !mo.tensor<[20, 5, 1, 6], bool>) -> !mo.tensor<[20, 5, 1, 6], bool>
    %4 = mo.not_equal(%3, %arg0) : (!mo.tensor<[20, 5, 1, 6], bool>, !mo.tensor<[20, 5, 1, 6], bool>) -> !mo.tensor<[20, 5, 1, 6], bool>
    mo.output %4 : !mo.tensor<[20, 5, 1, 6], bool>
  }
}
