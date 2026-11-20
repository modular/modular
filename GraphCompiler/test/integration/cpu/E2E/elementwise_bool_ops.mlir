// RUN: mo-opt %s --mo-shape-inference --gc-pipeline=constant-fold=false -o %t.mlir

// RUN: mo-opt %s --mo-shape-inference --gc-pipeline="constant-fold=false fuse-jit-kernels=false" -o %t_no_fuse.mlir
// RUN: python3 %mo_vs_numpy_py -n ElementwiseBoolOps -f %t_no_fuse.mlir | FileCheck %s

// CHECK: All MO values match numpy

module {
  mo.graph @serving_default<D0, D1, D2, D3, D4, D5, D6, D7>(%arg0: !mo.tensor<[D0, D1, D2, D3], bool>, %arg1: !mo.tensor<[D4, D5, D6, D7], f32>) -> !mo.tensor<[D0, D1, D2, D3], bool> attributes {argument_names = ["input_1", "input_2"], result_names = ["lambda"]} {
    %0 = mo.is_inf(%arg1) : (!mo.tensor<[D4, D5, D6, D7], f32>) -> !mo.tensor<[D4, D5, D6, D7], bool>
    %1 = mo.is_nan(%arg1) : (!mo.tensor<[D4, D5, D6, D7], f32>) -> !mo.tensor<[D4, D5, D6, D7], bool>
    %2 = mo.shape_of(%1) : (!mo.tensor<[D4, D5, D6, D7], bool>) -> !mo.tensor<[4], si64>
    %3 = mo.shape_of(%arg0) : (!mo.tensor<[D0, D1, D2, D3], bool>) -> !mo.tensor<[4], si64>
    %4 = mo.broadcast_shape(%2, %3) : (!mo.tensor<[4], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[4], si64>
    %5 = mo.broadcast_to(%1, %4) : (!mo.tensor<[D4, D5, D6, D7], bool>, !mo.tensor<[4], si64>) -> !mo.tensor<[D0, D1, D2, D3], bool>
    %6 = mo.broadcast_to(%arg0, %4) : (!mo.tensor<[D0, D1, D2, D3], bool>, !mo.tensor<[4], si64>) -> !mo.tensor<[D0, D1, D2, D3], bool>
    %7 = mo.and(%5, %6) : !mo.tensor<[D0, D1, D2, D3], bool>
    %8 = mo.shape_of(%7) : (!mo.tensor<[D0, D1, D2, D3], bool>) -> !mo.tensor<[4], si64>
    %9 = mo.shape_of(%0) : (!mo.tensor<[D4, D5, D6, D7], bool>) -> !mo.tensor<[4], si64>
    %10 = mo.broadcast_shape(%8, %9) : (!mo.tensor<[4], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[4], si64>
    %11 = mo.broadcast_to(%7, %10) : (!mo.tensor<[D0, D1, D2, D3], bool>, !mo.tensor<[4], si64>) -> !mo.tensor<[D0, D1, D2, D3], bool>
    %12 = mo.broadcast_to(%0, %10) : (!mo.tensor<[D4, D5, D6, D7], bool>, !mo.tensor<[4], si64>) -> !mo.tensor<[D0, D1, D2, D3], bool>
    %13 = mo.equal(%11, %12) : (!mo.tensor<[D0, D1, D2, D3], bool>, !mo.tensor<[D0, D1, D2, D3], bool>) -> !mo.tensor<[D0, D1, D2, D3], bool>
    %14 = mo.shape_of(%13) : (!mo.tensor<[D0, D1, D2, D3], bool>) -> !mo.tensor<[4], si64>
    %15 = mo.shape_of(%arg0) : (!mo.tensor<[D0, D1, D2, D3], bool>) -> !mo.tensor<[4], si64>
    %16 = mo.broadcast_shape(%14, %15) : (!mo.tensor<[4], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[4], si64>
    %17 = mo.broadcast_to(%13, %16) : (!mo.tensor<[D0, D1, D2, D3], bool>, !mo.tensor<[4], si64>) -> !mo.tensor<[D0, D1, D2, D3], bool>
    %18 = mo.broadcast_to(%arg0, %16) : (!mo.tensor<[D0, D1, D2, D3], bool>, !mo.tensor<[4], si64>) -> !mo.tensor<[D0, D1, D2, D3], bool>
    %19 = mo.not_equal(%17, %18) : (!mo.tensor<[D0, D1, D2, D3], bool>, !mo.tensor<[D0, D1, D2, D3], bool>) -> !mo.tensor<[D0, D1, D2, D3], bool>
    mo.output %19 : !mo.tensor<[D0, D1, D2, D3], bool>
  }
}
