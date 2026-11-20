// RUN: mo-opt %s -o %t.mlir --gc-pipeline="kernel-library-paths=%kernel_api_test_pkg"
// RUN: mt --execute %t.mlir --result-output-style=full --input-shapes=4 --model-inputs=ones | FileCheck %s

// A kernel with a struct parameter and trait arguments must still apply a
// fn-level default the op omits. The trait arguments' implicit origin
// declarations are fn-level params with no slot in the merged param list, so
// counting them when locating the fn-level defaults reads the preceding
// param's slot instead and fails to infer `defaulted`.

// CHECK: Running 'trait_struct_param_default':
// CHECK-NEXT: negate = True
// CHECK-NEXT: required = 7
// CHECK-NEXT: defaulted = -1
// CHECK: 'trait_struct_param_default' returned tensor<4xf32> [-1.000000e+00, -1.000000e+00, -1.000000e+00, -1.000000e+00]
mo.graph @trait_struct_param_default(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
    %z = mo.custom [symbol = "trait_kernel_with_struct_param_default", device = #M.device_ref<"cpu", 0>, parameters = {negate = true, required = 7}] (%x) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
    mo.output %z : !mo.tensor<[4], f32>
}
