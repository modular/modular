// RUN: mo-opt %s --gc-pipeline=kernel-library-paths="%kernel_api_test_pkg" -o %t.mlir
// RUN: mt --execute %t.mlir --model-inputs=ones --result-output-style=full | FileCheck %s

// CHECK-LABEL: Running 'supports_scalars':
// CHECK-NEXT: datatype is float32
// CHECK-NEXT: value is 1.0
// CHECK-NEXT: 'supports_scalars' returned tensor<f32>
// Ensures we can properly lower and run a kernel which uses a mojo `Scalar`
mo.graph @supports_scalars(%a : !mo.tensor<[], f32>) -> !mo.tensor<[], f32> {
  %0 = mo.custom [symbol = "supports_scalar_kernel", device = #M.device_ref<"cpu", 0>](%a, %a) : (!mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[], f32>
  mo.output %0 : !mo.tensor<[], f32>
}
