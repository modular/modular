// RUN: mo-opt %s -o %t.mlir --gc-pipeline="kernel-library-paths=%kernel_api_test_pkg" --allow-unregistered-dialect
// RUN: mt --execute %t.mlir --result-output-style=full --model-inputs=ones | FileCheck %s

// Ensures constant fusion keeps positional trait-contract metadata aligned when
// a trait-bound tensor kernel also receives a Mojo `Scalar`.

// CHECK-LABEL: Running 'trait_supports_constant_scalar':
// CHECK-NEXT: datatype is float32
// CHECK-NEXT: value is 2.0
// CHECK-NEXT: 'trait_supports_constant_scalar' returned tensor<f32>
mo.graph @trait_supports_constant_scalar(
  %input : !mo.tensor<[], f32>
) -> !mo.tensor<[], f32> {
  %scalar = mo.constant <#M.dense_array<2.0> : tensor<f32>> : !mo.tensor<[], f32>
  %output = mo.custom [symbol = "trait_supports_scalar_kernel", device = #M.device_ref<"cpu", 0>](%input, %scalar) : (!mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[], f32>
  mo.output %output : !mo.tensor<[], f32>
}
