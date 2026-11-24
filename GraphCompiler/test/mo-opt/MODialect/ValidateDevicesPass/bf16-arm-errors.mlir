// RUN: mo-opt --split-input-file --mo-validate-devices="default-device-target-triple=arm64-apple-darwin23.6.0 default-device-target-arch=aarch64" --gc-pipeline %s

// These tests verify that the ValidateDevicesPass allows the BF16 data type
// on ARM architectures (bf16 is permitted for casting purposes).

mo.graph @cast<dim>(%arg0: !mo.tensor<[dim], f32>) -> !mo.tensor<[dim], bf16> attributes {argument_names = ["input0"], result_names = ["output0"]} {
  %0 = mo.cast(%arg0) { device_ref = #M.device_ref<"cpu", 0> } : (!mo.tensor<[dim], f32>) -> !mo.tensor<[dim], bf16>
  mo.output %0 : !mo.tensor<[dim], bf16>
}

// -----

mo.graph @cast<dim>(%arg0: !mo.tensor<[dim], bf16>) -> !mo.tensor<[dim], f32> attributes {argument_names = ["input0"], result_names = ["output0"]} {
  %0 = mo.cast(%arg0) { device_ref = #M.device_ref<"cpu", 0> } : (!mo.tensor<[dim], bf16>) -> !mo.tensor<[dim], f32>
  mo.output %0 : !mo.tensor<[dim], f32>
}
