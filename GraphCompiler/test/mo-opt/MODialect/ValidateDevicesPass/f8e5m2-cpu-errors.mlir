// RUN: mo-opt --split-input-file --verify-diagnostics --mo-validate-devices="default-device-target-triple=arm64-apple-darwin23.6.0 default-device-target-arch=aarch64" --gc-pipeline %s

// RUN: mo-opt --split-input-file --verify-diagnostics --mo-validate-devices="default-device-target-triple=x86_64-unknown-linux default-device-target-arch=x86_64" --gc-pipeline %s

mo.graph @cast<dim>(%arg0: !mo.tensor<[dim], f32>) -> !mo.tensor<[dim], f8e5m2> attributes {argument_names = ["input0"], result_names = ["output0"]} {
  %0 = mo.chain.create()
  // expected-error @below {{The f8e5m2 data type is not supported on device 'cpu:0'.}}
  %1 = mo.cast(%arg0) { device_ref = #M.device_ref<"cpu", 0> } : (!mo.tensor<[dim], f32>) -> !mo.tensor<[dim], f8e5m2>
  mo.output %1 : !mo.tensor<[dim], f8e5m2>
}

// -----

// expected-error @below {{The f8e5m2 data type is not supported on device 'cpu:0'.}}
mo.graph @cast<dim>(%arg0: !mo.tensor<[dim], f8e5m2>) -> !mo.tensor<[dim], f32> attributes {argument_names = ["input0"], result_names = ["output0"]} {
  %0 = mo.chain.create()
  %1 = mo.cast(%arg0) { device_ref = #M.device_ref<"cpu", 0> } : (!mo.tensor<[dim], f8e5m2>) -> !mo.tensor<[dim], f32>
  mo.output %1 : !mo.tensor<[dim], f32>
}
