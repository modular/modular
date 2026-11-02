// RUN: mo-opt --split-input-file --verify-diagnostics --mo-validate-devices="default-device-target-triple=arm64-apple-darwin23.6.0 default-device-target-arch=aarch64" %s

// RUN: mo-opt --split-input-file --verify-diagnostics --mo-validate-devices="default-device-target-triple=x86_64-unknown-linux default-device-target-arch=x86_64" %s

mo.graph @cast(%arg0: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f8e5m2> {
  %0 = mo.chain.create()
  // expected-error @below {{The f8e5m2 data type is not supported on device 'cpu:0'.}}
  %1 = mo.cast(%arg0) { device_ref = #M.device_ref<"cpu", 0> } : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f8e5m2>
  mo.output %1 : !mo.tensor<[4], f8e5m2>
}

// -----

// The host f8e5m2 input is data movement and is allowed; the cast that reads
// it is not.
mo.graph @cast(%arg0: !mo.tensor<[4], f8e5m2>) -> !mo.tensor<[4], f32> {
  %0 = mo.chain.create()
  // expected-error @below {{The f8e5m2 data type is not supported on device 'cpu:0'.}}
  %1 = mo.cast(%arg0) { device_ref = #M.device_ref<"cpu", 0> } : (!mo.tensor<[4], f8e5m2>) -> !mo.tensor<[4], f32>
  mo.output %1 : !mo.tensor<[4], f32>
}
