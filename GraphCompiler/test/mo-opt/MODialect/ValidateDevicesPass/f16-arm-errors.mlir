// RUN: mo-opt --split-input-file --verify-diagnostics --mo-validate-devices="default-device-target-triple=arm64-apple-darwin23.6.0 default-device-target-arch=aarch64" %s

// These tests verify that the ValidateDevicesPass correctly reports errors for
// the F16 data type on ARM architectures.

mo.graph @cast(%arg0: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f16> {
  // expected-error @below {{The f16 data type is not supported on device 'cpu:0'.}}
  %0 = mo.cast(%arg0) { device_ref = #M.device_ref<"cpu", 0> } : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f16>
  mo.output %0 : !mo.tensor<[4], f16>
}

// -----

// The host f16 input is data movement and is allowed; the cast that reads it
// is not.
mo.graph @cast(%arg0: !mo.tensor<[4], f16>) -> !mo.tensor<[4], f32> {
  // expected-error @below {{The f16 data type is not supported on device 'cpu:0'.}}
  %0 = mo.cast(%arg0) { device_ref = #M.device_ref<"cpu", 0> } : (!mo.tensor<[4], f16>) -> !mo.tensor<[4], f32>
  mo.output %0 : !mo.tensor<[4], f32>
}
