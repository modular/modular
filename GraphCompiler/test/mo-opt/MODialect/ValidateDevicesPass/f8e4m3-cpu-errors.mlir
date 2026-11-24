// RUN: mo-opt --split-input-file --verify-diagnostics --mo-validate-devices="default-device-target-triple=arm64-apple-darwin23.6.0 default-device-target-arch=aarch64" %s

// RUN: mo-opt --split-input-file --verify-diagnostics --mo-validate-devices="default-device-target-triple=x86_64-unknown-linux default-device-target-arch=x86_64" %s

mo.graph @cast(%arg0: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f8e4m3fn> {
  %0 = mo.chain.create()
  // expected-error @below {{The f8e4m3fn data type is not supported on device 'cpu:0'.}}
  %1 = mo.cast(%arg0) { device_ref = #M.device_ref<"cpu", 0> } : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f8e4m3fn>
  mo.output %1 : !mo.tensor<[4], f8e4m3fn>
}

// -----

// The host f8e4m3fn input is data movement and is allowed; the cast that
// reads it is not.
mo.graph @cast(%arg0: !mo.tensor<[4], f8e4m3fn>) -> !mo.tensor<[4], f32> {
  %0 = mo.chain.create()
  // expected-error @below {{The f8e4m3fn data type is not supported on device 'cpu:0'.}}
  %1 = mo.cast(%arg0) { device_ref = #M.device_ref<"cpu", 0> } : (!mo.tensor<[4], f8e4m3fn>) -> !mo.tensor<[4], f32>
  mo.output %1 : !mo.tensor<[4], f32>
}

// -----

// `mo.static.reshape` implements no exemption, so a host FP8 reshape is
// rejected.
mo.graph @static_reshape_f8e4m3(%arg0: !mo.tensor<[4], f8e4m3fn>) -> !mo.tensor<[2, 2], f8e4m3fn> {
  %0 = mo.chain.create()
  // expected-error @below {{The f8e4m3fn data type is not supported on device 'cpu:0'.}}
  %1 = mo.static.reshape(%arg0) : !mo.tensor<[4], f8e4m3fn> -> !mo.tensor<[2, 2], f8e4m3fn>
  mo.output %1 : !mo.tensor<[2, 2], f8e4m3fn>
}

// -----

// `mo.static.broadcast_to` implements no exemption either.
mo.graph @static_broadcast_f8e4m3(%arg0: !mo.tensor<[], f8e4m3fn>) -> !mo.tensor<[4], f8e4m3fn> {
  %0 = mo.chain.create()
  // expected-error @below {{The f8e4m3fn data type is not supported on device 'cpu:0'.}}
  %1 = mo.static.broadcast_to(%arg0) : !mo.tensor<[], f8e4m3fn> -> !mo.tensor<[4], f8e4m3fn>
  mo.output %1 : !mo.tensor<[4], f8e4m3fn>
}

// -----

// `mo.debug.tensor.print` reports that not every data type may pass through it.
mo.graph @print_f8e4m3(%arg0: !mo.tensor<[4], f8e4m3fn>) {
  %0 = mo.chain.create()
  // expected-error @below {{The f8e4m3fn data type is not supported on device 'cpu:0'.}}
  mo.debug.tensor.print(%0, %arg0) [label = "weights"] : !mo.tensor<[4], f8e4m3fn>
  mo.output
}
