// RUN: mo-opt --split-input-file --verify-diagnostics --mo-validate-devices="default-device-target-triple=arm64-apple-darwin23.6.0 default-device-target-arch=aarch64" %s

// RUN: mo-opt --split-input-file --verify-diagnostics --mo-validate-devices="default-device-target-triple=x86_64-unknown-linux default-device-target-arch=x86_64" %s

// E8M0 has no host kernels, so a host graph may only move it, never compute on
// it.

mo.graph @cast_to_e8m0(%arg0: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f8e8m0fnu> {
  %0 = mo.chain.create()
  // expected-error @below {{The f8e8m0fnu data type is not supported on device 'cpu:0'.}}
  %1 = mo.cast(%arg0) { device_ref = #M.device_ref<"cpu", 0> } : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f8e8m0fnu>
  mo.output %1 : !mo.tensor<[4], f8e8m0fnu>
}

// -----

// The operand check: the cast's result is f32, but its input is E8M0.
mo.graph @cast_from_e8m0(%arg0: !mo.tensor<[4], f8e8m0fnu>) -> !mo.tensor<[4], f32> {
  %0 = mo.chain.create()
  // expected-error @below {{The f8e8m0fnu data type is not supported on device 'cpu:0'.}}
  %1 = mo.cast(%arg0) { device_ref = #M.device_ref<"cpu", 0> } : (!mo.tensor<[4], f8e8m0fnu>) -> !mo.tensor<[4], f32>
  mo.output %1 : !mo.tensor<[4], f32>
}

// -----

mo.graph @add_e8m0(%arg0: !mo.tensor<[4], f8e8m0fnu>, %arg1: !mo.tensor<[4], f8e8m0fnu>) -> !mo.tensor<[4], f8e8m0fnu> {
  %0 = mo.chain.create()
  // expected-error @below {{The f8e8m0fnu data type is not supported on device 'cpu:0'.}}
  %1 = mo.add(%arg0, %arg1) { device_ref = #M.device_ref<"cpu", 0> } : !mo.tensor<[4], f8e8m0fnu>
  mo.output %1 : !mo.tensor<[4], f8e8m0fnu>
}

// -----

// The add nested in the preserved `mo.if` is still checked.
mo.graph @add_e8m0_in_if(%arg0: !mo.tensor<[4], f8e8m0fnu>, %arg1: !mo.tensor<[4], f8e8m0fnu>, %cond: !mo.tensor<[], bool>) -> !mo.tensor<[4], f8e8m0fnu> {
  %0 = mo.chain.create()
  %1 = mo.if (%cond : !mo.tensor<[], bool>) -> !mo.tensor<[4], f8e8m0fnu> {
    // expected-error @below {{The f8e8m0fnu data type is not supported on device 'cpu:0'.}}
    %2 = mo.add(%arg0, %arg1) { device_ref = #M.device_ref<"cpu", 0> } : !mo.tensor<[4], f8e8m0fnu>
    mo.yield %2 : !mo.tensor<[4], f8e8m0fnu>
  } else {
    mo.yield %arg0 : !mo.tensor<[4], f8e8m0fnu>
  }
  mo.output %1 : !mo.tensor<[4], f8e8m0fnu>
}

// -----

// A preserved op's result is checked once a kernel op consumes it.
mo.graph @cast_of_transferred_e8m0() -> !mo.tensor<[2, 2], f32> {
  %0 = mo.chain.create()
  %1 = mo.constant.external[name = "weight_scale", align = 16, device = #M.device_ref<"gpu", 0>] : !mo.tensor<[2, 2], f8e8m0fnu, gpu:0>
  %2, %3 = mo.transfer[%0] %1 : !mo.tensor<[2, 2], f8e8m0fnu, gpu:0> to <"cpu", 0>
  // expected-error @below {{The f8e8m0fnu data type is not supported on device 'cpu:0'.}}
  %4 = mo.cast(%2) { device_ref = #M.device_ref<"cpu", 0> } : (!mo.tensor<[2, 2], f8e8m0fnu>) -> !mo.tensor<[2, 2], f32>
  mo.output %4 : !mo.tensor<[2, 2], f32>
}
