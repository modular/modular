// RUN: mo-opt --split-input-file --mo-validate-devices="default-device-target-triple=arm64-apple-darwin23.6.0 default-device-target-arch=aarch64" %s

// RUN: mo-opt --split-input-file --mo-validate-devices="default-device-target-triple=x86_64-unknown-linux default-device-target-arch=x86_64" %s

// A host may hold FP8 and E8M0 tensors, since nothing here computes on them.

mo.graph @host_f8e4m3_constant() -> !mo.tensor<[4], f8e4m3fn> {
  %0 = mo.constant.external[name = "weight", align = 16, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[4], f8e4m3fn>
  mo.output %0 : !mo.tensor<[4], f8e4m3fn>
}

// -----

mo.graph @host_e8m0_constant() -> !mo.tensor<[4], f8e8m0fnu> {
  %0 = mo.constant.external[name = "weight_scale", align = 16, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[4], f8e8m0fnu>
  mo.output %0 : !mo.tensor<[4], f8e8m0fnu>
}

// -----

mo.graph @host_f8e4m3_input(%arg0: !mo.tensor<[4], f8e4m3fn>) -> !mo.tensor<[4], f8e4m3fn> {
  mo.output %arg0 : !mo.tensor<[4], f8e4m3fn>
}

// -----

mo.graph @host_e8m0_input(%arg0: !mo.tensor<[4], f8e8m0fnu>) -> !mo.tensor<[4], f8e8m0fnu> {
  mo.output %arg0 : !mo.tensor<[4], f8e8m0fnu>
}
