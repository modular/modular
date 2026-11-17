// RUN: mo-opt --split-input-file --mo-validate-devices="default-device-target-triple=arm64-apple-darwin23.6.0 default-device-target-arch=aarch64" %s

// RUN: mo-opt --split-input-file --mo-validate-devices="default-device-target-triple=x86_64-unknown-linux default-device-target-arch=x86_64" %s

// A host may move FP8 and E8M0 bytes between devices.

mo.graph @transfer_f8e4m3_to_host() -> !mo.tensor<[4], f8e4m3fn> {
  %0 = mo.chain.create()
  %1 = mo.constant.external[name = "weight", align = 16, device = #M.device_ref<"gpu", 0>] : !mo.tensor<[4], f8e4m3fn, gpu:0>
  %2, %3 = mo.transfer[%0] %1 : !mo.tensor<[4], f8e4m3fn, gpu:0> to <"cpu", 0>
  mo.output %2 : !mo.tensor<[4], f8e4m3fn>
}

// -----

mo.graph @transfer_f8e8m0_from_host() -> !mo.tensor<[4], f8e8m0fnu, gpu:0> {
  %0 = mo.chain.create()
  %1 = mo.constant.external[name = "weight_scale", align = 16, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[4], f8e8m0fnu>
  %2, %3 = mo.transfer[%0] %1 : !mo.tensor<[4], f8e8m0fnu> to <"gpu", 0>
  mo.output %2 : !mo.tensor<[4], f8e8m0fnu, gpu:0>
}
