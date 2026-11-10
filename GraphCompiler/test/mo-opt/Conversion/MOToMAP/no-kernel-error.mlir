// RUN: mo-opt %s --mo-to-map --verify-diagnostics

// Nothing after `mo-to-map` lowers an MO kernel op, so an op with no
// registered kernel fails the pass instead of lingering.

mo.graph @customWithoutKernel(%input: !mo.tensor<[4, 3], f32>) -> !mo.tensor<[4, 3], f32> {
  // expected-error @below {{'mo.custom' op MOToMAP: no kernel registered for 'user_op_without_kernel'}}
  %res = mo.custom [symbol = "user_op_without_kernel", device = #M.device_ref<"cpu", 0>](%input) : (!mo.tensor<[4, 3], f32>) -> !mo.tensor<[4, 3], f32>
  mo.output %res : !mo.tensor<[4, 3], f32>
}
