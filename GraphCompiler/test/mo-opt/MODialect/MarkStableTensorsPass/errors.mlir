// RUN: mo-opt --split-input-file --verify-diagnostics \
// RUN:   --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" \
// RUN:   --mo-mark-stable-tensors %s

// The op's operands must map onto the kernel's `build_graph` signature before
// any stable position can be named; a mismatch is reported on the op.
mo.graph @too_many_operands(%arg0: !mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0>
    attributes {isDeviceGraph} {
  // expected-error @below {{cannot map operands onto the `build_graph` function of kernel}}
  %0 = mo.custom [symbol = "imposter_stable_single_build_graph", device = #M.device_ref<"gpu", 0>]
    (%arg0, %arg0) : (!mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0>
  mo.output %0 : !mo.tensor<[4], f32, gpu:0>
}
