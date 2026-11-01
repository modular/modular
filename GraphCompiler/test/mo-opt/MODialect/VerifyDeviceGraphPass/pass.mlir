// RUN: mo-opt --split-input-file --verify-diagnostics \
// RUN:   --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" \
// RUN:   --mo-verify-device-graph %s

// Graph without the isDeviceGraph attribute is a no-op for the verifier,
// regardless of whether its body uses kernels that lack `build_graph`.
mo.graph @no_attr(%arg0: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> {
  %0 = mo.custom [symbol = "imposter_add_elementwise", device = #M.device_ref<"cpu", 0>]
    (%arg0, %arg0) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>)
                  -> !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// -----

// Without the isDeviceGraph attribute, control-flow ops are allowed.
mo.graph @control_flow_no_attr(%cond: !mo.tensor<[], bool>) -> !mo.tensor<[2, 3], si32> {
  %ret = mo.if (%cond : !mo.tensor<[], bool>) -> !mo.tensor<[2, 3], si32> {
    %cst = mo.constant <#M.dense_array<1, 0, 0, 1, 0, 1> : tensor<2x3xsi32>> : !mo.tensor<[2, 3], si32>
    mo.yield %cst : !mo.tensor<[2, 3], si32>
  } else {
    %cst = mo.constant <#M.dense_array<1, 0, 0, 1, 0, 1> : tensor<2x3xsi32>> : !mo.tensor<[2, 3], si32>
    mo.yield %cst : !mo.tensor<[2, 3], si32>
  }
  mo.output %ret : !mo.tensor<[2, 3], si32>
}

// -----

// Graph flagged for device-graph synthesis whose body contains only intrinsic
// MO ops (mo.constant, mo.output). Intrinsic ops have no kernel registration
// and are silently skipped.
mo.graph @intrinsic_only(%arg0: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> isDeviceGraph {
  mo.output %arg0 : !mo.tensor<[2, 3], f32>
}

// -----

// Graph flagged for device-graph synthesis whose body uses a kernel that
// exposes a `build_graph` entry point. Should pass cleanly.
mo.graph @with_build_graph(%arg0: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> isDeviceGraph {
  %0 = mo.custom [symbol = "imposter_build_graph_capable", device = #M.device_ref<"cpu", 0>]
    (%arg0) : (!mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}
