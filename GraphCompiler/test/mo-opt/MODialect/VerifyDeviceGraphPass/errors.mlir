// RUN: mo-opt --split-input-file --verify-diagnostics \
// RUN:   --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" \
// RUN:   --mo-verify-device-graph %s

// Device-graphable graph whose body uses a kernel that only exposes
// `execute` (no `build_graph`). The verifier must emit an error on the
// offending op.
mo.graph @missing_build_graph(%arg0: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> isDeviceGraph {
  // expected-error @below {{op 'mo.custom' is in a device-graphable mo.graph but its kernel registration ('imposter_add_elementwise'}}
  %0 = mo.custom [symbol = "imposter_add_elementwise", device = #M.device_ref<"cpu", 0>]
    (%arg0, %arg0) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>)
                  -> !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// -----

// Device-graphable graph cannot contain control-flow ops.
// `mo.if` is hard-rejected (will be relaxed in a later milestone).
mo.graph @rejects_if(%cond: !mo.tensor<[], bool>) -> !mo.tensor<[2, 3], si32> isDeviceGraph {
  // expected-error @below {{device-graphable mo.graph cannot contain control-flow op 'mo.if'}}
  %ret = mo.if (%cond : !mo.tensor<[], bool>) -> !mo.tensor<[2, 3], si32> {
    %cst = mo.constant <#M.dense_array<1, 0, 0, 1, 0, 1> : tensor<2x3xsi32>> : !mo.tensor<[2, 3], si32>
    mo.yield %cst : !mo.tensor<[2, 3], si32>
  } else {
    %cst = mo.constant <#M.dense_array<1, 0, 0, 1, 0, 1> : tensor<2x3xsi32>> : !mo.tensor<[2, 3], si32>
    mo.yield %cst : !mo.tensor<[2, 3], si32>
  }
  mo.output %ret : !mo.tensor<[2, 3], si32>
}
