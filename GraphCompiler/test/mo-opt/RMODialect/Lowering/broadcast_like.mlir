// RUN: mo-opt --split-input-file --mo-legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

// All tests in this file exercised broadcast_to / rmo.rebind_tensor_shape
// patterns where the result was an unranked tensor (`!mo.tensor<Sh, dtype>`).
// Unranked tensors are now banned, so these tests are no longer applicable.

// CHECK: module
mo.graph @placeholder(%arg0: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> {
  mo.output %arg0 : !mo.tensor<[2, 3], f32>
}
