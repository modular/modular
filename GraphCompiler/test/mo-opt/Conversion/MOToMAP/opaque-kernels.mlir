// RUN: mo-opt --mlir-print-local-scope %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map | FileCheck %s
// RUN: mo-opt --mlir-print-local-scope %s --mo-to-map | FileCheck %s --check-prefix=DISABLED

// MOToMAP lowers each MO op with a registered opaque (non-elementwise,
// non-view) kernel into a `map.iter.opaque` wrapping a `map.call.opaque`,
// driven by the kernel's operand mapping. This test exercises the output-fusion
// coverage the pass supports today, using dedicated signature-only kernels from
// %kernel_api_test_lib (the `map_lower_*` kernels in the testing package):
//
//   - a fused output (FusedOutputTensor) becomes a store epilogue passed to the
//     call, while
//   - a plain output (OutputTensor) is passed to the call raw.
//
// The callee symbol carries the kernel package/module, so it is matched with a
// regex rather than spelled out.

// A single output-fusible result: one store epilogue, and the call writes to
// that epilogue.

// CHECK-LABEL: mo.graph @one_fused
// CHECK: %[[MM:.*]] = map.iter.opaque
// CHECK: inputs (%[[A:.*]]: !mo.tensor<[2, 2], f32>)
// CHECK: outputs (%[[O:.*]] : !mo.tensor<[2, 2], f32>) {
// CHECK:   %[[EPI:.*]] = map.iter.epilogue (%{{.*}} : f32, %{{.*}} : !map.index<2>) capturing {%[[CAP:.*]] = %[[O]] : !mo.tensor<[2, 2], f32>} {
// CHECK:     map.store %[[CAP]][%{{.*}}] = %{{.*}} : !mo.tensor<[2, 2], f32>
// CHECK:   } : !mo.tensor<[2, 2], f32>
// CHECK:   map.call.opaque @{{.*}}MapLowerOneFusedOutput[] inputs(%[[A]]) outputs(%[[EPI]]) kernelName = "map_lower_one_fused_output" : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
// CHECK: } {device = #M.device_ref<"cpu", 0>, fusionInfo = #mogg.fusion<{{.*}}custom__map_lower_one_fused_output
mo.graph @one_fused(%a: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %c = mo.custom [symbol = "map_lower_one_fused_output", device = #M.device_ref<"cpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %c : !mo.tensor<[2, 2], f32>
}

// A single plain (non-fusible) result: no epilogue, and the call writes to the
// raw output block argument.

// CHECK-LABEL: mo.graph @one_plain
// CHECK: %[[MM:.*]] = map.iter.opaque
// CHECK: inputs (%[[A:.*]]: !mo.tensor<[2, 2], f32>)
// CHECK: outputs (%[[O:.*]] : !mo.tensor<[2, 2], f32>) {
// CHECK-NOT: map.iter.epilogue
// CHECK:   map.call.opaque @{{.*}}MapLowerOnePlainOutput[] inputs(%[[A]]) outputs(%[[O]]) kernelName = "map_lower_one_plain_output" : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
mo.graph @one_plain(%a: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %c = mo.custom [symbol = "map_lower_one_plain_output", device = #M.device_ref<"cpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %c : !mo.tensor<[2, 2], f32>
}

// Two output-fusible results: one store epilogue each, and the call writes to
// both epilogues.

// CHECK-LABEL: mo.graph @two_fused
// CHECK: %[[MM:.*]]:2 = map.iter.opaque
// CHECK: inputs (%[[A:.*]]: !mo.tensor<[2, 2], f32>)
// CHECK: outputs (%[[O0:.*]] : !mo.tensor<[2, 2], f32>, %[[O1:.*]] : !mo.tensor<[2, 2], f32>) {
// CHECK:   %[[EPI0:.*]] = map.iter.epilogue (%{{.*}} : f32, %{{.*}} : !map.index<2>) capturing {%{{.*}} = %[[O0]] : !mo.tensor<[2, 2], f32>} {
// CHECK:   %[[EPI1:.*]] = map.iter.epilogue (%{{.*}} : f32, %{{.*}} : !map.index<2>) capturing {%{{.*}} = %[[O1]] : !mo.tensor<[2, 2], f32>} {
// CHECK:   map.call.opaque @{{.*}}MapLowerTwoFusedOutputs[] inputs(%[[A]]) outputs(%[[EPI0]], %[[EPI1]]) kernelName = "map_lower_two_fused_outputs" : (!mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
mo.graph @two_fused(%a: !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %c:2 = mo.custom [symbol = "map_lower_two_fused_outputs", device = #M.device_ref<"cpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
  mo.output %c#0, %c#1 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// Two plain results: no epilogue, and the call writes to both raw outputs.

// CHECK-LABEL: mo.graph @two_plain
// CHECK: %[[MM:.*]]:2 = map.iter.opaque
// CHECK: inputs (%[[A:.*]]: !mo.tensor<[2, 2], f32>)
// CHECK: outputs (%[[O0:.*]] : !mo.tensor<[2, 2], f32>, %[[O1:.*]] : !mo.tensor<[2, 2], f32>) {
// CHECK-NOT: map.iter.epilogue
// CHECK:   map.call.opaque @{{.*}}MapLowerTwoPlainOutputs[] inputs(%[[A]]) outputs(%[[O0]], %[[O1]]) kernelName = "map_lower_two_plain_outputs" : (!mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
mo.graph @two_plain(%a: !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %c:2 = mo.custom [symbol = "map_lower_two_plain_outputs", device = #M.device_ref<"cpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
  mo.output %c#0, %c#1 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// Mixed results: the first output is fusible (store epilogue), the second is
// plain (raw). The call writes to the epilogue for the first and the raw block
// argument for the second, and there is exactly one epilogue.

// CHECK-LABEL: mo.graph @two_partial
// CHECK: %[[MM:.*]]:2 = map.iter.opaque
// CHECK: inputs (%[[A:.*]]: !mo.tensor<[2, 2], f32>)
// CHECK: outputs (%[[O0:.*]] : !mo.tensor<[2, 2], f32>, %[[O1:.*]] : !mo.tensor<[2, 2], f32>) {
// CHECK:   %[[EPI0:.*]] = map.iter.epilogue (%{{.*}} : f32, %{{.*}} : !map.index<2>) capturing {%{{.*}} = %[[O0]] : !mo.tensor<[2, 2], f32>} {
// CHECK-NOT: map.iter.epilogue
// CHECK:   map.call.opaque @{{.*}}MapLowerPartialFusedOutputs[] inputs(%[[A]]) outputs(%[[EPI0]], %[[O1]]) kernelName = "map_lower_partial_fused_outputs" : (!mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
mo.graph @two_partial(%a: !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %c:2 = mo.custom [symbol = "map_lower_partial_fused_outputs", device = #M.device_ref<"cpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
  mo.output %c#0, %c#1 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// With advanced fusion disabled the pass is a no-op: the custom ops survive and
// no map ops are produced.

// DISABLED-NOT: map.iter.opaque
// DISABLED: mo.custom
