// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map | FileCheck %s
// RUN: mo-opt %s --mo-to-map | FileCheck %s --check-prefix=DISABLED

// CHECK-LABEL: mo.graph @gather
// CHECK: %[[G:.*]] = map.iter.opaque
// CHECK: inputs (%[[DATA:.*]]: !mo.tensor<[3, 3], f32>, %[[INDICES:.*]]: !mo.tensor<[2, 2], si32>)
// CHECK: outputs (%[[OUT:.*]] : !mo.tensor<[2, 2, 3], f32>) {
// CHECK:   %[[PD:.*]] = map.iter.prologue (%{{.*}} : !map.index<2>) capturing {%[[CD:.*]] = %[[DATA]] : !mo.tensor<[3, 3], f32>} {
// CHECK:     %{{.*}} = map.load %[[CD]][%{{.*}}] : !mo.tensor<[3, 3], f32>
// CHECK:   } : !mo.tensor<[3, 3], f32>
// CHECK:   %[[PI:.*]] = map.iter.prologue (%{{.*}} : !map.index<2>) capturing {%[[CI:.*]] = %[[INDICES]] : !mo.tensor<[2, 2], si32>} {
// CHECK:     %{{.*}} = map.load %[[CI]][%{{.*}}] : !mo.tensor<[2, 2], si32>
// CHECK:   } : !mo.tensor<[2, 2], si32>
// CHECK:   %[[EPI:.*]] = map.iter.epilogue (%{{.*}} : f32, %{{.*}} : !map.index<3>) capturing {%[[CO:.*]] = %[[OUT]] : !mo.tensor<[2, 2, 3], f32>} {
// CHECK:     map.store %[[CO]][%{{.*}}] = %{{.*}} : !mo.tensor<[2, 2, 3], f32>
// CHECK:   } : !mo.tensor<[2, 2, 3], f32>
// CHECK:   map.call.opaque @{{.*}}Gather[axis = 0 : index] inputs(%[[PD]], %[[PI]]) outputs(%[[EPI]]) kernelName = "mo.gather" : (!mo.tensor<[3, 3], f32>, !mo.tensor<[2, 2], si32>) -> !mo.tensor<[2, 2, 3], f32>
mo.graph @gather(%data: !mo.tensor<[3, 3], f32>, %indices: !mo.tensor<[2, 2], si32>) -> !mo.tensor<[2, 2, 3], f32> {
  %out = mo.gather(%data, %indices) [axis = 0] : (!mo.tensor<[3, 3], f32>, !mo.tensor<[2, 2], si32>) -> !mo.tensor<[2, 2, 3], f32>
  mo.output %out : !mo.tensor<[2, 2, 3], f32>
}

// CHECK-LABEL: mo.graph @two_fused
// CHECK: %[[MM:.*]] = map.iter.opaque
// CHECK: inputs (%[[A:.*]]: !mo.tensor<[2, 2], f32>, %[[B:.*]]: !mo.tensor<[2, 2], f32>)
// CHECK: outputs (%[[O:.*]] : !mo.tensor<[2, 2], f32>) {
// CHECK:   %[[PA:.*]] = map.iter.prologue (%{{.*}} : !map.index<2>) capturing {%[[CA:.*]] = %[[A]] : !mo.tensor<[2, 2], f32>} {
// CHECK:     %{{.*}} = map.load %[[CA]][%{{.*}}] : !mo.tensor<[2, 2], f32>
// CHECK:   } : !mo.tensor<[2, 2], f32>
// CHECK:   %[[PB:.*]] = map.iter.prologue (%{{.*}} : !map.index<2>) capturing {%[[CB:.*]] = %[[B]] : !mo.tensor<[2, 2], f32>} {
// CHECK:     %{{.*}} = map.load %[[CB]][%{{.*}}] : !mo.tensor<[2, 2], f32>
// CHECK:   } : !mo.tensor<[2, 2], f32>
// CHECK:   map.call.opaque @{{.*}}MapLowerTwoFusedInputs[] inputs(%[[PA]], %[[PB]]) outputs(%[[O]]) kernelName = "map_lower_two_fused_inputs" : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
mo.graph @two_fused(%a: !mo.tensor<[2, 2], f32>, %b: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %c = mo.custom [symbol = "map_lower_two_fused_inputs", device = #M.device_ref<"cpu", 0>](%a, %b) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %c : !mo.tensor<[2, 2], f32>
}

// CHECK-LABEL: mo.graph @partial_fused
// CHECK: %[[MM:.*]] = map.iter.opaque
// CHECK: inputs (%[[A:.*]]: !mo.tensor<[2, 2], f32>, %[[B:.*]]: !mo.tensor<[2, 2], f32>)
// CHECK: outputs (%[[O:.*]] : !mo.tensor<[2, 2], f32>) {
// CHECK:   %[[PA:.*]] = map.iter.prologue (%{{.*}} : !map.index<2>) capturing {%[[CA:.*]] = %[[A]] : !mo.tensor<[2, 2], f32>} {
// CHECK:     %{{.*}} = map.load %[[CA]][%{{.*}}] : !mo.tensor<[2, 2], f32>
// CHECK:   } : !mo.tensor<[2, 2], f32>
// CHECK-NOT: map.iter.prologue
// CHECK:   map.call.opaque @{{.*}}MapLowerPartialFusedInputs[] inputs(%[[PA]], %[[B]]) outputs(%[[O]]) kernelName = "map_lower_partial_fused_inputs" : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
mo.graph @partial_fused(%a: !mo.tensor<[2, 2], f32>, %b: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %c = mo.custom [symbol = "map_lower_partial_fused_inputs", device = #M.device_ref<"cpu", 0>](%a, %b) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %c : !mo.tensor<[2, 2], f32>
}

// With advanced fusion disabled the pass is a no-op: the custom ops survive
// and no map ops are produced.

// DISABLED-NOT: map.iter.opaque
// DISABLED: mo.custom
