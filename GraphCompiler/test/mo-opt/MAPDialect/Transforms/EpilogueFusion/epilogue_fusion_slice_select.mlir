// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-horizontal --map-fuse-epilogue | FileCheck %s

// Horizontal fusion moves the window's store into the concat's epilogue, inside
// a `map.index.slice.select`. The window's own reader is then a single-use
// consumer of an epilogue result, the edge epilogue fusion takes -- but it
// would rebuild the epilogue's stores whole, and the one in the select writes
// only the rows the slice selects. So the relu stays a kernel of its own,
// reading the window.

// CHECK-LABEL: mo.graph @concat_window_relu
// CHECK:       %[[CAT:[a-z0-9]+]]:2 = map.iter.opaque
// CHECK:       map.iter.epilogue
// CHECK-NEXT:  map.store %{{.*}} : !mo.tensor<[4, 9, 16], f32>
// CHECK-NEXT:  map.index.slice.select{{.*}} -> (%[[IOUT:[a-z0-9]+]]) : !map.index<3> {
// CHECK-NEXT:  map.store %{{.*}}[%[[IOUT]]] = %{{.*}} : !mo.tensor<[4, 8, 16], f32>
// CHECK-NEXT:  }
// CHECK-NEXT:  }
// CHECK-NEXT:  map.call.opaque @{{.*}}.FusedVariadicInputs
// CHECK:       %[[RELU:[a-z0-9]+]] = map.iter.elementwise<[4, 8, 16]>
// CHECK-NEXT:  inputs (%[[CAT]]#1: !mo.tensor<[4, 8, 16], f32>)
// CHECK:       map.call.elementwise @{{.*}}.ReLU
// CHECK:       mo.output %[[CAT]]#0, %[[RELU]]
mo.graph @concat_window_relu(
  %a: !mo.tensor<[4, 8, 16], f32>,
  %b: !mo.tensor<[4, 1, 16], f32>
) -> (!mo.tensor<[4, 9, 16], f32>, !mo.tensor<[4, 8, 16], f32>) {
  %starts = mo.constant <#M.dense_array<0, 1, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %stops = mo.constant <#M.dense_array<4, 9, 16> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %steps = mo.constant <#M.dense_array<1, 1, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %concat = mo.custom [symbol = "fused_variadic_inputs", device = #M.device_ref<"cpu", 0>, parameters = {axis = 1}](%a, %b) : (!mo.tensor<[4, 8, 16], f32>, !mo.tensor<[4, 1, 16], f32>) -> !mo.tensor<[4, 9, 16], f32>
  %window = mo.slice(%concat, %starts, %stops, %steps) : (!mo.tensor<[4, 9, 16], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 8, 16], f32>
  %relu = mo.relu(%window) : !mo.tensor<[4, 8, 16], f32>
  mo.output %concat, %relu : !mo.tensor<[4, 9, 16], f32>, !mo.tensor<[4, 8, 16], f32>
}
