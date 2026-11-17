// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" \
// RUN:   --mo-to-map --map-to-mogg --mogg-conform-call-operands \
// RUN:   --mogg-form-graph-blobs="max-calls=2" | FileCheck %s
// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" \
// RUN:   --mo-to-map --map-to-mogg --mogg-conform-call-operands \
// RUN:   --mogg-form-graph-blobs="max-calls=0" | FileCheck %s --check-prefix=OFF

// With `max-calls` at 0, no routines fuse.
// OFF-NOT: mogg.graph_blob

// A blob that holds `max-calls` routines, one call each, ends before the next,
// which starts a new blob.
// CHECK-LABEL: mo.graph @sequence
// CHECK:       %[[R0:.*]]:2 = mogg.routine (%{{.*}} : !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
// CHECK:         mogg.graph_blob {
// CHECK-COUNT-2:   mogg.call.execute_device
// CHECK:       %[[R1:.*]]:2 = mogg.routine (%[[R0]]#1 : !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
// CHECK:         mogg.graph_blob {
// CHECK-COUNT-2:   mogg.call.execute_device
// CHECK-NOT:   mogg.routine
// CHECK:       mo.output %[[R1]]#1
mo.graph @sequence(%a: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %b = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %c = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%b) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %d = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%c) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %e = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%d) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %e : !mo.tensor<[2, 2], f32>
}

// Each device's blob counts only its own routines, so interleaved devices with
// two routines each still form a blob of two on each device.
// CHECK-LABEL: mo.graph @interleaved_devices
// CHECK:       %[[R0:.*]]:2 = mogg.routine
// CHECK:         attributes {device = #M.device_ref<"gpu", 0>
// CHECK-NEXT:      mogg.graph_blob {
// CHECK-COUNT-2:   mogg.call.execute_device
// CHECK:       %[[R1:.*]]:2 = mogg.routine
// CHECK:         attributes {device = #M.device_ref<"gpu", 1>
// CHECK-NEXT:      mogg.graph_blob {
// CHECK-COUNT-2:   mogg.call.execute_device
// CHECK-NOT:   mogg.routine
// CHECK:       mo.output %[[R0]]#1, %[[R1]]#1
mo.graph @interleaved_devices(%a: !mo.tensor<[2, 2], f32>, %x: !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %b = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %y = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 1>](%x) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %c = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%b) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %z = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 1>](%y) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %c, %z : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}
