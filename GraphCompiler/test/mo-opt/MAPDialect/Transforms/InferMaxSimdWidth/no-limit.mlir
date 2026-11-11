// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-fuse-kernels \
// RUN:   --map-outline-structured-kernel-bodies --map-livm \
// RUN:   --map-infer-max-simd-width | FileCheck %s

// No op limits the SIMD width yet, so no loop gets a `max_simd_width`. Views on
// loads are hoisted out of the body before the pass runs, and a slice select
// keeps the loop's innermost coordinate, so neither is index computation.

// CHECK-LABEL: mo.graph @elementwise
// CHECK:       map.iter.elementwise.body<[4, 16]>
// CHECK-NOT:   max_simd_width
// CHECK:       mo.output
mo.graph @elementwise(%a: !mo.tensor<[4, 16], f32>, %b: !mo.tensor<[4, 16], f32>) -> !mo.tensor<[4, 16], f32> {
  %x = mo.add(%a, %b) : !mo.tensor<[4, 16], f32>
  %y = mo.relu(%x) : !mo.tensor<[4, 16], f32>
  mo.output %y : !mo.tensor<[4, 16], f32>
}

// CHECK-LABEL: mo.graph @views
// CHECK:       map.iter.elementwise.body<[16, 4]>
// CHECK-NOT:   max_simd_width
// CHECK:       mo.output
mo.graph @views(%a: !mo.tensor<[4, 16], f32>, %b: !mo.tensor<[4], f32>) -> !mo.tensor<[16, 4], f32> {
  %perm = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %t = mo.transpose(%a, %perm) : (!mo.tensor<[4, 16], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[16, 4], f32>
  %bb = mo.static.broadcast_to(%b) : !mo.tensor<[4], f32> -> !mo.tensor<[16, 4], f32>
  %y = mo.add(%t, %bb) : !mo.tensor<[16, 4], f32>
  mo.output %y : !mo.tensor<[16, 4], f32>
}

// The consumer of a window fuses as a slice select in the producer's loop.
// CHECK-LABEL: mo.graph @slice_select
// CHECK:       map.iter.elementwise.body<[4, 8, 16]>
// CHECK-NOT:   max_simd_width
// CHECK:       map.index.slice.select
// CHECK-NOT:   max_simd_width
// CHECK:       mo.output
mo.graph @slice_select(%a: !mo.tensor<[4, 8, 16], f32>, %b: !mo.tensor<[4, 8, 16], f32>) -> (!mo.tensor<[4, 8, 16], f32>, !mo.tensor<[4, 4, 16], f32>) {
  %starts = mo.constant <#M.dense_array<0, 4, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %stops = mo.constant <#M.dense_array<4, 8, 16> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %steps = mo.constant <#M.dense_array<1, 1, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %x = mo.add(%a, %b) : !mo.tensor<[4, 8, 16], f32>
  %w = mo.slice(%x, %starts, %stops, %steps) : (!mo.tensor<[4, 8, 16], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 4, 16], f32>
  %y = mo.relu(%w) : !mo.tensor<[4, 4, 16], f32>
  mo.output %x, %y : !mo.tensor<[4, 8, 16], f32>, !mo.tensor<[4, 4, 16], f32>
}
