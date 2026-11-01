// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-allocate-accumulating-buffers --map-to-mogg --mogg-pipeline | FileCheck %s

// Outlined stubs are module-level siblings of the graphs, so `mogg-pipeline`
// canonicalizes and CSEs the whole module, not just the graphs. The input and
// output have the same dynamic shape, so `map-to-mogg` builds two identical
// `mogg.shape.create` ops; only one survives in the stub.
// CHECK-LABEL: mogg.func
// CHECK:       mogg.shape.create
// CHECK-NOT:   mogg.shape.create
// CHECK:       mogg.output
mo.graph @dyn(%a: !mo.tensor<[batch, 4], f32>) -> !mo.tensor<[batch, 4], f32> {
  %c = mo.custom [symbol = "map_lower_one_plain_output", device = #M.device_ref<"cpu", 0>](%a) : (!mo.tensor<[batch, 4], f32>) -> !mo.tensor<[batch, 4], f32>
  mo.output %c : !mo.tensor<[batch, 4], f32>
}
