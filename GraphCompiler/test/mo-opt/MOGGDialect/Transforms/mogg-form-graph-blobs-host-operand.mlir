// RUN: mo-opt %s --pass-pipeline='builtin.module(mogg-import-packages{mojo-pkg-paths=%kernel_api_test_lib}, mo-to-map, map-outline-structured-kernel-bodies, mo.graph(map-livm, map.iter.opaque(canonicalize)), canonicalize, cse, map-to-mogg, mogg-conform-call-operands, mogg-form-graph-blobs{max-calls=100})' \
// RUN:   | FileCheck %s

// A slice's bounds are read on the host, so a launch can't take the view and
// the loop that reads it stays out of the blob. The add and the second
// declarative call still form one.
// CHECK-LABEL: mo.graph @slice_capture
// CHECK:       mogg.routine
// CHECK-NOT:     mogg.graph_blob
// CHECK:         mogg.call.opaque
// CHECK:       mogg.routine
// CHECK-NOT:     mogg.graph_blob
// CHECK:         mogg.tensor.create.slice
// CHECK:         mogg.call.foreach
// CHECK:       mogg.routine
// CHECK-NEXT:    inputs
// CHECK-NEXT:    outputs
// CHECK-NEXT:    context
// CHECK-NEXT:    mogg.graph_blob {
// CHECK:           mogg.call.foreach
// CHECK:           mogg.call.opaque
// CHECK:       mo.output
mo.graph @slice_capture(%a: !mo.tensor<[2, 4], f32, gpu:0>, %bias: !mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0> {
  %b = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%a) : (!mo.tensor<[2, 4], f32, gpu:0>) -> !mo.tensor<[2, 4], f32, gpu:0>
  %start = mo.constant <#M.dense_array<0, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop = mo.constant <#M.dense_array<2, 4> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %s = mo.slice(%b, %start, %stop, %step) : (!mo.tensor<[2, 4], f32, gpu:0>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32, gpu:0>
  %c = mo.add(%s, %bias) : !mo.tensor<[2, 2], f32, gpu:0>
  %d = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%c) : (!mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0>
  mo.output %d : !mo.tensor<[2, 2], f32, gpu:0>
}
