// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" \
// RUN:   --mo-to-map --map-fuse-kernels --map-outline-structured-kernel-bodies \
// RUN:   --canonicalize --cse --map-to-mogg --mogg-conform-call-operands \
// RUN:   --mogg-form-graph-blobs="max-calls=100" | FileCheck %s

// A generated elementwise kernel between two declarative ones joins their
// blob, which runs its loop between the two calls.
// CHECK-LABEL: mo.graph @declarative_elementwise_declarative
// CHECK:       mogg.routine
// CHECK:         mogg.graph_blob {
// CHECK:           mogg.call.opaque @{{.*}}MapLowerDeclarative
// CHECK:           mogg.call.foreach @builtin_primitives.primitives._foreach
// CHECK:           mogg.call.opaque @{{.*}}MapLowerDeclarative
// CHECK-NOT:   mogg.routine
// CHECK:       mo.output
mo.graph @declarative_elementwise_declarative(%a: !mo.tensor<[2, 2], f32, gpu:0>, %bias: !mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0> {
  %b = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%a) : (!mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0>
  %c = mo.add(%b, %bias) : !mo.tensor<[2, 2], f32, gpu:0>
  %d = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%c) : (!mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0>
  mo.output %d : !mo.tensor<[2, 2], f32, gpu:0>
}

// Generated kernels form a blob with no declarative call: the blob's launch
// comes from its device.
// CHECK-LABEL: mo.graph @elementwise_only
// CHECK:       mogg.routine
// CHECK:         mogg.graph_blob {
// CHECK-COUNT-2:   mogg.call.foreach
// CHECK:       mo.output
mo.graph @elementwise_only(%a: !mo.tensor<[2, 2], f32, gpu:0>, %b: !mo.tensor<[2, 2], f32, gpu:0>, %x: !mo.tensor<[2, 2], f32, gpu:0>, %y: !mo.tensor<[2, 2], f32, gpu:0>) -> (!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:0>) {
  %c = mo.add(%a, %b) : !mo.tensor<[2, 2], f32, gpu:0>
  %d = mo.mul(%x, %y) : !mo.tensor<[2, 2], f32, gpu:0>
  mo.output %c, %d : !mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:0>
}

// The host has no blob launch, so generated kernels there stay apart.
// CHECK-LABEL: mo.graph @elementwise_on_host
// CHECK-NOT:   mogg.graph_blob
// CHECK:       mo.output
mo.graph @elementwise_on_host(%a: !mo.tensor<[2, 2], f32>, %b: !mo.tensor<[2, 2], f32>, %x: !mo.tensor<[2, 2], f32>, %y: !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %c = mo.add(%a, %b) : !mo.tensor<[2, 2], f32>
  %d = mo.mul(%x, %y) : !mo.tensor<[2, 2], f32>
  mo.output %c, %d : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}
