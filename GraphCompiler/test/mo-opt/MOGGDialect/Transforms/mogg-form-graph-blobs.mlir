// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" \
// RUN:   --mo-to-map --map-to-mogg --mogg-conform-call-operands \
// RUN:   --mogg-form-graph-blobs | FileCheck %s

// A sequence of declarative routines becomes one routine whose body is a
// `mogg.graph_blob` holding theirs in order. It reads the sequence's input and
// writes every member's result.
// CHECK-LABEL: mo.graph @sequence
// CHECK:       %[[R:.*]]:3 = mogg.routine (%{{.*}} : !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
// CHECK-NEXT:    inputs (%[[A:[^ ,)]+]])
// CHECK-NEXT:    outputs (%[[B:[^ ,)]+]], %[[C:[^ ,)]+]], %[[D:[^ ,)]+]])
// CHECK-NEXT:    context (%[[CTX:[^ ,)]+]])
// CHECK:         mogg.graph_blob {
// CHECK:           mogg.tensor.create(%[[A]],
// CHECK:           mogg.tensor.create(%[[B]],
// CHECK:           mogg.call.execute_device @{{.*}}MapLowerDeclarative {{.*}}context(%[[CTX]]
// CHECK:           mogg.tensor.create(%[[B]],
// CHECK:           mogg.tensor.create(%[[C]],
// CHECK:           mogg.call.execute_device @{{.*}}MapLowerDeclarative
// CHECK:           mogg.tensor.create(%[[C]],
// CHECK:           mogg.tensor.create(%[[D]],
// CHECK:           mogg.call.execute_device @{{.*}}MapLowerDeclarative
// CHECK-NOT:   mogg.routine
// CHECK:       mo.output %[[R]]#2
mo.graph @sequence(%a: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %b = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %c = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%b) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %d = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%c) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %d : !mo.tensor<[2, 2], f32>
}

// An intermediate with a reader outside the sequence ends it.
// CHECK-LABEL: mo.graph @shared_intermediate
// CHECK-NOT:   mogg.graph_blob
// CHECK:       mo.output
mo.graph @shared_intermediate(%a: !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %b = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %c = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%b) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %b, %c : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// A kernel with `execute` between two declarative ones ends the sequence.
// CHECK-LABEL: mo.graph @execute_between
// CHECK-NOT:   mogg.graph_blob
// CHECK:       mo.output
mo.graph @execute_between(%a: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %b = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %c = mo.custom [symbol = "map_lower_one_plain_output", device = #M.device_ref<"gpu", 0>](%b) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %d = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%c) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %d : !mo.tensor<[2, 2], f32>
}

// Routines on different devices stay apart.
// CHECK-LABEL: mo.graph @two_devices
// CHECK-NOT:   mogg.graph_blob
// CHECK:       mo.output
mo.graph @two_devices(%a: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %b = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %c = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 1>](%b) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %c : !mo.tensor<[2, 2], f32>
}

// A member that reads both the previous member's result and the sequence's
// input shares the fused routine's one input for it.
// CHECK-LABEL: mo.graph @shared_input
// CHECK:       mogg.routine (%{{[^ ,]+}} : !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
// CHECK-NEXT:    inputs (%[[A:[^ ,)]+]])
// CHECK:         mogg.graph_blob {
// CHECK:           mogg.call.execute_device @{{.*}}MapLowerDeclarative{{ }}
// CHECK:           mogg.tensor.create(%[[A]],
// CHECK:           mogg.call.execute_device @{{.*}}MapLowerDeclarativeBinary
// CHECK-NOT:   mogg.routine
mo.graph @shared_input(%a: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %b = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %c = mo.custom [symbol = "map_lower_declarative_binary", device = #M.device_ref<"gpu", 0>](%b, %a) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %c : !mo.tensor<[2, 2], f32>
}

// A sequence in a `mo.parallel` body fuses too; `MOGGToMGP` later stamps the
// body out once per device.
// CHECK-LABEL: mo.graph @parallel
// CHECK:       mo.parallel
// CHECK:         mogg.routine
// CHECK:           mogg.graph_blob {
// CHECK-COUNT-2:     mogg.call.execute_device @{{.*}}MapLowerDeclarative
// CHECK-NOT:     mogg.routine
// CHECK:         mo.yield
mo.graph @parallel(%a: !mo.tensor<[2, 2], f32, gpu:0>, %b: !mo.tensor<[2, 2], f32, gpu:1>) -> (!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:1>) {
  %bundle = mo.tensor.bundle(%a, %b) : (!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:1>) -> (!mo.bundle<[!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:1>]>)
  %res = mo.parallel (%x) in (%bundle : !mo.bundle<[!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:1>]>) -> (!mo.bundle<[!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:1>]>) {
    %1 = mo.custom [symbol = "map_lower_declarative"](%x) : (!mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0>
    %2 = mo.custom [symbol = "map_lower_declarative"](%1) : (!mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0>
    mo.yield %2 : !mo.tensor<[2, 2], f32, gpu:0>
  }
  %r0, %r1 = mo.tensor.unbundle(%res) : (!mo.bundle<[!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:1>]>) -> (!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:1>)
  mo.output %r0, %r1 : !mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:1>
}
