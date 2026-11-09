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

// The graph's output doesn't end the blob: the caller gets it only when the
// whole execution ends. So an intermediate the graph returns still fuses with
// the member that reads it.
// CHECK-LABEL: mo.graph @shared_intermediate
// CHECK:       %[[R:.*]]:2 = mogg.routine
// CHECK:         mogg.graph_blob {
// CHECK:       mo.output %[[R]]#0, %[[R]]#1
mo.graph @shared_intermediate(%a: !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %b = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %c = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%b) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %b, %c : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// A result read by two members fans out inside one blob.
// CHECK-LABEL: mo.graph @fan_out
// CHECK:       %[[R:.*]]:3 = mogg.routine
// CHECK:         mogg.graph_blob {
// CHECK-COUNT-3:   mogg.call.execute_device
// CHECK:       mo.output %[[R]]#1, %[[R]]#2
mo.graph @fan_out(%a: !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %b = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %c = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%b) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %d = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%b) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %c, %d : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// Siblings that share no input join one blob too, when they run on one device.
// CHECK-LABEL: mo.graph @siblings
// CHECK:       %[[R:.*]]:2 = mogg.routine (%{{.*}}, %{{.*}} : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
// CHECK:         mogg.graph_blob {
// CHECK:       mo.output %[[R]]#0, %[[R]]#1
mo.graph @siblings(%a:!mo.tensor<[2, 2], f32>, %x: !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %b = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %c = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%x) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %b, %c : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// An elementwise loop between members that doesn't read the blob leaves it
// open.
// CHECK-LABEL: mo.graph @loop_beside
// CHECK:       map.iter.elementwise
// CHECK:       %[[R:.*]]:2 = mogg.routine
// CHECK:         mogg.graph_blob {
// CHECK:       mo.output
mo.graph @loop_beside(%a: !mo.tensor<[2, 2], f32>, %x: !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[4], f32>) {
  %b = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %r = mo.static.reshape(%x) : !mo.tensor<[2, 2], f32> -> !mo.tensor<[4], f32>
  %c = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%b) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %c, %r : !mo.tensor<[2, 2], f32>, !mo.tensor<[4], f32>
}

// A kernel with `execute` that reads a member ends the blob there, so that
// result doesn't wait for the independent member after it.
// CHECK-LABEL: mo.graph @read_outside
// CHECK-NOT:   mogg.graph_blob
// CHECK:       mo.output
mo.graph @read_outside(%a: !mo.tensor<[2, 2], f32>, %x: !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %b = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %t = mo.custom [symbol = "map_lower_one_plain_output", device = #M.device_ref<"gpu", 0>](%b) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %c = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%x) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %t, %c : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// The same with a host kernel reading a member: that member stays alone, and
// the independent members after it still fuse.
// CHECK-LABEL: mo.graph @host_read
// CHECK:       %[[B:.*]] = mogg.routine (%{{.*}} : !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
// CHECK:       mogg.routine (%[[B]] : !mo.tensor<[2, 2], f32>)
// CHECK:         mogg.call.opaque @{{.*}}MapLowerOnePlainOutput
// CHECK:       mogg.routine (%{{.*}} : !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
// CHECK:         mogg.graph_blob {
// CHECK:       mo.output
mo.graph @host_read(%a: !mo.tensor<[2, 2], f32>, %x: !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %b = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %t = mo.custom [symbol = "map_lower_one_plain_output", device = #M.device_ref<"cpu", 0>](%b) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %c = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%x) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %d = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%c) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %t, %d : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// A kernel with `execute` between two members ends the blob at the first, so
// neither side has two members.
// CHECK-LABEL: mo.graph @execute_between
// CHECK-NOT:   mogg.graph_blob
// CHECK:       mo.output
mo.graph @execute_between(%a: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %b = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %c = mo.custom [symbol = "map_lower_one_plain_output", device = #M.device_ref<"gpu", 0>](%b) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %d = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%c) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %d : !mo.tensor<[2, 2], f32>
}

// The same with the kernel before the member it reads: the pass walks the body
// in dataflow order, so the blob still ends at that member.
// CHECK-LABEL: mo.graph @execute_between_out_of_order
// CHECK-NOT:   mogg.graph_blob
// CHECK:       mo.output
mo.graph @execute_between_out_of_order(%a: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %c = mo.custom [symbol = "map_lower_one_plain_output", device = #M.device_ref<"gpu", 0>](%b) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %b = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %d = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%c) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %d : !mo.tensor<[2, 2], f32>
}

// A sequence computed only from constants runs once at init, so it stays out
// of blobs; the sequence on the graph's argument fuses.
// CHECK-LABEL: mo.graph @invariant
// CHECK:       mogg.routine (%{{.*}} : !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
// CHECK:       mogg.routine (%{{.*}} : !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
// CHECK:       mogg.routine (%{{.*}} : !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
// CHECK:         mogg.graph_blob {
// CHECK:       mo.output
mo.graph @invariant(%a: !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %w = mo.constant <#M.dense_array<1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00> : tensor<2x2xf32>> : !mo.tensor<[2, 2], f32>
  %x = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%w) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %y = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%x) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %b = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %c = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%b) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %y, %c : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// A graph region can use a value before its definition. The pass sorts the
// body first, so a constant defined after its readers still keeps them out;
// the sort moves them after the routines on the graph's argument.
// CHECK-LABEL: mo.graph @invariant_out_of_order
// CHECK:       mo.constant
// CHECK:       mogg.routine (%{{.*}} : !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
// CHECK:         mogg.graph_blob {
// CHECK:       mogg.routine (%{{.*}} : !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
// CHECK:       mogg.routine (%{{.*}} : !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
// CHECK:       mo.output
mo.graph @invariant_out_of_order(%a: !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %x = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%w) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %y = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%x) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %w = mo.constant <#M.dense_array<1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00> : tensor<2x2xf32>> : !mo.tensor<[2, 2], f32>
  %b = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %c = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%b) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %y, %c : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// Weights are invariant too, so the sequence on them stays out of blobs. The
// one on an aliased external constant fuses: something else can update it
// between executions.
// CHECK-LABEL: mo.graph @external
// CHECK:       mogg.routine (%{{.*}} : !mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0>
// CHECK:       mogg.routine (%{{.*}} : !mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0>
// CHECK:       mogg.routine (%{{.*}} : !mo.tensor<[2, 2], f32, gpu:0>) -> (!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:0>)
// CHECK:         mogg.graph_blob {
// CHECK:       mo.output
mo.graph @external() -> (!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:0>) {
  %w = mo.constant.external [name = "w", align = 16, device = #M.device_ref<"gpu", 0>] : !mo.tensor<[2, 2], f32, gpu:0>
  %u = mo.constant.external [name = "u", align = 16, device = #M.device_ref<"gpu", 0>, hasAlias = true] : !mo.tensor<[2, 2], f32, gpu:0>
  %x = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%w) : (!mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0>
  %y = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%x) : (!mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0>
  %b = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%u) : (!mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0>
  %c = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%b) : (!mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0>
  mo.output %y, %c : !mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:0>
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

// A member that reads both an earlier member's result and the blob's input
// shares the fused routine's one input for it.
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

// Routines in a `mo.parallel` body form their own blob; `MOGGToMGP` later
// stamps the body out once per device.
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

// Control flow ends the open blob, so the members before a `mo.if` and the
// independent ones after it form two blobs, not one. A branch's routines form
// their own: they run only when the condition picks that branch.
// CHECK-LABEL: mo.graph @control_flow
// CHECK:       mogg.routine (%{{.*}} : !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
// CHECK:         mogg.graph_blob {
// CHECK:       mo.if
// CHECK:         mogg.routine (%{{.*}} : !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
// CHECK:           mogg.graph_blob {
// CHECK:       mogg.routine (%{{.*}} : !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
// CHECK:         mogg.graph_blob {
// CHECK:       mo.output
mo.graph @control_flow(%a: !mo.tensor<[2, 2], f32>, %x: !mo.tensor<[2, 2], f32>, %y: !mo.tensor<[2, 2], f32>, %cond: !mo.tensor<[], bool>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %b = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %c = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%b) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %r = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.tensor<[2, 2], f32> {
    %1 = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%x) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
    %2 = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%1) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
    mo.yield %2 : !mo.tensor<[2, 2], f32>
  } else {
    mo.yield %x : !mo.tensor<[2, 2], f32>
  }
  %d = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%y) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %e = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%d) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %c, %r, %e : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// A routine inside a branch that reads a member is outside the blob, like any
// other reader there, so the blob ends at that member and the independent
// member after it stays alone.
// CHECK-LABEL: mo.graph @read_in_branch
// CHECK-NOT:   mogg.graph_blob
// CHECK:       mo.output
mo.graph @read_in_branch(%a: !mo.tensor<[2, 2], f32>, %x: !mo.tensor<[2, 2], f32>, %cond: !mo.tensor<[], bool>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %b = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %c = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%x) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %r = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.tensor<[2, 2], f32> {
    %1 = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%b) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
    mo.yield %1 : !mo.tensor<[2, 2], f32>
  } else {
    mo.yield %b : !mo.tensor<[2, 2], f32>
  }
  mo.output %c, %r : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// Members that update a buffer in place join a blob under its chain, as a KV
// cache store and the attention reading the cache do. The blob takes the buffer
// once, waits on what its members' chains wait on from outside, and returns one
// chain. The blob runs its members in order, so the join ordering the second
// store after the first goes away.
// CHECK-LABEL: mo.graph @buffer_chain
// CHECK:       %[[CH:.*]] = mo.chain.create()
// CHECK-NEXT:  %[[R:.*]]:4 = mogg.routine (%{{.*}}, %{{.*}} : !mo.tensor<[2, 2], f32>, !mo.buffer<[2, 2], f32>) chain(%[[CH]]) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.chain)
// CHECK:         mogg.graph_blob {
// CHECK:           mogg.call.execute_device @{{.*}}MapLowerDeclarativeInplace inputs({{.*}} : !mogg.tensor, !mogg.buffer)
// CHECK:           mogg.call.execute_device @{{.*}}MapLowerDeclarativeInplace inputs({{.*}} : !mogg.tensor, !mogg.buffer)
// CHECK:           mogg.call.execute_device @{{.*}}MapLowerDeclarative inputs
// CHECK-NOT:   mo.chain.create
// CHECK:       mo.output %[[R]]#2
mo.graph @buffer_chain(%a: !mo.tensor<[2, 2], f32>, %cache: !mo.buffer<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %ch0 = mo.chain.create()
  %b, %ch1 = mo.custom [symbol = "map_lower_declarative_inplace", device = #M.device_ref<"gpu", 0>](%a, %cache, %ch0) : (!mo.tensor<[2, 2], f32>, !mo.buffer<[2, 2], f32>, !mo.chain) -> (!mo.tensor<[2, 2], f32>, !mo.chain)
  %ch2 = mo.chain.create(%ch1, %ch0) : !mo.chain, !mo.chain
  %c, %ch3 = mo.custom [symbol = "map_lower_declarative_inplace", device = #M.device_ref<"gpu", 0>](%b, %cache, %ch2) : (!mo.tensor<[2, 2], f32>, !mo.buffer<[2, 2], f32>, !mo.chain) -> (!mo.tensor<[2, 2], f32>, !mo.chain)
  %d = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%c) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %d : !mo.tensor<[2, 2], f32>
}

// A kernel with `execute` that updates the buffer between two members reads
// the first one's chain, so the blob ends there and neither side has two
// members.
// CHECK-LABEL: mo.graph @buffer_write_between
// CHECK-NOT:   mogg.graph_blob
// CHECK:       mo.output
mo.graph @buffer_write_between(%a: !mo.tensor<[2, 2], f32>, %cache: !mo.buffer<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.chain) {
  %ch0 = mo.chain.create()
  %b, %ch1 = mo.custom [symbol = "map_lower_declarative_inplace", device = #M.device_ref<"gpu", 0>](%a, %cache, %ch0) : (!mo.tensor<[2, 2], f32>, !mo.buffer<[2, 2], f32>, !mo.chain) -> (!mo.tensor<[2, 2], f32>, !mo.chain)
  %ch2 = mo.custom [symbol = "map_lower_inplace_copy", device = #M.device_ref<"gpu", 0>](%cache, %b, %ch1) : (!mo.buffer<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.chain) -> !mo.chain
  %c, %ch3 = mo.custom [symbol = "map_lower_declarative_inplace", device = #M.device_ref<"gpu", 0>](%b, %cache, %ch2) : (!mo.tensor<[2, 2], f32>, !mo.buffer<[2, 2], f32>, !mo.chain) -> (!mo.tensor<[2, 2], f32>, !mo.chain)
  mo.output %c, %ch3 : !mo.tensor<[2, 2], f32>, !mo.chain
}

// When the members wait on several chains from outside the blob, the blob waits
// on a new join of them, placed before it. Here the join between the members
// also orders the second store after an independent write to another buffer.
// CHECK-LABEL: mo.graph @buffer_chain_outside_waits
// CHECK:       %[[CH:.*]] = mo.chain.create()
// CHECK:       %[[COPY:.*]] = mogg.routine
// CHECK:       %[[WAIT:.*]] = mo.chain.create(%[[CH]], %[[COPY]]
// CHECK-NEXT:  %[[R:.*]]:3 = mogg.routine ({{.*}}) chain(%[[WAIT]])
// CHECK:         mogg.graph_blob {
// CHECK:       mo.output %[[R]]#1
mo.graph @buffer_chain_outside_waits(%a: !mo.tensor<[2, 2], f32>, %cache: !mo.buffer<[2, 2], f32>, %other: !mo.buffer<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %ch0 = mo.chain.create()
  %b, %ch1 = mo.custom [symbol = "map_lower_declarative_inplace", device = #M.device_ref<"gpu", 0>](%a, %cache, %ch0) : (!mo.tensor<[2, 2], f32>, !mo.buffer<[2, 2], f32>, !mo.chain) -> (!mo.tensor<[2, 2], f32>, !mo.chain)
  %chx = mo.custom [symbol = "map_lower_inplace_copy", device = #M.device_ref<"gpu", 0>](%other, %a, %ch0) : (!mo.buffer<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.chain) -> !mo.chain
  %ch2 = mo.chain.create(%ch1, %chx) : !mo.chain, !mo.chain
  %c, %ch3 = mo.custom [symbol = "map_lower_declarative_inplace", device = #M.device_ref<"gpu", 0>](%b, %cache, %ch2) : (!mo.tensor<[2, 2], f32>, !mo.buffer<[2, 2], f32>, !mo.chain) -> (!mo.tensor<[2, 2], f32>, !mo.chain)
  mo.output %c : !mo.tensor<[2, 2], f32>
}

// A member's chain that the graph returns becomes the blob's chain, and a join
// between the members that the graph also returns reads the blob's chain. The
// join stays where it was, before the blob: a graph region allows that.
// CHECK-LABEL: mo.graph @buffer_chain_returned
// CHECK:       %[[CH:.*]] = mo.chain.create()
// CHECK-NEXT:  %[[JOIN:.*]] = mo.chain.create(%[[R:[0-9]+]]#2, %[[CH]]
// CHECK-NEXT:  %[[R]]:3 = mogg.routine ({{.*}}) chain(%[[CH]])
// CHECK:         mogg.graph_blob {
// CHECK:       mo.output %[[R]]#1, %[[JOIN]], %[[R]]#2
mo.graph @buffer_chain_returned(%a: !mo.tensor<[2, 2], f32>, %cache: !mo.buffer<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.chain, !mo.chain) {
  %ch0 = mo.chain.create()
  %b, %ch1 = mo.custom [symbol = "map_lower_declarative_inplace", device = #M.device_ref<"gpu", 0>](%a, %cache, %ch0) : (!mo.tensor<[2, 2], f32>, !mo.buffer<[2, 2], f32>, !mo.chain) -> (!mo.tensor<[2, 2], f32>, !mo.chain)
  %ch2 = mo.chain.create(%ch1, %ch0) : !mo.chain, !mo.chain
  %c, %ch3 = mo.custom [symbol = "map_lower_declarative_inplace", device = #M.device_ref<"gpu", 0>](%b, %cache, %ch2) : (!mo.tensor<[2, 2], f32>, !mo.buffer<[2, 2], f32>, !mo.chain) -> (!mo.tensor<[2, 2], f32>, !mo.chain)
  mo.output %c, %ch2, %ch3 : !mo.tensor<[2, 2], f32>, !mo.chain, !mo.chain
}
