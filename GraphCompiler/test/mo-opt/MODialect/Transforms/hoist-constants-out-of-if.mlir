// RUN: mo-opt --pass-pipeline="builtin.module(mo.graph(mo-hoist-constants-out-of-if))" --split-input-file %s | FileCheck %s

// Both branches transform the same weight the same way: one transform moves
// in front of the if, and both branches use it.
// CHECK-LABEL: mo.graph @same_transform
mo.graph @same_transform(%x: !mo.tensor<[4], f32>, %cond: !mo.tensor<[], bool>) -> (!mo.tensor<[4], f32>) {
  // CHECK: %[[W:.+]] = mo.constant.external [name = "w"
  // CHECK: %[[WN:.+]] = mo.negative(%[[W]])
  // CHECK: mo.if
  // CHECK-NOT: mo.negative
  // CHECK: mo.add(%{{.+}}, %[[WN]])
  // CHECK: } else {
  // CHECK-NOT: mo.negative
  // CHECK: mo.mul(%{{.+}}, %[[WN]])
  %w = mo.constant.external [name = "w", align = 1, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[4], f32>
  %r = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.tensor<[4], f32> {
    %wn = mo.negative(%w) : !mo.tensor<[4], f32>
    %a = mo.add(%x, %wn) : !mo.tensor<[4], f32>
    mo.yield %a : !mo.tensor<[4], f32>
  } else {
    %wn = mo.negative(%w) : !mo.tensor<[4], f32>
    %b = mo.mul(%x, %wn) : !mo.tensor<[4], f32>
    mo.yield %b : !mo.tensor<[4], f32>
  }
  mo.output %r : !mo.tensor<[4], f32>
}

// -----

// Each branch declares the weight itself: the declarations match, so one
// declaration and one transform move out.
// CHECK-LABEL: mo.graph @weight_declared_in_each_branch
mo.graph @weight_declared_in_each_branch(%x: !mo.tensor<[4], f32>, %cond: !mo.tensor<[], bool>) -> (!mo.tensor<[4], f32>) {
  // CHECK: %[[W:.+]] = mo.constant.external [name = "w"
  // CHECK: %[[WN:.+]] = mo.negative(%[[W]])
  // CHECK: mo.if
  // CHECK-NEXT: mo.add(%{{.+}}, %[[WN]])
  // CHECK: } else {
  // CHECK-NEXT: mo.mul(%{{.+}}, %[[WN]])
  %r = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.tensor<[4], f32> {
    %w = mo.constant.external [name = "w", align = 1, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[4], f32>
    %wn = mo.negative(%w) : !mo.tensor<[4], f32>
    %a = mo.add(%x, %wn) : !mo.tensor<[4], f32>
    mo.yield %a : !mo.tensor<[4], f32>
  } else {
    %w = mo.constant.external [name = "w", align = 1, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[4], f32>
    %wn = mo.negative(%w) : !mo.tensor<[4], f32>
    %b = mo.mul(%x, %wn) : !mo.tensor<[4], f32>
    mo.yield %b : !mo.tensor<[4], f32>
  }
  mo.output %r : !mo.tensor<[4], f32>
}

// -----

// Only one branch transforms the weight: the transform stays in that branch.
// CHECK-LABEL: mo.graph @one_branch
mo.graph @one_branch(%x: !mo.tensor<[4], f32>, %cond: !mo.tensor<[], bool>) -> (!mo.tensor<[4], f32>) {
  // CHECK: mo.if
  // CHECK-NEXT: mo.negative
  // CHECK: } else {
  %w = mo.constant.external [name = "w", align = 1, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[4], f32>
  %r = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.tensor<[4], f32> {
    %wn = mo.negative(%w) : !mo.tensor<[4], f32>
    %a = mo.add(%x, %wn) : !mo.tensor<[4], f32>
    mo.yield %a : !mo.tensor<[4], f32>
  } else {
    mo.yield %x : !mo.tensor<[4], f32>
  }
  mo.output %r : !mo.tensor<[4], f32>
}

// -----

// The branches transform the weight differently: each branch keeps its own
// transform.
// CHECK-LABEL: mo.graph @different_transforms
mo.graph @different_transforms(%x: !mo.tensor<[4], f32>, %cond: !mo.tensor<[], bool>) -> (!mo.tensor<[4], f32>) {
  // CHECK: mo.if
  // CHECK-NEXT: mo.negative
  // CHECK: } else {
  // CHECK-NEXT: mo.relu
  %w = mo.constant.external [name = "w", align = 1, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[4], f32>
  %r = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.tensor<[4], f32> {
    %wn = mo.negative(%w) : !mo.tensor<[4], f32>
    %a = mo.add(%x, %wn) : !mo.tensor<[4], f32>
    mo.yield %a : !mo.tensor<[4], f32>
  } else {
    %wr = mo.relu(%w) : !mo.tensor<[4], f32>
    %b = mo.add(%x, %wr) : !mo.tensor<[4], f32>
    mo.yield %b : !mo.tensor<[4], f32>
  }
  mo.output %r : !mo.tensor<[4], f32>
}

// -----

// The branches share the first transform of a chain: it moves out, and the
// transform that only one branch applies to its result stays.
// CHECK-LABEL: mo.graph @shared_prefix
mo.graph @shared_prefix(%x: !mo.tensor<[4], f32>, %cond: !mo.tensor<[], bool>) -> (!mo.tensor<[4], f32>) {
  // CHECK: %[[WN:.+]] = mo.negative
  // CHECK: mo.if
  // CHECK-NEXT: mo.add(%{{.+}}, %[[WN]])
  // CHECK: } else {
  // CHECK-NEXT: %[[WR:.+]] = mo.relu(%[[WN]])
  // CHECK-NEXT: mo.add(%{{.+}}, %[[WR]])
  %w = mo.constant.external [name = "w", align = 1, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[4], f32>
  %r = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.tensor<[4], f32> {
    %wn = mo.negative(%w) : !mo.tensor<[4], f32>
    %a = mo.add(%x, %wn) : !mo.tensor<[4], f32>
    mo.yield %a : !mo.tensor<[4], f32>
  } else {
    %wn = mo.negative(%w) : !mo.tensor<[4], f32>
    %wr = mo.relu(%wn) : !mo.tensor<[4], f32>
    %b = mo.add(%x, %wr) : !mo.tensor<[4], f32>
    mo.yield %b : !mo.tensor<[4], f32>
  }
  mo.output %r : !mo.tensor<[4], f32>
}

// -----

// Ops on a graph input stay in their branch, even when both branches have
// them.
// CHECK-LABEL: mo.graph @input_derived
mo.graph @input_derived(%x: !mo.tensor<[4], f32>, %cond: !mo.tensor<[], bool>) -> (!mo.tensor<[4], f32>) {
  // CHECK: mo.if
  // CHECK-NEXT: mo.negative
  // CHECK: } else {
  // CHECK-NEXT: mo.negative
  %r = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.tensor<[4], f32> {
    %xn = mo.negative(%x) : !mo.tensor<[4], f32>
    mo.yield %xn : !mo.tensor<[4], f32>
  } else {
    %xn = mo.negative(%x) : !mo.tensor<[4], f32>
    %xr = mo.relu(%xn) : !mo.tensor<[4], f32>
    mo.yield %xr : !mo.tensor<[4], f32>
  }
  mo.output %r : !mo.tensor<[4], f32>
}

// -----

// A transform that every branch of a chain of ifs computes moves out of all
// of them.
// CHECK-LABEL: mo.graph @nested
mo.graph @nested(%x: !mo.tensor<[4], f32>, %c0: !mo.tensor<[], bool>, %c1: !mo.tensor<[], bool>) -> (!mo.tensor<[4], f32>) {
  // CHECK: %[[W:.+]] = mo.constant.external [name = "w"
  // CHECK: %[[WN:.+]] = mo.negative(%[[W]])
  // CHECK: mo.if
  // CHECK-NEXT: mo.add(%{{.+}}, %[[WN]])
  // CHECK: } else {
  // CHECK-NEXT: mo.if
  // CHECK-NEXT: mo.mul(%{{.+}}, %[[WN]])
  // CHECK: } else {
  // CHECK-NEXT: mo.max(%{{.+}}, %[[WN]])
  %w = mo.constant.external [name = "w", align = 1, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[4], f32>
  %r = mo.if(%c0 : !mo.tensor<[], bool>) -> !mo.tensor<[4], f32> {
    %wn = mo.negative(%w) : !mo.tensor<[4], f32>
    %a = mo.add(%x, %wn) : !mo.tensor<[4], f32>
    mo.yield %a : !mo.tensor<[4], f32>
  } else {
    %inner = mo.if(%c1 : !mo.tensor<[], bool>) -> !mo.tensor<[4], f32> {
      %wn = mo.negative(%w) : !mo.tensor<[4], f32>
      %b = mo.mul(%x, %wn) : !mo.tensor<[4], f32>
      mo.yield %b : !mo.tensor<[4], f32>
    } else {
      %wn = mo.negative(%w) : !mo.tensor<[4], f32>
      %c = mo.max(%x, %wn) : !mo.tensor<[4], f32>
      mo.yield %c : !mo.tensor<[4], f32>
    }
    mo.yield %inner : !mo.tensor<[4], f32>
  }
  mo.output %r : !mo.tensor<[4], f32>
}

// -----

// The inner if computes the transform in only one of its branches, so the
// transform stays in both ifs.
// CHECK-LABEL: mo.graph @nested_one_inner_branch
mo.graph @nested_one_inner_branch(%x: !mo.tensor<[4], f32>, %c0: !mo.tensor<[], bool>, %c1: !mo.tensor<[], bool>) -> (!mo.tensor<[4], f32>) {
  // CHECK: mo.if
  // CHECK-NEXT: mo.negative
  // CHECK: } else {
  // CHECK-NEXT: mo.if
  // CHECK-NEXT: mo.negative
  %w = mo.constant.external [name = "w", align = 1, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[4], f32>
  %r = mo.if(%c0 : !mo.tensor<[], bool>) -> !mo.tensor<[4], f32> {
    %wn = mo.negative(%w) : !mo.tensor<[4], f32>
    %a = mo.add(%x, %wn) : !mo.tensor<[4], f32>
    mo.yield %a : !mo.tensor<[4], f32>
  } else {
    %inner = mo.if(%c1 : !mo.tensor<[], bool>) -> !mo.tensor<[4], f32> {
      %wn = mo.negative(%w) : !mo.tensor<[4], f32>
      %b = mo.mul(%x, %wn) : !mo.tensor<[4], f32>
      mo.yield %b : !mo.tensor<[4], f32>
    } else {
      mo.yield %x : !mo.tensor<[4], f32>
    }
    mo.yield %inner : !mo.tensor<[4], f32>
  }
  mo.output %r : !mo.tensor<[4], f32>
}

// -----

// One branch computes the transform twice: all three copies become one.
// CHECK-LABEL: mo.graph @duplicate_in_branch
mo.graph @duplicate_in_branch(%x: !mo.tensor<[4], f32>, %cond: !mo.tensor<[], bool>) -> (!mo.tensor<[4], f32>) {
  // CHECK: %[[WN:.+]] = mo.negative
  // CHECK: mo.if
  // CHECK-NEXT: %[[A:.+]] = mo.add(%{{.+}}, %[[WN]])
  // CHECK-NEXT: mo.mul(%[[A]], %[[WN]])
  // CHECK: } else {
  // CHECK-NEXT: mo.add(%{{.+}}, %[[WN]])
  %w = mo.constant.external [name = "w", align = 1, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[4], f32>
  %r = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.tensor<[4], f32> {
    %wn0 = mo.negative(%w) : !mo.tensor<[4], f32>
    %wn1 = mo.negative(%w) : !mo.tensor<[4], f32>
    %a = mo.add(%x, %wn0) : !mo.tensor<[4], f32>
    %b = mo.mul(%a, %wn1) : !mo.tensor<[4], f32>
    mo.yield %b : !mo.tensor<[4], f32>
  } else {
    %wn = mo.negative(%w) : !mo.tensor<[4], f32>
    %c = mo.add(%x, %wn) : !mo.tensor<[4], f32>
    mo.yield %c : !mo.tensor<[4], f32>
  }
  mo.output %r : !mo.tensor<[4], f32>
}

// -----

// A broadcast stays in its branch, next to its user, and so do the ops on its
// result, even when both branches have them.
// CHECK-LABEL: mo.graph @broadcast_stays
mo.graph @broadcast_stays(%x: !mo.tensor<[2, 4], f32>, %cond: !mo.tensor<[], bool>) -> (!mo.tensor<[2, 4], f32>) {
  // CHECK: %[[W:.+]] = mo.constant.external [name = "w"
  // CHECK: mo.if
  // CHECK-NEXT: %[[B:.+]] = mo.static.broadcast_to(%[[W]])
  // CHECK-NEXT: %[[N:.+]] = mo.negative(%[[B]])
  // CHECK-NEXT: mo.add(%{{.+}}, %[[N]])
  // CHECK: } else {
  // CHECK-NEXT: mo.static.broadcast_to(%[[W]])
  %w = mo.constant.external [name = "w", align = 1, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[4], f32>
  %r = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.tensor<[2, 4], f32> {
    %b = mo.static.broadcast_to(%w) : !mo.tensor<[4], f32> -> !mo.tensor<[2, 4], f32>
    %n = mo.negative(%b) : !mo.tensor<[2, 4], f32>
    %a = mo.add(%x, %n) : !mo.tensor<[2, 4], f32>
    mo.yield %a : !mo.tensor<[2, 4], f32>
  } else {
    %b = mo.static.broadcast_to(%w) : !mo.tensor<[4], f32> -> !mo.tensor<[2, 4], f32>
    %n = mo.negative(%b) : !mo.tensor<[2, 4], f32>
    %m = mo.mul(%x, %n) : !mo.tensor<[2, 4], f32>
    mo.yield %m : !mo.tensor<[2, 4], f32>
  }
  mo.output %r : !mo.tensor<[2, 4], f32>
}

// -----

// A constant moves out with the weight op that reads it; ops in the branch
// that read it use the moved constant.
// CHECK-LABEL: mo.graph @weight_with_branch_constant
mo.graph @weight_with_branch_constant(%x: !mo.tensor<[2], f32>, %cond: !mo.tensor<[], bool>) -> (!mo.tensor<[2], f32>) {
  // CHECK-DAG: %[[W:.+]] = mo.constant.external [name = "w"
  // CHECK-DAG: %[[C:.+]] = mo.constant <
  // CHECK: %[[WC:.+]] = mo.mul(%[[W]], %[[C]])
  // CHECK: mo.if
  // CHECK-NEXT: %[[A:.+]] = mo.add(%{{.+}}, %[[C]])
  // CHECK-NEXT: mo.add(%[[A]], %[[WC]])
  // CHECK: } else {
  // CHECK-NEXT: mo.mul(%{{.+}}, %[[WC]])
  %w = mo.constant.external [name = "w", align = 1, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[2], f32>
  %r = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.tensor<[2], f32> {
    %c = mo.constant <#M.dense_array<1.0, 0.0> : tensor<2xf32>> : !mo.tensor<[2], f32>
    %wc = mo.mul(%w, %c) : !mo.tensor<[2], f32>
    %a = mo.add(%x, %c) : !mo.tensor<[2], f32>
    %b = mo.add(%a, %wc) : !mo.tensor<[2], f32>
    mo.yield %b : !mo.tensor<[2], f32>
  } else {
    %c = mo.constant <#M.dense_array<1.0, 0.0> : tensor<2xf32>> : !mo.tensor<[2], f32>
    %wc = mo.mul(%w, %c) : !mo.tensor<[2], f32>
    %d = mo.mul(%x, %wc) : !mo.tensor<[2], f32>
    mo.yield %d : !mo.tensor<[2], f32>
  }
  mo.output %r : !mo.tensor<[2], f32>
}

// -----

// A weight transfer ordered on a root chain moves out when both branches make
// it; an op on the transferred value that one branch adds stays there.
// CHECK-LABEL: mo.graph @chained_transfer
mo.graph @chained_transfer(%cond: !mo.tensor<[], bool>) -> (!mo.tensor<[2], f32, gpu:0>) {
  // CHECK: %[[W:.+]] = mo.constant.external [name = "w"
  // CHECK: %[[CH:.+]] = mo.chain.create()
  // CHECK: %[[T:.+]], %{{.+}} = mo.transfer[%[[CH]]] %[[W]]
  // CHECK: mo.if
  // CHECK-NEXT: %[[N:.+]] = mo.negative(%[[T]])
  // CHECK-NEXT: mo.yield %[[N]]
  // CHECK: } else {
  // CHECK-NEXT: mo.yield %[[T]]
  %w = mo.constant.external [name = "w", align = 1, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[2], f32>
  %ch = mo.chain.create()
  %r = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.tensor<[2], f32, gpu:0> {
    %t, %out = mo.transfer[%ch] %w : !mo.tensor<[2], f32> to <"gpu", 0>
    %n = mo.negative(%t) : !mo.tensor<[2], f32, gpu:0>
    mo.yield %n : !mo.tensor<[2], f32, gpu:0>
  } else {
    %t, %out = mo.transfer[%ch] %w : !mo.tensor<[2], f32> to <"gpu", 0>
    mo.yield %t : !mo.tensor<[2], f32, gpu:0>
  }
  mo.output %r : !mo.tensor<[2], f32, gpu:0>
}

// -----

// A concat of weights holds no more elements than its operands, so it moves
// out.
// CHECK-LABEL: mo.graph @concat_moves
mo.graph @concat_moves(%x: !mo.tensor<[4], f32>, %cond: !mo.tensor<[], bool>) -> (!mo.tensor<[4], f32>) {
  // CHECK: %[[C:.+]] = mo.concat
  // CHECK: mo.if
  // CHECK-NEXT: mo.add(%{{.+}}, %[[C]])
  // CHECK: } else {
  // CHECK-NEXT: mo.mul(%{{.+}}, %[[C]])
  %a = mo.constant.external [name = "a", align = 1, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[2], f32>
  %b = mo.constant.external [name = "b", align = 1, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[2], f32>
  %r = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.tensor<[4], f32> {
    %c = mo.concat[0](%a, %b) : (!mo.tensor<[2], f32>, !mo.tensor<[2], f32>) -> !mo.tensor<[4], f32>
    %s = mo.add(%x, %c) : !mo.tensor<[4], f32>
    mo.yield %s : !mo.tensor<[4], f32>
  } else {
    %c = mo.concat[0](%a, %b) : (!mo.tensor<[2], f32>, !mo.tensor<[2], f32>) -> !mo.tensor<[4], f32>
    %p = mo.mul(%x, %c) : !mo.tensor<[4], f32>
    mo.yield %p : !mo.tensor<[4], f32>
  }
  mo.output %r : !mo.tensor<[4], f32>
}

// -----

// A broadcast that reaches its user in the branch only through a transfer
// moves out with the transfer.
// CHECK-LABEL: mo.graph @broadcast_through_transfer
mo.graph @broadcast_through_transfer(%x: !mo.tensor<[2, 4], f32, gpu:0>, %cond: !mo.tensor<[], bool>) -> (!mo.tensor<[2, 4], f32, gpu:0>) {
  // CHECK: %[[B:.+]] = mo.static.broadcast_to
  // CHECK: %[[T:.+]], %{{.+}} = mo.transfer[%{{.+}}] %[[B]]
  // CHECK: mo.if
  // CHECK-NEXT: mo.add(%{{.+}}, %[[T]])
  // CHECK: } else {
  // CHECK-NEXT: mo.mul(%{{.+}}, %[[T]])
  %w = mo.constant.external [name = "w", align = 1, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[4], f32>
  %ch = mo.chain.create()
  %r = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.tensor<[2, 4], f32, gpu:0> {
    %b = mo.static.broadcast_to(%w) : !mo.tensor<[4], f32> -> !mo.tensor<[2, 4], f32>
    %t, %out = mo.transfer[%ch] %b : !mo.tensor<[2, 4], f32> to <"gpu", 0>
    %a = mo.add(%x, %t) : !mo.tensor<[2, 4], f32, gpu:0>
    mo.yield %a : !mo.tensor<[2, 4], f32, gpu:0>
  } else {
    %b = mo.static.broadcast_to(%w) : !mo.tensor<[4], f32> -> !mo.tensor<[2, 4], f32>
    %t, %out = mo.transfer[%ch] %b : !mo.tensor<[2, 4], f32> to <"gpu", 0>
    %m = mo.mul(%x, %t) : !mo.tensor<[2, 4], f32, gpu:0>
    mo.yield %m : !mo.tensor<[2, 4], f32, gpu:0>
  }
  mo.output %r : !mo.tensor<[2, 4], f32, gpu:0>
}
