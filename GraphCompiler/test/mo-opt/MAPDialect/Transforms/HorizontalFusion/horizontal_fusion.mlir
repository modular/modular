// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-fuse-horizontal | FileCheck %s
// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-fuse-horizontal --mlir-print-debuginfo --mlir-print-local-scope | FileCheck %s --check-prefix=LOC

// Two adds over the same index space sharing %x fuse into one kernel with two
// stores. The duplicate load of %x survives in isolation because the CSE that
// collapses it runs in `FuseKernels`, not in this pass.

// CHECK-LABEL: mo.graph @shared_operand
// CHECK-SAME:  (%[[X:.*]]: !mo.tensor<[8], f32>, %[[Y:.*]]: !mo.tensor<[8], f32>, %[[Z:.*]]: !mo.tensor<[8], f32>)
// CHECK-NEXT:  %[[OUT:.*]]:2 = map.iter.elementwise<[8]>
// CHECK-NEXT:  inputs (%[[X]]: !mo.tensor<[8], f32>, %[[Y]]: !mo.tensor<[8], f32>, %[[Z]]: !mo.tensor<[8], f32>)
// CHECK-NEXT:  outputs (%[[O0:.*]]: !mo.tensor<[8], f32>, %[[O1:.*]]: !mo.tensor<[8], f32>)
// CHECK:       map.store %[[O0]]
// CHECK:       map.store %[[O1]]
// CHECK:       mo.output %[[OUT]]#0, %[[OUT]]#1
mo.graph @shared_operand(
  %x: !mo.tensor<[8], f32>,
  %y: !mo.tensor<[8], f32>,
  %z: !mo.tensor<[8], f32>
) -> (!mo.tensor<[8], f32>, !mo.tensor<[8], f32>) {
  %a = mo.add(%x, %y) : !mo.tensor<[8], f32>
  %b = mo.add(%x, %z) : !mo.tensor<[8], f32>
  mo.output %a, %b : !mo.tensor<[8], f32>, !mo.tensor<[8], f32>
}

// Sharing an operand is the whole trigger, so disjoint inputs stay apart.

// CHECK-LABEL: mo.graph @no_shared_operand
// CHECK-NOT:   map.iter.elementwise<[8]>{{.*}}:2
// CHECK:       map.iter.elementwise<[8]>
// CHECK:       map.iter.elementwise<[8]>
mo.graph @no_shared_operand(
  %w: !mo.tensor<[8], f32>,
  %x: !mo.tensor<[8], f32>,
  %y: !mo.tensor<[8], f32>,
  %z: !mo.tensor<[8], f32>
) -> (!mo.tensor<[8], f32>, !mo.tensor<[8], f32>) {
  %a = mo.add(%w, %x) : !mo.tensor<[8], f32>
  %b = mo.add(%y, %z) : !mo.tensor<[8], f32>
  mo.output %a, %b : !mo.tensor<[8], f32>, !mo.tensor<[8], f32>
}

// Siblings need not agree on output dtype: each store keeps its own.

// CHECK-LABEL: mo.graph @add_and_cast_share_operand
// CHECK:       %[[OUT:.*]]:2 = map.iter.elementwise<[8]>
// CHECK:       outputs (%[[O0:[a-z0-9]+]]: !mo.tensor<[8], f32>, %[[O1:[a-z0-9]+]]: !mo.tensor<[8], si32>)
// CHECK:       map.store %[[O0]][%{{.*}}] = %{{.*}} : !mo.tensor<[8], f32>
// CHECK:       %[[CAST:.*]] = map.call.elementwise @{{.*}}.Cast(%{{.*}}) : !map.index<1>, f32 -> si32
// CHECK-NEXT:  map.store %[[O1]][%{{.*}}] = %[[CAST]] : !mo.tensor<[8], si32>
// CHECK:       mo.output %[[OUT]]#0, %[[OUT]]#1
mo.graph @add_and_cast_share_operand(
  %x: !mo.tensor<[8], f32>,
  %y: !mo.tensor<[8], f32>
) -> (!mo.tensor<[8], f32>, !mo.tensor<[8], si32>) {
  %a = mo.add(%x, %y) : !mo.tensor<[8], f32>
  %c = mo.cast(%x) : (!mo.tensor<[8], f32>) -> !mo.tensor<[8], si32>
  mo.output %a, %c : !mo.tensor<[8], f32>, !mo.tensor<[8], si32>
}

// A parametric extent: the pair is matched on the dim parameter, not a
// constant, so the fused kernel iterates `[n]` and both stores land in it.

// CHECK-LABEL: mo.graph @two_adds_dynamic<n>
// CHECK:       %[[OUT:.*]]:2 = map.iter.elementwise<[n]>
// CHECK-NEXT:  inputs (%{{.*}}: !mo.tensor<[n], f32>, %{{.*}}: !mo.tensor<[n], f32>, %{{.*}}: !mo.tensor<[n], f32>)
// CHECK-NEXT:  outputs (%[[O0:[a-z0-9]+]]: !mo.tensor<[n], f32>, %[[O1:[a-z0-9]+]]: !mo.tensor<[n], f32>)
// CHECK:       map.store %[[O0]]
// CHECK:       map.store %[[O1]]
// CHECK:       mo.output %[[OUT]]#0, %[[OUT]]#1
mo.graph @two_adds_dynamic<n>(
  %x: !mo.tensor<[n], f32>,
  %y: !mo.tensor<[n], f32>,
  %z: !mo.tensor<[n], f32>
) -> (!mo.tensor<[n], f32>, !mo.tensor<[n], f32>) {
  %a = mo.add(%x, %y) : !mo.tensor<[n], f32>
  %b = mo.add(%x, %z) : !mo.tensor<[n], f32>
  mo.output %a, %b : !mo.tensor<[n], f32>, !mo.tensor<[n], f32>
}

// A reader between the two kernels blocks the pair, and it counts even when it
// sits inside a region: the fused kernel would be built at the second one's
// position, leaving the `mo.if` above it consuming a result produced below --
// a cycle. Tight captures here matter: a fused pair prints as `%N:2 =`, which
// these patterns cannot match.

// CHECK-LABEL: mo.graph @reader_inside_branch
// CHECK:       %[[A:[a-z0-9]+]] = map.iter.elementwise<[4]>
// CHECK:       %[[IF:[a-z0-9]+]] = mo.if
// CHECK:       map.iter.elementwise<[4]>
// CHECK:       %[[B:[a-z0-9]+]] = map.iter.elementwise<[4]>
// CHECK:       mo.output %[[B]], %[[IF]]
mo.graph @reader_inside_branch(
  %cond: !mo.tensor<[], bool>,
  %a: !mo.tensor<[4], f32>,
  %b: !mo.tensor<[4], f32>,
  %c: !mo.tensor<[4], f32>,
  %d: !mo.tensor<[4], f32>
) -> (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) {
  %x = mo.add(%a, %b) : !mo.tensor<[4], f32>
  %t = mo.if (%cond : !mo.tensor<[], bool>) -> !mo.tensor<[4], f32> {
    %m = mo.mul(%x, %c) : !mo.tensor<[4], f32>
    mo.yield %m : !mo.tensor<[4], f32>
  } else {
    mo.yield %c : !mo.tensor<[4], f32>
  }
  %y = mo.add(%a, %d) : !mo.tensor<[4], f32>
  mo.output %y, %t : !mo.tensor<[4], f32>, !mo.tensor<[4], f32>
}

// The merged kernel launches async if either member does, as input fusion
// combines the launch policy (a transpose launches sync), keeps the discardable
// attributes of both, and fuses their locations.

// CHECK-LABEL: mo.graph @merge_launch_attrs_and_locs
// CHECK:       %[[OUT:.*]]:2 = map.iter.elementwise<[8]>
// CHECK:       } {fusionInfo = {{.*}}, hasReadEffect = false, hasWriteEffect = false, test.only_b}
// CHECK-NEXT:  mo.output %[[OUT]]#0, %[[OUT]]#1
// LOC-LABEL:   mo.graph @merge_launch_attrs_and_locs
// LOC:         test.only_b} loc(fused["a", "b"])
mo.graph @merge_launch_attrs_and_locs(
  %x: !mo.tensor<[8], f32>
) -> (!mo.tensor<[8], f32>, !mo.tensor<[8], f32>) {
  %a = map.iter.elementwise<[8]> inputs (%x: !mo.tensor<[8], f32>) outputs (%out: !mo.tensor<[8], f32>) index (%idx: !map.index<1>) -> !mo.tensor<[8], f32> {
    %v = map.load %x[%idx] : !mo.tensor<[8], f32>
    map.store %out[%idx] = %v : !mo.tensor<[8], f32>
  } {dispatch = #mogg.dispatch<immediate>} loc("a")
  %b = map.iter.elementwise<[8]> inputs (%x: !mo.tensor<[8], f32>) outputs (%out: !mo.tensor<[8], f32>) index (%idx: !map.index<1>) -> !mo.tensor<[8], f32> {
    %v = map.load %x[%idx] : !mo.tensor<[8], f32>
    map.store %out[%idx] = %v : !mo.tensor<[8], f32>
  } {test.only_b} loc("b")
  mo.output %a, %b : !mo.tensor<[8], f32>, !mo.tensor<[8], f32>
}

// Two slices of one tensor share their loaded input, but each is a pure view
// that `RaiseBackContiguousViews` turns into a zero-copy view. Merged, they
// would be a kernel copying both windows, so they stay apart.

// CHECK-LABEL: mo.graph @two_views_of_one_tensor
// CHECK:       map.iter.elementwise<[4, 64]>
// CHECK-NOT:   map.store
// CHECK:       map.store
// CHECK-NOT:   map.store
// CHECK:       map.iter.elementwise<[4, 64]>
// CHECK-NOT:   map.store
// CHECK:       map.store
// CHECK-NOT:   map.store
// CHECK:       mo.output
mo.graph @two_views_of_one_tensor(%w: !mo.tensor<[8, 64], f32>) -> (!mo.tensor<[4, 64], f32>, !mo.tensor<[4, 64], f32>) {
  %start0 = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop0 = mo.constant <#M.dense_array<4, 64> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %start1 = mo.constant <#M.dense_array<4, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop1 = mo.constant <#M.dense_array<8, 64> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %v0 = mo.slice(%w, %start0, %stop0, %step) : (
    !mo.tensor<[8, 64], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[4, 64], f32>
  %v1 = mo.slice(%w, %start1, %stop1, %step) : (
    !mo.tensor<[8, 64], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[4, 64], f32>
  mo.output %v0, %v1 : !mo.tensor<[4, 64], f32>, !mo.tensor<[4, 64], f32>
}
