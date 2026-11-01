// RUN: mo-opt --mlir-print-local-scope %s --mogg-import-packages --mo-to-map --map-fuse-horizontal --map-fuse-epilogue | FileCheck %s

// A multi-output consumer fuses into its producer's epilogue, which grows the
// producer one destination-passing result per extra output. Horizontal fusion
// runs first here because that is the only way a multi-output foreach arises:
// the two adds share `%mm`, so they merge, and only then does `%mm` have the
// single use epilogue fusion requires.

// Each store must land on the producer's own output, not on one of the inputs
// threaded in as a capture beside them.

// CHECK-LABEL: mo.graph @matmul_two_adds
// CHECK:       %[[OUT:.*]]:2 = map.iter.opaque
// CHECK:       outputs (%[[O0:.*]] : !mo.tensor<[8, 8], f32>, %[[O1:.*]] : !mo.tensor<[8, 8], f32>)
// CHECK-NEXT:  %[[EPI:.*]] = map.iter.epilogue (%[[VAL:.*]] : f32, %{{.*}} : !map.index<2>) capturing {%[[CAP0:.*]] = %[[O0]] : {{.*}}, %[[CAPZ:.*]] = %{{.*}}, %[[CAPW:.*]] = %{{.*}}, %[[CAP1:.*]] = %[[O1]] : !mo.tensor<[8, 8], f32>}
// CHECK-NEXT:  %[[ZV:.*]] = map.load %[[CAPZ]]
// CHECK-NEXT:  %[[ADD0:.*]] = map.call.elementwise @{{.*}}.Add(%{{.*}}, %[[VAL]], %[[ZV]])
// CHECK-NEXT:  %[[WV:.*]] = map.load %[[CAPW]]
// CHECK-NEXT:  %[[ADD1:.*]] = map.call.elementwise @{{.*}}.Add(%{{.*}}, %[[VAL]], %[[WV]])
// CHECK-NEXT:  map.store %[[CAP0]][%{{.*}}] = %[[ADD0]]
// CHECK-NEXT:  map.store %[[CAP1]][%{{.*}}] = %[[ADD1]]
// CHECK:       map.call.opaque @{{.*}}.Matmul{{.*}} inputs(%{{.*}}, %{{.*}}) outputs(%[[EPI]])
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_epilogue<[#mogg.fusion_leaf<"mo.matmul"{{.*}}, #mogg.fusion_horizontal<[#mogg.fusion_leaf<"mo.add"{{.*}}, #mogg.fusion_leaf<"mo.add"{{.*}}]>]
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output %[[OUT]]#0, %[[OUT]]#1
mo.graph @matmul_two_adds(
  %x: !mo.tensor<[8, 8], f32>,
  %y: !mo.tensor<[8, 8], f32>,
  %z: !mo.tensor<[8, 8], f32>,
  %w: !mo.tensor<[8, 8], f32>
) -> (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>) {
  %mm = mo.matmul(%x, %y) : (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 8], f32>
  %a = mo.add(%mm, %z) : !mo.tensor<[8, 8], f32>
  %b = mo.add(%mm, %w) : !mo.tensor<[8, 8], f32>
  mo.output %a, %b : !mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>
}

// Growing by more than one: three consumers, three destinations, one kernel.

// CHECK-LABEL: mo.graph @matmul_three_adds
// CHECK:       %[[OUT3:.*]]:3 = map.iter.opaque
// CHECK:       outputs (%{{.*}} : !mo.tensor<[8, 8], f32>, %{{.*}} : !mo.tensor<[8, 8], f32>, %{{.*}} : !mo.tensor<[8, 8], f32>)
// CHECK-COUNT-3: map.store
// CHECK-NOT:   map.store
// CHECK:       map.call.opaque @{{.*}}.Matmul
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output %[[OUT3]]#0, %[[OUT3]]#1, %[[OUT3]]#2
mo.graph @matmul_three_adds(
  %x: !mo.tensor<[8, 8], f32>,
  %y: !mo.tensor<[8, 8], f32>,
  %z: !mo.tensor<[8, 8], f32>,
  %w: !mo.tensor<[8, 8], f32>,
  %v: !mo.tensor<[8, 8], f32>
) -> (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>) {
  %mm = mo.matmul(%x, %y) : (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 8], f32>
  %a = mo.add(%mm, %z) : !mo.tensor<[8, 8], f32>
  %b = mo.add(%mm, %w) : !mo.tensor<[8, 8], f32>
  %c = mo.add(%mm, %v) : !mo.tensor<[8, 8], f32>
  mo.output %a, %b, %c : !mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>
}

// The destinations need not share a dtype. Which of them backs the slice the
// kernel binds is `AllocateAccumulatingBuffers`' job (see its own test); here
// the point is that each store keeps the dtype of the output it writes.

// CHECK-LABEL: mo.graph @matmul_add_and_cast
// CHECK:       %[[OUT:.*]]:2 = map.iter.opaque
// CHECK:       outputs (%[[O0:.*]] : !mo.tensor<[2, 2], f32>, %[[O1:.*]] : !mo.tensor<[2, 2], si32>)
// CHECK-NEXT:  map.iter.epilogue (%[[VAL:.*]] : f32, %{{.*}} : !map.index<2>) capturing {%[[CAP0:.*]] = %[[O0]] : {{.*}}, %{{.*}}, %[[CAP1:.*]] = %[[O1]] : !mo.tensor<[2, 2], si32>}
// CHECK:       map.store %[[CAP0]][%{{.*}}] = %{{.*}} : !mo.tensor<[2, 2], f32>
// CHECK:       map.store %[[CAP1]][%{{.*}}] = %{{.*}} : !mo.tensor<[2, 2], si32>
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output %[[OUT]]#0, %[[OUT]]#1
mo.graph @matmul_add_and_cast(
  %a: !mo.tensor<[2, 2], f32>,
  %b: !mo.tensor<[2, 2], f32>,
  %y: !mo.tensor<[2, 2], f32>
) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], si32>) {
  %mm = mo.matmul(%a, %b) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %s = mo.add(%mm, %y) : !mo.tensor<[2, 2], f32>
  %c = mo.cast(%mm) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], si32>
  mo.output %s, %c : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], si32>
}

// The same chain over a parametric square extent: one dim parameter binds the
// matmul, both adds and both outputs.

// CHECK-LABEL: mo.graph @matmul_two_adds_dynamic<N>
// CHECK:       %[[OUT:.*]]:2 = map.iter.opaque
// CHECK:       outputs (%[[O0:.*]] : !mo.tensor<[N, N], f32>, %[[O1:.*]] : !mo.tensor<[N, N], f32>)
// CHECK-NEXT:  %[[EPI:.*]] = map.iter.epilogue (%{{.*}} : f32, %{{.*}} : !map.index<2>) capturing {%[[CAP0:.*]] = %[[O0]] : {{.*}}, %{{.*}}, %{{.*}}, %[[CAP1:.*]] = %[[O1]] : !mo.tensor<[N, N], f32>}
// CHECK:       map.store %[[CAP0]]
// CHECK:       map.store %[[CAP1]]
// CHECK:       map.call.opaque @{{.*}}.Matmul{{.*}} outputs(%[[EPI]])
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output %[[OUT]]#0, %[[OUT]]#1
mo.graph @matmul_two_adds_dynamic<N>(
  %x: !mo.tensor<[N, N], f32>,
  %y: !mo.tensor<[N, N], f32>,
  %z: !mo.tensor<[N, N], f32>,
  %w: !mo.tensor<[N, N], f32>
) -> (!mo.tensor<[N, N], f32>, !mo.tensor<[N, N], f32>) {
  %mm = mo.matmul(%x, %y) : (!mo.tensor<[N, N], f32>, !mo.tensor<[N, N], f32>) -> !mo.tensor<[N, N], f32>
  %a = mo.add(%mm, %z) : !mo.tensor<[N, N], f32>
  %b = mo.add(%mm, %w) : !mo.tensor<[N, N], f32>
  mo.output %a, %b : !mo.tensor<[N, N], f32>, !mo.tensor<[N, N], f32>
}

// A producer whose epilogue already stores more than once takes no further
// consumer: fusing rebuilds a single store, which cannot serve the
// destinations it already has. The two adds merge and fuse in as above,
// leaving `%c` -- an ordinary single-result consumer of one of those outputs
// -- as its own kernel. Without the check it reached `getUniqueStore`, whose
// "exactly one store" assertion the grown epilogue no longer satisfies.

// CHECK-LABEL: mo.graph @matmul_two_adds_then_consumer
// CHECK:       %[[OUT:.*]]:2 = map.iter.opaque
// CHECK:       map.iter.epilogue
// CHECK-COUNT-2: map.store
// CHECK-NOT:   map.store
// CHECK:       map.call.opaque @{{.*}}.Matmul
// CHECK:       map.iter.elementwise
// CHECK:       map.call.elementwise @{{.*}}.Add
mo.graph @matmul_two_adds_then_consumer(
  %x: !mo.tensor<[8, 8], f32>,
  %y: !mo.tensor<[8, 8], f32>,
  %z: !mo.tensor<[8, 8], f32>,
  %w: !mo.tensor<[8, 8], f32>,
  %v: !mo.tensor<[8, 8], f32>
) -> (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>) {
  %mm = mo.matmul(%x, %y) : (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 8], f32>
  %a = mo.add(%mm, %z) : !mo.tensor<[8, 8], f32>
  %b = mo.add(%mm, %w) : !mo.tensor<[8, 8], f32>
  %c = mo.add(%a, %v) : !mo.tensor<[8, 8], f32>
  mo.output %c, %b : !mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>
}
