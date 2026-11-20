// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-fuse-horizontal | FileCheck %s
// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-fuse-horizontal --mlir-print-debuginfo --mlir-print-local-scope | FileCheck %s --check-prefix=LOC

// A consumer fuses into its producer's epilogue even when the producer's own
// result is still read elsewhere. `MOToMAP` gives every fusible output an
// epilogue whose body is the bare store, so that store and the consumer's are
// siblings over one index space: the producer keeps writing the result it owes
// its other reader, and the consumer stores beside it into a grown output.

// CHECK-LABEL: mo.graph @matmul_result_escapes
// CHECK:       %[[OUT:.*]]:2 = map.iter.opaque
// CHECK:       outputs (%[[O0:[a-z0-9]+]] : !mo.tensor<[8, 8], f32>, %[[O1:[a-z0-9]+]] : !mo.tensor<[8, 8], f32>)
// CHECK:       %[[EPI:[a-z0-9]+]] = map.iter.epilogue (%[[VAL:[a-z0-9]+]] : f32, %{{.*}} : !map.index<2>) capturing {%[[CAP0:[a-z0-9]+]] = %[[O0]] : {{.*}}, %[[CAPZ:[a-z0-9]+]] = %{{.*}}, %[[CAP1:[a-z0-9]+]] = %[[O1]] : !mo.tensor<[8, 8], f32>}
// The producer's own store survives -- `%mm` is still an output of the graph.
// CHECK-NEXT:  map.store %[[CAP0]][%{{.*}}] = %[[VAL]]
// CHECK-NEXT:  %[[ZV:[a-z0-9]+]] = map.load %[[CAPZ]]
// CHECK-NEXT:  %[[ADD:[a-z0-9]+]] = map.call.elementwise @{{.*}}.Add(%{{.*}}, %[[VAL]], %[[ZV]])
// CHECK-NEXT:  map.store %[[CAP1]][%{{.*}}] = %[[ADD]]
// CHECK:       map.call.opaque @{{.*}}.Matmul{{.*}} outputs(%[[EPI]])
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output %[[OUT]]#0, %[[OUT]]#1
// The merged kernel's location fuses both kernels'.
// LOC-LABEL:   mo.graph @matmul_result_escapes
// LOC:         map.call.opaque @{{.*}}.Matmul
// LOC-NEXT:    } {{.*}} loc(fused["add", "mm"])
mo.graph @matmul_result_escapes(
  %x: !mo.tensor<[8, 8], f32>,
  %y: !mo.tensor<[8, 8], f32>,
  %z: !mo.tensor<[8, 8], f32>
) -> (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>) {
  %mm = mo.matmul(%x, %y) : (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 8], f32> loc("mm")
  %a = mo.add(%mm, %z) : !mo.tensor<[8, 8], f32> loc("add")
  mo.output %mm, %a : !mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>
}

// A consumer that also reads one of the producer's inputs threads it in a
// second time; the merged kernel keeps one copy.

// CHECK-LABEL: mo.graph @matmul_result_escapes_shared_input
// CHECK:       map.iter.opaque
// CHECK-NEXT:  inputs (%{{[a-z0-9]+}}: !mo.tensor<[8, 8], f32>, %{{[a-z0-9]+}}: !mo.tensor<[8, 8], f32>)
// CHECK-NEXT:  outputs
mo.graph @matmul_result_escapes_shared_input(
  %x: !mo.tensor<[8, 8], f32>,
  %y: !mo.tensor<[8, 8], f32>
) -> (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>) {
  %mm = mo.matmul(%x, %y) : (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 8], f32>
  %a = mo.add(%mm, %x) : !mo.tensor<[8, 8], f32>
  mo.output %mm, %a : !mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>
}

// Two consumers and an escape. Eligibility takes only a lone consumer, so the
// first round leaves this alone and `FuseHorizontal` merges the adds; the pair
// it leaves behind is the lone consumer a later round fuses, giving three
// stores -- the escape's and one per add.

// CHECK-LABEL: mo.graph @matmul_result_escapes_two_consumers
// CHECK:       %[[OUT3:.*]]:3 = map.iter.opaque
// CHECK:       outputs (%{{.*}} : !mo.tensor<[8, 8], f32>, %{{.*}} : !mo.tensor<[8, 8], f32>, %{{.*}} : !mo.tensor<[8, 8], f32>)
// CHECK-COUNT-3: map.store
// CHECK-NOT:   map.store
// CHECK:       map.call.opaque @{{.*}}.Matmul
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output %[[OUT3]]#0, %[[OUT3]]#1, %[[OUT3]]#2
mo.graph @matmul_result_escapes_two_consumers(
  %x: !mo.tensor<[8, 8], f32>,
  %y: !mo.tensor<[8, 8], f32>,
  %z: !mo.tensor<[8, 8], f32>,
  %w: !mo.tensor<[8, 8], f32>
) -> (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>) {
  %mm = mo.matmul(%x, %y) : (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 8], f32>
  %a = mo.add(%mm, %z) : !mo.tensor<[8, 8], f32>
  %b = mo.add(%mm, %w) : !mo.tensor<[8, 8], f32>
  mo.output %mm, %a, %b : !mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>
}

// The escape is an `mo.if` whose branches read the result, and the consumer
// also reads what the `mo.if` yields. The branches do not list what they read
// as operands, so only they carry the edge back to the matmul: merging the add
// in would have the matmul wait on the `mo.if`, which waits on the matmul.

// CHECK-LABEL: mo.graph @matmul_result_read_in_branch
// CHECK:       %[[MM:.*]] = map.iter.opaque
// CHECK-NEXT:  inputs
// CHECK-NEXT:  outputs (%{{[a-z0-9]+}} : !mo.tensor<[8, 8], f32>) {
// CHECK:       map.call.opaque @{{.*}}.Matmul
// CHECK:       %[[R:.*]] = mo.if
// CHECK:       } else {
// CHECK:       mo.yield
// CHECK:       %[[ADD:.*]] = map.iter.elementwise
// CHECK-NEXT:  inputs (%[[MM]]: !mo.tensor<[8, 8], f32>, %[[R]]: !mo.tensor<[8, 8], f32>)
// CHECK:       mo.output %[[ADD]]
mo.graph @matmul_result_read_in_branch(
  %x: !mo.tensor<[8, 8], f32>,
  %y: !mo.tensor<[8, 8], f32>,
  %z: !mo.tensor<[8, 8], f32>,
  %cond: !mo.tensor<[], bool>
) -> !mo.tensor<[8, 8], f32> {
  %mm = mo.matmul(%x, %y) : (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 8], f32>
  %r = mo.if (%cond : !mo.tensor<[], bool>) -> !mo.tensor<[8, 8], f32> {
    %m = mo.mul(%mm, %z) : !mo.tensor<[8, 8], f32>
    mo.yield %m : !mo.tensor<[8, 8], f32>
  } else {
    %s = mo.sub(%mm, %z) : !mo.tensor<[8, 8], f32>
    mo.yield %s : !mo.tensor<[8, 8], f32>
  }
  %a = mo.add(%mm, %r) : !mo.tensor<[8, 8], f32>
  mo.output %a : !mo.tensor<[8, 8], f32>
}
