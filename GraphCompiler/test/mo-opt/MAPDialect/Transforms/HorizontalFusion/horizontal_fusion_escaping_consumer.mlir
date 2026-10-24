// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-fuse-horizontal | FileCheck %s
// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-fuse-kernels | FileCheck %s --check-prefix=PIPE
// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-fuse-views --map-fuse-horizontal --map-fuse-epilogue | FileCheck %s --check-prefix=EPI

// An elementwise producer whose result is read by one consumer and also
// escapes merges with that consumer: the merged kernel keeps the producer's
// store for the escaping reader, and the consumer takes the value the producer
// just computed instead of loading it back. Input fusion leaves this case
// alone, as duplicating the producer would not free its store.

// CHECK-LABEL: mo.graph @add_relu_escapes
// CHECK:       %[[OUT:.*]]:2 = map.iter.elementwise<[8, 8]>
// CHECK:       outputs (%[[X:[a-z0-9]+]]: {{.*}}, %[[Y:[a-z0-9]+]]: {{.*}})
// CHECK:       %[[ADD:[0-9]+]] = map.call.elementwise @{{.*}}.Add
// CHECK-NEXT:  map.store %[[X]][%{{.*}}] = %[[ADD]]
// CHECK-NEXT:  %[[RELU:[0-9]+]] = map.call.elementwise @{{.*}}.ReLU(%{{.*}}, %[[ADD]])
// CHECK-NEXT:  map.store %[[Y]][%{{.*}}] = %[[RELU]]
// CHECK:       fusionInfo = #add_relu{{[0-9]*}},
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output %[[OUT]]#0, %[[OUT]]#1
// PIPE-LABEL:  mo.graph @add_relu_escapes
// PIPE-COUNT-1: map.iter.elementwise
// PIPE-NOT:    map.iter.elementwise
// PIPE:        mo.output
mo.graph @add_relu_escapes(%a: !mo.tensor<[8, 8], f32>, %b: !mo.tensor<[8, 8], f32>) -> (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>) {
  %x = mo.add(%a, %b) : !mo.tensor<[8, 8], f32>
  %y = mo.relu(%x) : !mo.tensor<[8, 8], f32>
  mo.output %x, %y : !mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>
}

// Two consumers pair up first, then the pair joins the producer.
// CHECK-LABEL: mo.graph @add_two_consumers_escape
// CHECK:       %[[OUT:.*]]:3 = map.iter.elementwise<[8, 8]>
// CHECK:       %[[ADD:[0-9]+]] = map.call.elementwise @{{.*}}.Add
// CHECK-NEXT:  map.store %{{.*}} = %[[ADD]]
// CHECK-NEXT:  map.call.elementwise @{{.*}}.ReLU(%{{.*}}, %[[ADD]])
// CHECK-NEXT:  map.store
// CHECK-NEXT:  map.call.elementwise @{{.*}}.Sin(%{{.*}}, %[[ADD]])
// CHECK-NEXT:  map.store
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output %[[OUT]]#0, %[[OUT]]#1, %[[OUT]]#2
mo.graph @add_two_consumers_escape(%a: !mo.tensor<[8, 8], f32>, %b: !mo.tensor<[8, 8], f32>) -> (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>) {
  %x = mo.add(%a, %b) : !mo.tensor<[8, 8], f32>
  %y = mo.relu(%x) : !mo.tensor<[8, 8], f32>
  %z = mo.sin(%x) : !mo.tensor<[8, 8], f32>
  mo.output %x, %y, %z : !mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>
}

// The consumer can change the element type: the cast takes the f32 sum and
// stores bf16.
// CHECK-LABEL: mo.graph @add_cast_escapes
// CHECK:       %[[OUT:.*]]:2 = map.iter.elementwise<[8, 4]>
// CHECK:       %[[ADD:[0-9]+]] = map.call.elementwise @{{.*}}.Add
// CHECK-NEXT:  map.store %{{.*}} = %[[ADD]] : !mo.tensor<[8, 4], f32>
// CHECK-NEXT:  %[[CAST:[0-9]+]] = map.call.elementwise @{{.*}}.Cast(%{{.*}}, %[[ADD]]) {{.*}} -> bf16
// CHECK-NEXT:  map.store %{{.*}} = %[[CAST]] : !mo.tensor<[8, 4], bf16>
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output %[[OUT]]#0, %[[OUT]]#1
mo.graph @add_cast_escapes(%a: !mo.tensor<[8, 4], f32>, %b: !mo.tensor<[8, 4], f32>) -> (!mo.tensor<[8, 4], f32>, !mo.tensor<[8, 4], bf16>) {
  %x = mo.add(%a, %b) : !mo.tensor<[8, 4], f32>
  %y = mo.cast(%x) : (!mo.tensor<[8, 4], f32>) -> !mo.tensor<[8, 4], bf16>
  mo.output %x, %y : !mo.tensor<[8, 4], f32>, !mo.tensor<[8, 4], bf16>
}

// The producer may read through a view, here a transpose view fusion folded
// into the relu: only the consumer has to read the result at its own index.
// PIPE-LABEL:  mo.graph @transposed_producer_escapes
// PIPE:        %[[OUT:.*]]:2 = map.iter.elementwise<[8, 4]>
// PIPE:        %[[IDX:[0-9]+]] = map.index.transpose
// PIPE-NEXT:   %[[A:[0-9]+]] = map.load %{{.*}}[%[[IDX]]]
// PIPE-NEXT:   %[[RELU:[0-9]+]] = map.call.elementwise @{{.*}}.ReLU(%{{.*}}, %[[A]])
// PIPE-NEXT:   map.store %{{.*}} = %[[RELU]]
// PIPE-NEXT:   %[[B:[0-9]+]] = map.load
// PIPE-NEXT:   map.call.elementwise @{{.*}}.Add(%{{.*}}, %[[RELU]], %[[B]])
// PIPE-NOT:    map.iter.elementwise
// PIPE:        mo.output %[[OUT]]#0, %[[OUT]]#1
mo.graph @transposed_producer_escapes(%a: !mo.tensor<[4, 8], f32>, %b: !mo.tensor<[8, 4], f32>) -> (!mo.tensor<[8, 4], f32>, !mo.tensor<[8, 4], f32>) {
  %perm = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %t = mo.transpose(%a, %perm) : (!mo.tensor<[4, 8], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[8, 4], f32>
  %x = mo.relu(%t) : !mo.tensor<[8, 4], f32>
  %y = mo.add(%x, %b) : !mo.tensor<[8, 4], f32>
  mo.output %x, %y : !mo.tensor<[8, 4], f32>, !mo.tensor<[8, 4], f32>
}

// A chain whose every link escapes merges link by link into one kernel.
// CHECK-LABEL: mo.graph @chain_escapes
// CHECK:       %[[OUT:.*]]:3 = map.iter.elementwise<[8, 8]>
// CHECK:       %[[ADD:[0-9]+]] = map.call.elementwise @{{.*}}.Add
// CHECK:       %[[MUL0:[0-9]+]] = map.call.elementwise @{{.*}}.Mul(%{{.*}}, %[[ADD]], %{{.*}})
// CHECK:       map.call.elementwise @{{.*}}.Mul(%{{.*}}, %[[MUL0]], %{{.*}})
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output %[[OUT]]#0, %[[OUT]]#1, %[[OUT]]#2
mo.graph @chain_escapes(%a: !mo.tensor<[8, 8], f32>, %b: !mo.tensor<[8, 8], f32>, %s: !mo.tensor<[8, 8], f32>, %t: !mo.tensor<[8, 8], f32>) -> (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>) {
  %x = mo.add(%a, %b) : !mo.tensor<[8, 8], f32>
  %y = mo.mul(%x, %s) : !mo.tensor<[8, 8], f32>
  %z = mo.mul(%y, %t) : !mo.tensor<[8, 8], f32>
  mo.output %x, %y, %z : !mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>
}

// Without an escaping reader the producer is input fusion's to inline, so
// this pass leaves the pair alone.
// CHECK-LABEL: mo.graph @sole_consumer
// CHECK-COUNT-2: map.iter.elementwise
mo.graph @sole_consumer(%a: !mo.tensor<[8, 8], f32>, %b: !mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 8], f32> {
  %x = mo.add(%a, %b) : !mo.tensor<[8, 8], f32>
  %y = mo.relu(%x) : !mo.tensor<[8, 8], f32>
  mo.output %y : !mo.tensor<[8, 8], f32>
}

// The consumer's other operand reads the producer through a matmul, so the
// merged kernel would consume its own result.
// CHECK-LABEL: mo.graph @residual_cycle
// CHECK:       map.iter.elementwise
// CHECK:       map.iter.opaque
// CHECK:       map.iter.elementwise
mo.graph @residual_cycle(%a: !mo.tensor<[8, 8], f32>, %b: !mo.tensor<[8, 8], f32>, %w: !mo.tensor<[8, 8], f32>) -> (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>) {
  %x = mo.add(%a, %b) : !mo.tensor<[8, 8], f32>
  %m = mo.matmul(%x, %w) : (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 8], f32>
  %y = mo.add(%x, %m) : !mo.tensor<[8, 8], f32>
  mo.output %x, %y : !mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>
}

// Once view fusion folds the transpose into the consumer, it reads the
// producer's result at a transposed index: not the element the producer holds
// at that iteration, so the two stay apart.
// PIPE-LABEL:  mo.graph @transposed_read
// PIPE:        %[[X:[0-9]+]] = map.iter.elementwise<[8, 8]>
// PIPE:        map.call.elementwise @{{.*}}.Add
// PIPE:        fusionInfo = #add{{[0-9]*}},
// PIPE:        %[[Y:[0-9]+]] = map.iter.elementwise<[8, 8]>
// PIPE-NEXT:   inputs (%[[X]]:
// PIPE:        %[[IDX:[0-9]+]] = map.index.transpose
// PIPE-NEXT:   %[[XV:[0-9]+]] = map.load %{{.*}}[%[[IDX]]]
// PIPE-NEXT:   map.call.elementwise @{{.*}}.ReLU(%{{.*}}, %[[XV]])
// PIPE:        mo.output %[[X]], %[[Y]]
mo.graph @transposed_read(%a: !mo.tensor<[8, 8], f32>, %b: !mo.tensor<[8, 8], f32>) -> (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>) {
  %x = mo.add(%a, %b) : !mo.tensor<[8, 8], f32>
  %perm = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %t = mo.transpose(%x, %perm) : (!mo.tensor<[8, 8], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[8, 8], f32>
  %y = mo.relu(%t) : !mo.tensor<[8, 8], f32>
  mo.output %x, %y : !mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>
}

// A contiguous reshape stays its own kernel, which can become a zero-copy view
// of `%a`; merging it with the relu would make that view a real copy.
// CHECK-LABEL: mo.graph @reshape_producer
// CHECK-COUNT-2: map.iter.elementwise<[32]>
// PIPE-LABEL:  mo.graph @reshape_producer
// PIPE-COUNT-2: map.iter.elementwise<[32]>
mo.graph @reshape_producer(%a: !mo.tensor<[4, 8], f32>) -> (!mo.tensor<[32], f32>, !mo.tensor<[32], f32>) {
  %x = mo.static.reshape(%a) : !mo.tensor<[4, 8], f32> -> !mo.tensor<[32], f32>
  %y = mo.relu(%x) : !mo.tensor<[32], f32>
  mo.output %x, %y : !mo.tensor<[32], f32>, !mo.tensor<[32], f32>
}

// Once view fusion folds the broadcast into the mul, the consumer iterates
// over [4, 8], not the producer's [8]: no single iteration holds both.
// PIPE-LABEL:  mo.graph @broadcast_consumer
// PIPE:        %[[X:[0-9]+]] = map.iter.elementwise<[8]>
// PIPE:        %[[Y:[0-9]+]] = map.iter.elementwise<[4, 8]>
// PIPE-NEXT:   inputs (%[[X]]:
// PIPE:        mo.output %[[X]], %[[Y]]
mo.graph @broadcast_consumer(%a: !mo.tensor<[8], f32>, %b: !mo.tensor<[8], f32>, %c: !mo.tensor<[4, 8], f32>) -> (!mo.tensor<[8], f32>, !mo.tensor<[4, 8], f32>) {
  %x = mo.add(%a, %b) : !mo.tensor<[8], f32>
  %xb = mo.static.broadcast_to(%x) : !mo.tensor<[8], f32> -> !mo.tensor<[4, 8], f32>
  %y = mo.mul(%xb, %c) : !mo.tensor<[4, 8], f32>
  mo.output %x, %y : !mo.tensor<[8], f32>, !mo.tensor<[4, 8], f32>
}

// The only foreach readers sit inside the branches, in other blocks, and
// fusion never crosses a block.
// CHECK-LABEL: mo.graph @consumers_in_branches
// CHECK:       %[[X:[0-9]+]] = map.iter.elementwise<[4]>
// CHECK:       mo.if
// CHECK-NEXT:  map.iter.elementwise<[4]>
// CHECK-NEXT:  inputs (%[[X]]:
// PIPE-LABEL:  mo.graph @consumers_in_branches
// PIPE:        %[[X:[0-9]+]] = map.iter.elementwise<[4]>
// PIPE:        mo.if
// PIPE-NEXT:   map.iter.elementwise<[4]>
// PIPE-NEXT:   inputs (%[[X]]:
mo.graph @consumers_in_branches(%a: !mo.tensor<[4], f32>, %b: !mo.tensor<[4], f32>, %cond: !mo.tensor<[], bool>) -> (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) {
  %x = mo.add(%a, %b) : !mo.tensor<[4], f32>
  %t = mo.if (%cond : !mo.tensor<[], bool>) -> !mo.tensor<[4], f32> {
    %r = mo.relu(%x) : !mo.tensor<[4], f32>
    mo.yield %r : !mo.tensor<[4], f32>
  } else {
    %s = mo.sin(%x) : !mo.tensor<[4], f32>
    mo.yield %s : !mo.tensor<[4], f32>
  }
  mo.output %x, %t : !mo.tensor<[4], f32>, !mo.tensor<[4], f32>
}

// A gather reads `%x` through its prologue, so the pair stays apart: later
// fusion recomputes the broadcast add in both the relu and the gather's
// prologue, and `%x` is never stored. Merging would store it instead.
// CHECK-LABEL: mo.graph @prologue_reader
// CHECK-COUNT-2: map.iter.elementwise<[3, 3]>
// PIPE-LABEL:  mo.graph @prologue_reader
// PIPE:        %{{[0-9]+}} = map.iter.elementwise<[3, 3]>
// PIPE:        map.call.elementwise @{{.*}}.Add
// PIPE:        map.call.elementwise @{{.*}}.ReLU
// PIPE:        map.iter.opaque
// PIPE:        map.iter.prologue
// PIPE:        map.call.elementwise @{{.*}}.Add
// PIPE-NOT:    map.iter.elementwise
// PIPE:        mo.output
mo.graph @prologue_reader(%a: !mo.tensor<[3], f32>, %b: !mo.tensor<[3, 3], f32>, %idx: !mo.tensor<[2, 2], si32>) -> (!mo.tensor<[3, 3], f32>, !mo.tensor<[2, 2, 3], f32>) {
  %ab = mo.static.broadcast_to(%a) : !mo.tensor<[3], f32> -> !mo.tensor<[3, 3], f32>
  %x = mo.add(%ab, %b) : !mo.tensor<[3, 3], f32>
  %y = mo.relu(%x) : !mo.tensor<[3, 3], f32>
  %g = mo.gather(%x, %idx) [axis = 0] : (!mo.tensor<[3, 3], f32>, !mo.tensor<[2, 2], si32>) -> !mo.tensor<[2, 2, 3], f32>
  mo.output %y, %g : !mo.tensor<[3, 3], f32>, !mo.tensor<[2, 2, 3], f32>
}

// The cost of that rule: no fusion recomputes a plain add, so `%x` stays
// stored for the gather anyway, and the skipped merge leaves a reload.
// PIPE-LABEL:   mo.graph @prologue_reader_not_recomputed
// PIPE-COUNT-2: map.iter.elementwise<[3, 3]>
mo.graph @prologue_reader_not_recomputed(%a: !mo.tensor<[3, 3], f32>, %b: !mo.tensor<[3, 3], f32>, %idx: !mo.tensor<[2, 2], si32>) -> (!mo.tensor<[3, 3], f32>, !mo.tensor<[2, 2, 3], f32>) {
  %x = mo.add(%a, %b) : !mo.tensor<[3, 3], f32>
  %y = mo.relu(%x) : !mo.tensor<[3, 3], f32>
  %g = mo.gather(%x, %idx) [axis = 0] : (!mo.tensor<[3, 3], f32>, !mo.tensor<[2, 2], si32>) -> !mo.tensor<[2, 2, 3], f32>
  mo.output %y, %g : !mo.tensor<[3, 3], f32>, !mo.tensor<[2, 2, 3], f32>
}

// The first add is returned and also read by the relu, so the two merge,
// then pair with the second add and move into the matmul's epilogue: one
// kernel storing both sums and the relu.
// EPI-LABEL:   mo.graph @escaping_pair_into_epilogue
// EPI:         %[[MM:[0-9]+]]:3 = map.iter.opaque
// EPI:         %[[A:[0-9]+]] = map.call.elementwise @{{.*}}.Add
// EPI:         map.call.elementwise @{{.*}}.ReLU(%{{.*}}, %[[A]])
// EPI-COUNT-3: map.store
// EPI-NOT:     map.iter.elementwise
// EPI:         mo.output %[[MM]]#2, %[[MM]]#1, %[[MM]]#0
mo.graph @escaping_pair_into_epilogue(%x: !mo.tensor<[8, 8], f32>, %y: !mo.tensor<[8, 8], f32>, %z: !mo.tensor<[8, 8], f32>, %w: !mo.tensor<[8, 8], f32>) -> (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>) {
  %mm = mo.matmul(%x, %y) : (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 8], f32>
  %a = mo.add(%mm, %z) : !mo.tensor<[8, 8], f32>
  %b = mo.add(%mm, %w) : !mo.tensor<[8, 8], f32>
  %r = mo.relu(%a) : !mo.tensor<[8, 8], f32>
  mo.output %r, %b, %a : !mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>
}
