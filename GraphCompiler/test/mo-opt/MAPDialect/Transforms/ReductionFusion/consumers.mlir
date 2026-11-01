// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_pkg" --mo-to-map --map-fuse-reductions | FileCheck %s

// A consumer over the per-row space ([M, 1]) that reads a per-row result at
// its own index becomes row-level ops after the fold, indexed by the row. It
// reads the value the fold stored; its result becomes a rowwise output.

// The old output goes away once nothing else reads it. `%b` becomes a rowwise
// input, loaded at the row.
// CHECK-LABEL: mo.graph @chain
// CHECK-SAME:  (%[[X:.*]]: !mo.tensor<[8, 128], f32>, %[[B:.*]]: !mo.tensor<[8, 1], f32>)
// CHECK:       %[[R:.*]] = map.iter.rowwise<[8, 128], axis = 1>
// CHECK-NEXT:  inputs  (%[[X]]: !mo.tensor<[8, 128], f32>, %[[B]]: !mo.tensor<[8, 1], f32>)
// CHECK-NEXT:  outputs (%[[OUT:.*]]: !mo.tensor<[8, 1], f32>)
// CHECK-NEXT:  row     (%[[ROW:.*]]: !map.index<2>) -> !mo.tensor<[8, 1], f32> {
// CHECK-NEXT:    %[[ACC:.*]] = map.iter.reduce @builtin_kernels.monoids.ReduceSum
// CHECK:         }
// CHECK-NEXT:    %[[BV:.*]] = map.load %[[B]][%[[ROW]]] : !mo.tensor<[8, 1], f32>
// CHECK-NEXT:    %[[Y:.*]] = map.call.elementwise @builtin_kernels.elementwise.Add(%[[ROW]], %[[ACC]], %[[BV]])
// CHECK-NEXT:    %[[Z:.*]] = map.call.elementwise @builtin_kernels.elementwise.Exp(%[[ROW]], %[[Y]])
// CHECK-NEXT:    map.store %[[OUT]][%[[ROW]]] = %[[Z]] : !mo.tensor<[8, 1], f32>
// CHECK-NEXT:  }
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output %[[R]]
mo.graph @chain(%x: !mo.tensor<[8, 128], f32>, %b: !mo.tensor<[8, 1], f32>) -> !mo.tensor<[8, 1], f32> {
  %r = mo.reduce.add(%x) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  %y = mo.add(%r, %b) : !mo.tensor<[8, 1], f32>
  %z = mo.exp(%y) : !mo.tensor<[8, 1], f32>
  mo.output %z : !mo.tensor<[8, 1], f32>
}

// A result that is also a graph output stays, next to the consumer's.
// CHECK-LABEL: mo.graph @sideOutput
// CHECK:       %[[R2:.*]]:2 = map.iter.rowwise<[8, 128], axis = 1>
// CHECK:       outputs (%[[O0:.*]]: !mo.tensor<[8, 1], f32>, %[[O1:.*]]: !mo.tensor<[8, 1], f32>)
// CHECK:       %[[ACC2:.*]] = map.iter.reduce
// CHECK:       map.store %[[O0]][%[[ROW2:.*]]] = %[[ACC2]]
// CHECK-NEXT:  %[[E:.*]] = map.call.elementwise @builtin_kernels.elementwise.Exp(%[[ROW2]], %[[ACC2]])
// CHECK-NEXT:  map.store %[[O1]][%[[ROW2]]] = %[[E]]
// CHECK:       mo.output %[[R2]]#1, %[[R2]]#0
mo.graph @sideOutput(%x: !mo.tensor<[8, 128], f32>) -> (!mo.tensor<[8, 1], f32>, !mo.tensor<[8, 1], f32>) {
  %r = mo.reduce.add(%x) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  %z = mo.exp(%r) : !mo.tensor<[8, 1], f32>
  mo.output %z, %r : !mo.tensor<[8, 1], f32>, !mo.tensor<[8, 1], f32>
}

// A multi-field monoid: the consumer reads one field, whose output goes away;
// the other stays, ahead of the consumer's new output.
// CHECK-LABEL: mo.graph @oneField
// CHECK:       %[[R3:.*]]:2 = map.iter.rowwise<[4, 16], axis = 1>
// CHECK:       outputs (%[[OI:.*]]: !mo.tensor<[4, 1], si64>, %[[ON:.*]]: !mo.tensor<[4, 1], f32>)
// CHECK:       %[[VALS:.*]], %[[IDXS:.*]] = map.iter.reduce @builtin_kernels.monoids.ArgMax
// CHECK:       map.store %[[OI]][{{.*}}] = %[[IDXS]]
// CHECK-NEXT:  %[[N:.*]] = map.call.elementwise @builtin_kernels.elementwise.Negative({{.*}}, %[[VALS]])
// CHECK-NEXT:  map.store %[[ON]][{{.*}}] = %[[N]]
// CHECK:       mo.output %[[R3]]#1, %[[R3]]#0
mo.graph @oneField(%x: !mo.tensor<[4, 16], f32>) -> (!mo.tensor<[4, 1], f32>, !mo.tensor<[4, 1], si64>) {
  %v, %i = mo.custom [symbol = "mo.reduce.arg_max.tmp", device = #M.device_ref<"cpu", 0>](%x) : (!mo.tensor<[4, 16], f32>) -> (!mo.tensor<[4, 1], f32>, !mo.tensor<[4, 1], si64>)
  %n = mo.negative(%v) : !mo.tensor<[4, 1], f32>
  mo.output %n, %i : !mo.tensor<[4, 1], f32>, !mo.tensor<[4, 1], si64>
}

// The consumer's other input is computed after the reduction, so the rowwise
// moves down to it. That input's producer then fuses at the row too.
// CHECK-LABEL: mo.graph @sinksPastInput
// CHECK-SAME:  (%[[X4:.*]]: !mo.tensor<[8, 128], f32>, %[[B4:.*]]: !mo.tensor<[8, 1], f32>)
// CHECK-NOT:   map.iter.elementwise
// CHECK:       map.iter.rowwise<[8, 128], axis = 1>
// CHECK-NEXT:  inputs  (%[[X4]]: !mo.tensor<[8, 128], f32>, %[[B4]]: !mo.tensor<[8, 1], f32>)
// CHECK:       map.load %[[B4]][%[[ROW4:[[:alnum:]_]+]]]
// CHECK:       %[[BB:.*]] = map.call.elementwise @builtin_kernels.elementwise.Mul(%[[ROW4]],
// CHECK-NEXT:  map.call.elementwise @builtin_kernels.elementwise.Add(%[[ROW4]], {{.*}}, %[[BB]])
mo.graph @sinksPastInput(%x: !mo.tensor<[8, 128], f32>, %b: !mo.tensor<[8, 1], f32>) -> !mo.tensor<[8, 1], f32> {
  %r = mo.reduce.add(%x) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  %bb = mo.mul(%b, %b) : !mo.tensor<[8, 1], f32>
  %y = mo.add(%r, %bb) : !mo.tensor<[8, 1], f32>
  mo.output %y : !mo.tensor<[8, 1], f32>
}

// The consumer's other input depends on the rowwise through another kernel,
// a cycle: it stays out.
// CHECK-LABEL: mo.graph @cycle
// CHECK:       %[[R5:.*]] = map.iter.rowwise<[8, 128], axis = 1>
// CHECK:       %[[T:.*]] = map.iter.rowwise<[8, 1], axis = 0>
// CHECK-NEXT:  inputs  (%[[R5]]: !mo.tensor<[8, 1], f32>)
// CHECK:       map.iter.elementwise<[8, 1]>
// CHECK:       inputs  (%[[R5]]: !mo.tensor<[8, 1], f32>, %{{.*}}: !mo.tensor<[8, 1], f32>)
mo.graph @cycle(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32> {
  %r = mo.reduce.add(%x) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  %t = mo.reduce.max(%r) : (!mo.tensor<[8, 1], f32>) -> !mo.tensor<[1, 1], f32>
  %tb = mo.static.broadcast_to(%t) : !mo.tensor<[1, 1], f32> -> !mo.tensor<[8, 1], f32>
  %y = mo.div(%r, %tb) : !mo.tensor<[8, 1], f32>
  mo.output %y : !mo.tensor<[8, 1], f32>
}

// A consumer over the full row space is a later step: it stays out.
// CHECK-LABEL: mo.graph @fullRow
// CHECK:       %[[R6:.*]] = map.iter.rowwise<[8, 128], axis = 1>
// CHECK:       map.iter.elementwise<[8, 128]>
mo.graph @fullRow(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 128], f32> {
  %r = mo.reduce.add(%x) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  %rb = mo.static.broadcast_to(%r) : !mo.tensor<[8, 1], f32> -> !mo.tensor<[8, 128], f32>
  %d = mo.div(%x, %rb) : !mo.tensor<[8, 128], f32>
  mo.output %d : !mo.tensor<[8, 128], f32>
}

// A consumer reading the result through an index transform (here, rows in
// reverse) stays out.
// CHECK-LABEL: mo.graph @transformedRead
// CHECK:       map.iter.rowwise<[8, 8], axis = 1>
// CHECK:       map.iter.elementwise<[8, 1]>
// CHECK:       map.index.slice
mo.graph @transformedRead(%x: !mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 1], f32> {
  %r = mo.reduce.add(%x) : (!mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 1], f32>
  %start = mo.constant <#M.dense_array<7, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop = mo.constant <#M.dense_array<-9, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<-1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %s = mo.slice(%r, %start, %stop, %step) : (!mo.tensor<[8, 1], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[8, 1], f32>
  mo.output %s : !mo.tensor<[8, 1], f32>
}

// A `mo.mutable.store` of a per-row result stores into its buffer from the
// rowwise, which then writes memory no result names.
// CHECK-LABEL: mo.graph @mutableStore(
// CHECK:       map.iter.rowwise<[8, 128], axis = 1>
// CHECK-NEXT:  inputs (%{{.*}}: !mo.tensor<[8, 128], f32>, %[[BUF:.*]]: !mo.buffer<[8, 1], f32>)
// CHECK-NEXT:  outputs ()
// CHECK-NEXT:  row (%[[ROW:.*]]: !map.index<2>) -> () {
// CHECK:       %[[V:.*]] = map.call.elementwise @{{.*}}MutableStore(%[[ROW]], %{{.*}})
// CHECK-NEXT:  map.store %[[BUF]][%[[ROW]]] = %[[V]] : !mo.buffer<[8, 1], f32>
// CHECK-NEXT:  } {{{.*}}hasWriteEffect = true}
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output
mo.graph @mutableStore(%x: !mo.tensor<[8, 128], f32>, %buf: !mo.buffer<[8, 1], f32>) {
  %chain = mo.chain.create()
  %r = mo.reduce.add(%x) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  %c = mo.mutable.store(%buf, %r, %chain) : (!mo.buffer<[8, 1], f32>, !mo.tensor<[8, 1], f32>, !mo.chain) -> !mo.chain
  mo.output
}

// A store whose chain is used stays its own kernel: the rowwise holds no
// chain to carry the ordering.
// CHECK-LABEL: mo.graph @mutableStoreChainUsed
// CHECK:       map.iter.rowwise
// CHECK:       map.iter.elementwise
// CHECK:       map.call.elementwise @{{.*}}MutableStore
mo.graph @mutableStoreChainUsed(%x: !mo.tensor<[8, 128], f32>, %buf: !mo.buffer<[8, 1], f32>) -> !mo.chain {
  %chain = mo.chain.create()
  %r = mo.reduce.add(%x) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  %c = mo.mutable.store(%buf, %r, %chain) : (!mo.buffer<[8, 1], f32>, !mo.tensor<[8, 1], f32>, !mo.chain) -> !mo.chain
  mo.output %c : !mo.chain
}

// A store ordered after another store keeps its own kernel: the rowwise
// holds no chain to order it after the first store.
// CHECK-LABEL: mo.graph @mutableStoreAfterStore
// CHECK:       map.iter.rowwise
// CHECK:       map.iter.elementwise
// CHECK:       map.call.elementwise @{{.*}}MutableStore
mo.graph @mutableStoreAfterStore(%x: !mo.tensor<[8, 128], f32>, %a: !mo.tensor<[8, 1], f32>, %buf: !mo.buffer<[8, 1], f32>) {
  %c0 = mo.chain.create()
  %c1 = mo.mutable.store(%buf, %a, %c0) : (!mo.buffer<[8, 1], f32>, !mo.tensor<[8, 1], f32>, !mo.chain) -> !mo.chain
  %r = mo.reduce.add(%x) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  %c2 = mo.mutable.store(%buf, %r, %c1) : (!mo.buffer<[8, 1], f32>, !mo.tensor<[8, 1], f32>, !mo.chain) -> !mo.chain
  mo.output
}
