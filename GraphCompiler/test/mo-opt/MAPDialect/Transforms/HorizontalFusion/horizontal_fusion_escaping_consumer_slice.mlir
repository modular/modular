// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-fuse-kernels | FileCheck %s --check-prefixes=CHECK,PIPE
// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-fuse-views --map-fuse-horizontal --map-fuse-epilogue | FileCheck %s
// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-fuse-kernels --map-outline-structured-kernel-bodies | FileCheck %s --check-prefix=OUTL

// Every RUN line folds the slice into its consumer with view fusion first, which
// is the form the merge matches; on its own, `--map-fuse-horizontal` would see
// the slice as a separate copy kernel.

// An elementwise producer whose result is returned, and read by one consumer
// through a strided slice: the consumer joins the producer's kernel inside a
// `map.index.slice.select`, so it runs only where the slice selects the element
// and stores at the slice's index. The slice's bounds stop being inputs, since
// the select carries them statically.

// A step on an outer dim. Outlining moves the body, select included, into a
// `map.iter.elementwise.body`, where the select indexes the leading argument.
// OUTL-LABEL:  mo.graph @step_escapes
// OUTL:        map.iter.elementwise.body<[4, 8, 16]>
// OUTL-NEXT:   index (%[[I:[a-z0-9]+]]: !map.index<3>)
// OUTL:        map.index.slice.select<{{.*}}> (%[[I]]) -> (%[[J:[a-z0-9]+]])
// OUTL-NEXT:   map.call.elementwise @{{.*}}.ReLU(%[[J]],
// CHECK-LABEL: mo.graph @step_escapes
// CHECK:       %[[OUT:[0-9]+]]:2 = map.iter.elementwise<[4, 8, 16]>
// CHECK-NEXT:  inputs  (%{{[a-z0-9]+}}: !mo.tensor<[4, 8, 16], f32>, %{{[a-z0-9]+}}: !mo.tensor<[4, 8, 16], f32>)
// CHECK-NEXT:  outputs (%[[X:[a-z0-9]+]]: {{.*}}, %[[Y:[a-z0-9]+]]: !mo.tensor<[4, 4, 16], f32>)
// CHECK-NEXT:  index   (%[[I:[a-z0-9]+]]: !map.index<3>)
// CHECK:       %[[ADD:[0-9]+]] = map.call.elementwise @{{.*}}.Add
// CHECK-NEXT:  map.store %[[X]][%[[I]]] = %[[ADD]]
// CHECK-NEXT:  map.index.slice.select<#map.index.slice.spec<[4, 8, 16] -> [4, 4, 16], starts = [0, 1, 0], stops = [4, 8, 16], steps = [1, 2, 1]>> (%[[I]]) -> (%[[J:[a-z0-9]+]])
// CHECK-NEXT:  %[[RELU:[0-9]+]] = map.call.elementwise @{{.*}}.ReLU(%[[J]], %[[ADD]])
// CHECK-NEXT:  map.store %[[Y]][%[[J]]] = %[[RELU]]
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output %[[OUT]]#0, %[[OUT]]#1
mo.graph @step_escapes(%a: !mo.tensor<[4, 8, 16], f32>, %b: !mo.tensor<[4, 8, 16], f32>) -> (!mo.tensor<[4, 8, 16], f32>, !mo.tensor<[4, 4, 16], f32>) {
  %starts = mo.constant <#M.dense_array<0, 1, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %stops = mo.constant <#M.dense_array<4, 8, 16> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %steps = mo.constant <#M.dense_array<1, 2, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %x = mo.add(%a, %b) : !mo.tensor<[4, 8, 16], f32>
  %w = mo.slice(%x, %starts, %stops, %steps) : (!mo.tensor<[4, 8, 16], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 4, 16], f32>
  %y = mo.relu(%w) : !mo.tensor<[4, 4, 16], f32>
  mo.output %x, %y : !mo.tensor<[4, 8, 16], f32>, !mo.tensor<[4, 4, 16], f32>
}

// A window of a middle dim: strided, since rows of the outer dim lie between
// its runs.
// CHECK-LABEL: mo.graph @window_escapes
// CHECK:       %[[OUT:[0-9]+]]:2 = map.iter.elementwise<[4, 8, 16]>
// CHECK-NEXT:  inputs  (%{{[a-z0-9]+}}: !mo.tensor<[4, 8, 16], f32>, %{{[a-z0-9]+}}: !mo.tensor<[4, 8, 16], f32>)
// CHECK-NEXT:  outputs (%[[X:[a-z0-9]+]]: {{.*}}, %[[Y:[a-z0-9]+]]: !mo.tensor<[4, 4, 16], f32>)
// CHECK-NEXT:  index   (%[[I:[a-z0-9]+]]: !map.index<3>)
// CHECK:       %[[ADD:[0-9]+]] = map.call.elementwise @{{.*}}.Add
// CHECK-NEXT:  map.store %[[X]][%[[I]]] = %[[ADD]]
// CHECK-NEXT:  map.index.slice.select<#map.index.slice.spec<[4, 8, 16] -> [4, 4, 16], starts = [0, 4, 0], stops = [4, 8, 16], steps = [1, 1, 1]>> (%[[I]]) -> (%[[J:[a-z0-9]+]])
// CHECK-NEXT:  %[[RELU:[0-9]+]] = map.call.elementwise @{{.*}}.ReLU(%[[J]], %[[ADD]])
// CHECK-NEXT:  map.store %[[Y]][%[[J]]] = %[[RELU]]
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output %[[OUT]]#0, %[[OUT]]#1
mo.graph @window_escapes(%a: !mo.tensor<[4, 8, 16], f32>, %b: !mo.tensor<[4, 8, 16], f32>) -> (!mo.tensor<[4, 8, 16], f32>, !mo.tensor<[4, 4, 16], f32>) {
  %starts = mo.constant <#M.dense_array<0, 4, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %stops = mo.constant <#M.dense_array<4, 8, 16> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %steps = mo.constant <#M.dense_array<1, 1, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %x = mo.add(%a, %b) : !mo.tensor<[4, 8, 16], f32>
  %w = mo.slice(%x, %starts, %stops, %steps) : (!mo.tensor<[4, 8, 16], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 4, 16], f32>
  %y = mo.relu(%w) : !mo.tensor<[4, 4, 16], f32>
  mo.output %x, %y : !mo.tensor<[4, 8, 16], f32>, !mo.tensor<[4, 4, 16], f32>
}

// A negative start counts from the end of its dim, like numpy's; the select
// holds it resolved.
// CHECK-LABEL: mo.graph @negative_start
// CHECK:       %[[OUT:[0-9]+]]:2 = map.iter.elementwise<[4, 8, 16]>
// CHECK:       map.index.slice.select<#map.index.slice.spec<[4, 8, 16] -> [4, 4, 16], starts = [0, 4, 0]
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output %[[OUT]]#0, %[[OUT]]#1
mo.graph @negative_start(%a: !mo.tensor<[4, 8, 16], f32>, %b: !mo.tensor<[4, 8, 16], f32>) -> (!mo.tensor<[4, 8, 16], f32>, !mo.tensor<[4, 4, 16], f32>) {
  %starts = mo.constant <#M.dense_array<0, -4, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %stops = mo.constant <#M.dense_array<4, 8, 16> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %steps = mo.constant <#M.dense_array<1, 1, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %x = mo.add(%a, %b) : !mo.tensor<[4, 8, 16], f32>
  %w = mo.slice(%x, %starts, %stops, %steps) : (!mo.tensor<[4, 8, 16], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 4, 16], f32>
  %y = mo.relu(%w) : !mo.tensor<[4, 4, 16], f32>
  mo.output %x, %y : !mo.tensor<[4, 8, 16], f32>, !mo.tensor<[4, 4, 16], f32>
}

// Two siblings pair first; the relu joins them through a select, and the sin
// then merges into that kernel, select and all.
// CHECK-LABEL: mo.graph @select_kernel_merges_again
// CHECK:       %[[OUT:[0-9]+]]:4 = map.iter.elementwise<[4, 8, 16]>
// CHECK-NEXT:  inputs
// CHECK-NEXT:  outputs (%[[X:[a-z0-9]+]]: {{.*}}, %[[Q:[a-z0-9]+]]: {{.*}}, %[[Y:[a-z0-9]+]]: {{.*}}, %[[R:[a-z0-9]+]]: {{.*}})
// CHECK:       %[[ADD:[0-9]+]] = map.call.elementwise @{{.*}}.Add
// CHECK:       %[[SUB:[0-9]+]] = map.call.elementwise @{{.*}}.Sub
// CHECK:       map.index.slice.select<{{.*}}> (%{{.*}}) -> (%[[J:[a-z0-9]+]])
// CHECK-NEXT:  %[[RELU:[0-9]+]] = map.call.elementwise @{{.*}}.ReLU(%[[J]], %[[ADD]])
// CHECK-NEXT:  map.store %[[Y]][%[[J]]] = %[[RELU]]
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[SIN:[0-9]+]] = map.call.elementwise @{{.*}}.Sin(%{{.*}}, %[[SUB]])
// CHECK-NEXT:  map.store %[[R]][%{{.*}}] = %[[SIN]]
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output %[[OUT]]#0, %[[OUT]]#1, %[[OUT]]#2, %[[OUT]]#3
mo.graph @select_kernel_merges_again(%a: !mo.tensor<[4, 8, 16], f32>, %b: !mo.tensor<[4, 8, 16], f32>, %c: !mo.tensor<[4, 8, 16], f32>) -> (!mo.tensor<[4, 8, 16], f32>, !mo.tensor<[4, 8, 16], f32>, !mo.tensor<[4, 4, 16], f32>, !mo.tensor<[4, 8, 16], f32>) {
  %starts = mo.constant <#M.dense_array<0, 1, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %stops = mo.constant <#M.dense_array<4, 8, 16> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %steps = mo.constant <#M.dense_array<1, 2, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %x = mo.add(%a, %b) : !mo.tensor<[4, 8, 16], f32>
  %q = mo.sub(%a, %c) : !mo.tensor<[4, 8, 16], f32>
  %w = mo.slice(%x, %starts, %stops, %steps) : (!mo.tensor<[4, 8, 16], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 4, 16], f32>
  %y = mo.relu(%w) : !mo.tensor<[4, 4, 16], f32>
  %r = mo.sin(%q) : !mo.tensor<[4, 8, 16], f32>
  mo.output %x, %q, %y, %r : !mo.tensor<[4, 8, 16], f32>, !mo.tensor<[4, 8, 16], f32>, !mo.tensor<[4, 4, 16], f32>, !mo.tensor<[4, 8, 16], f32>
}

// Two outer dims, each bounded on both ends.
// CHECK-LABEL: mo.graph @two_dims_escape
// CHECK:       %[[OUT:[0-9]+]]:2 = map.iter.elementwise<[4, 8, 16]>
// CHECK-NEXT:  inputs  (%{{[a-z0-9]+}}: !mo.tensor<[4, 8, 16], f32>, %{{[a-z0-9]+}}: !mo.tensor<[4, 8, 16], f32>)
// CHECK-NEXT:  outputs (%[[X:[a-z0-9]+]]: {{.*}}, %[[Y:[a-z0-9]+]]: !mo.tensor<[2, 4, 16], f32>)
// CHECK-NEXT:  index   (%[[I:[a-z0-9]+]]: !map.index<3>)
// CHECK:       %[[ADD:[0-9]+]] = map.call.elementwise @{{.*}}.Add
// CHECK-NEXT:  map.store %[[X]][%[[I]]] = %[[ADD]]
// CHECK-NEXT:  map.index.slice.select<#map.index.slice.spec<[4, 8, 16] -> [2, 4, 16], starts = [1, 2, 0], stops = [3, 6, 16], steps = [1, 1, 1]>> (%[[I]]) -> (%[[J:[a-z0-9]+]])
// CHECK-NEXT:  %[[RELU:[0-9]+]] = map.call.elementwise @{{.*}}.ReLU(%[[J]], %[[ADD]])
// CHECK-NEXT:  map.store %[[Y]][%[[J]]] = %[[RELU]]
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output %[[OUT]]#0, %[[OUT]]#1
mo.graph @two_dims_escape(%a: !mo.tensor<[4, 8, 16], f32>, %b: !mo.tensor<[4, 8, 16], f32>) -> (!mo.tensor<[4, 8, 16], f32>, !mo.tensor<[2, 4, 16], f32>) {
  %starts = mo.constant <#M.dense_array<1, 2, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %stops = mo.constant <#M.dense_array<3, 6, 16> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %steps = mo.constant <#M.dense_array<1, 1, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %x = mo.add(%a, %b) : !mo.tensor<[4, 8, 16], f32>
  %w = mo.slice(%x, %starts, %stops, %steps) : (!mo.tensor<[4, 8, 16], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[2, 4, 16], f32>
  %y = mo.relu(%w) : !mo.tensor<[2, 4, 16], f32>
  mo.output %x, %y : !mo.tensor<[4, 8, 16], f32>, !mo.tensor<[2, 4, 16], f32>
}

// The cases below stay two kernels, the consumer reading the producer's result
// through the slice.

// A contiguous window is an offset view, better left to view fusion than paid
// for with a select.
// CHECK-LABEL: mo.graph @contiguous_window
// CHECK-NOT:   map.index.slice.select
// CHECK:       map.iter.elementwise<[16, 64]>
// CHECK-NOT:   map.index.slice.select
// CHECK:       map.iter.elementwise
// CHECK-NOT:   map.index.slice.select
// CHECK:       map.index.slice<
// CHECK-NOT:   map.index.slice.select
// CHECK:       mo.output
mo.graph @contiguous_window(%a: !mo.tensor<[16, 64], f32>, %b: !mo.tensor<[16, 64], f32>) -> (!mo.tensor<[16, 64], f32>, !mo.tensor<[8, 64], f32>) {
  %starts = mo.constant <#M.dense_array<8, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stops = mo.constant <#M.dense_array<16, 64> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %steps = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %x = mo.add(%a, %b) : !mo.tensor<[16, 64], f32>
  %w = mo.slice(%x, %starts, %stops, %steps) : (!mo.tensor<[16, 64], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[8, 64], f32>
  %y = mo.relu(%w) : !mo.tensor<[8, 64], f32>
  mo.output %x, %y : !mo.tensor<[16, 64], f32>, !mo.tensor<[8, 64], f32>
}

// The store is vectorized along the innermost dim, so a slice of it can't be
// selected per element.
// CHECK-LABEL: mo.graph @inner_dim
// CHECK-NOT:   map.index.slice.select
// CHECK:       map.iter.elementwise<[4, 8, 16]>
// CHECK-NOT:   map.index.slice.select
// CHECK:       map.iter.elementwise
// CHECK-NOT:   map.index.slice.select
// CHECK:       map.index.slice<
// CHECK-NOT:   map.index.slice.select
// CHECK:       mo.output
mo.graph @inner_dim(%a: !mo.tensor<[4, 8, 16], f32>, %b: !mo.tensor<[4, 8, 16], f32>) -> (!mo.tensor<[4, 8, 16], f32>, !mo.tensor<[4, 8, 8], f32>) {
  %starts = mo.constant <#M.dense_array<0, 0, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %stops = mo.constant <#M.dense_array<4, 8, 8> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %steps = mo.constant <#M.dense_array<1, 1, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %x = mo.add(%a, %b) : !mo.tensor<[4, 8, 16], f32>
  %w = mo.slice(%x, %starts, %stops, %steps) : (!mo.tensor<[4, 8, 16], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 8, 8], f32>
  %y = mo.relu(%w) : !mo.tensor<[4, 8, 8], f32>
  mo.output %x, %y : !mo.tensor<[4, 8, 16], f32>, !mo.tensor<[4, 8, 8], f32>
}

// A reversed slice is not one the select takes.
// CHECK-LABEL: mo.graph @neg_step
// CHECK-NOT:   map.index.slice.select
// CHECK:       map.iter.elementwise<[4, 8, 16]>
// CHECK-NOT:   map.index.slice.select
// CHECK:       map.iter.elementwise
// CHECK-NOT:   map.index.slice.select
// CHECK:       map.index.slice<
// CHECK-NOT:   map.index.slice.select
// CHECK:       mo.output
mo.graph @neg_step(%a: !mo.tensor<[4, 8, 16], f32>, %b: !mo.tensor<[4, 8, 16], f32>) -> (!mo.tensor<[4, 8, 16], f32>, !mo.tensor<[4, 8, 16], f32>) {
  %starts = mo.constant <#M.dense_array<0, 7, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %stops = mo.constant <#M.dense_array<4, -9, 16> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %steps = mo.constant <#M.dense_array<1, -1, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %x = mo.add(%a, %b) : !mo.tensor<[4, 8, 16], f32>
  %w = mo.slice(%x, %starts, %stops, %steps) : (!mo.tensor<[4, 8, 16], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 8, 16], f32>
  %y = mo.relu(%w) : !mo.tensor<[4, 8, 16], f32>
  mo.output %x, %y : !mo.tensor<[4, 8, 16], f32>, !mo.tensor<[4, 8, 16], f32>
}

// Runtime starts and steps leave the select no static bounds to carry.
// CHECK-LABEL: mo.graph @dynamic_bounds
// CHECK-NOT:   map.index.slice.select
// CHECK:       map.iter.elementwise<[4, 8, 16]>
// CHECK-NOT:   map.index.slice.select
// CHECK:       map.iter.elementwise
// CHECK-NOT:   map.index.slice.select
// CHECK:       map.index.slice<
// CHECK-NOT:   map.index.slice.select
// CHECK:       mo.output
mo.graph @dynamic_bounds(%a: !mo.tensor<[4, 8, 16], f32>, %b: !mo.tensor<[4, 8, 16], f32>, %starts: !mo.tensor<[3], si64>, %stops: !mo.tensor<[3], si64>, %steps: !mo.tensor<[3], si64>) -> (!mo.tensor<[4, 8, 16], f32>, !mo.tensor<[4, 4, 16], f32>) {
  %x = mo.add(%a, %b) : !mo.tensor<[4, 8, 16], f32>
  %w = mo.slice(%x, %starts, %stops, %steps) : (!mo.tensor<[4, 8, 16], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 4, 16], f32>
  %y = mo.relu(%w) : !mo.tensor<[4, 4, 16], f32>
  mo.output %x, %y : !mo.tensor<[4, 8, 16], f32>, !mo.tensor<[4, 4, 16], f32>
}

// A dynamic outer dim merges too: the select reads its runtime shapes, which
// the foreach lowering builds in the kernel scope.
// CHECK-LABEL: mo.graph @dynamic_shape
// CHECK:       %[[OUT:[0-9]+]]:2 = map.iter.elementwise<[n, 8, 16]>
// CHECK:       map.index.slice.select<#map.index.slice.spec<[n, 8, 16] -> [n, 4, 16]
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output %[[OUT]]#0, %[[OUT]]#1
mo.graph @dynamic_shape<n>(%a: !mo.tensor<[n, 8, 16], f32>, %b: !mo.tensor<[n, 8, 16], f32>) -> (!mo.tensor<[n, 8, 16], f32>, !mo.tensor<[n, 4, 16], f32>) {
  %starts = mo.constant <#M.dense_array<0, 1, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %stops = mo.constant <#M.dense_array<1073741824, 8, 16> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %steps = mo.constant <#M.dense_array<1, 2, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %x = mo.add(%a, %b) : !mo.tensor<[n, 8, 16], f32>
  %w = mo.slice(%x, %starts, %stops, %steps) : (!mo.tensor<[n, 8, 16], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[n, 4, 16], f32>
  %y = mo.relu(%w) : !mo.tensor<[n, 4, 16], f32>
  mo.output %x, %y : !mo.tensor<[n, 8, 16], f32>, !mo.tensor<[n, 4, 16], f32>
}

// Once elementwise fusion folds the second add into the relu, the consumer
// also reads `%c`, which the merge doesn't take yet.
// CHECK-LABEL: mo.graph @other_input
// PIPE-NOT:    map.index.slice.select
// PIPE:        map.iter.elementwise<[4, 8, 16]>
// PIPE-NOT:    map.index.slice.select
// PIPE:        map.iter.elementwise
// PIPE-NOT:    map.index.slice.select
// PIPE:        map.index.slice<
// PIPE-NOT:    map.index.slice.select
// PIPE:        mo.output
mo.graph @other_input(%a: !mo.tensor<[4, 8, 16], f32>, %b: !mo.tensor<[4, 8, 16], f32>, %c: !mo.tensor<[4, 4, 16], f32>) -> (!mo.tensor<[4, 8, 16], f32>, !mo.tensor<[4, 4, 16], f32>) {
  %starts = mo.constant <#M.dense_array<0, 1, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %stops = mo.constant <#M.dense_array<4, 8, 16> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %steps = mo.constant <#M.dense_array<1, 2, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %x = mo.add(%a, %b) : !mo.tensor<[4, 8, 16], f32>
  %w = mo.slice(%x, %starts, %stops, %steps) : (!mo.tensor<[4, 8, 16], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 4, 16], f32>
  %r = mo.relu(%w) : !mo.tensor<[4, 4, 16], f32>
  %y = mo.add(%r, %c) : !mo.tensor<[4, 4, 16], f32>
  mo.output %x, %y : !mo.tensor<[4, 8, 16], f32>, !mo.tensor<[4, 4, 16], f32>
}

// Without another reader there is nothing to merge for: the result needn't be
// stored at all.
// CHECK-LABEL: mo.graph @sole_reader
// CHECK-NOT:   map.index.slice.select
// CHECK:       mo.output
mo.graph @sole_reader(%a: !mo.tensor<[4, 8, 16], f32>, %b: !mo.tensor<[4, 8, 16], f32>) -> !mo.tensor<[4, 4, 16], f32> {
  %starts = mo.constant <#M.dense_array<0, 1, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %stops = mo.constant <#M.dense_array<4, 8, 16> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %steps = mo.constant <#M.dense_array<1, 2, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %x = mo.add(%a, %b) : !mo.tensor<[4, 8, 16], f32>
  %w = mo.slice(%x, %starts, %stops, %steps) : (!mo.tensor<[4, 8, 16], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 4, 16], f32>
  %y = mo.relu(%w) : !mo.tensor<[4, 4, 16], f32>
  mo.output %y : !mo.tensor<[4, 4, 16], f32>
}

// The add moves into the matmul's epilogue and the relu follows it there
// inside a select. Merged first (the second RUN line), the pair moves in together,
// its select kept at the epilogue's index.
// CHECK-LABEL: mo.graph @matmul_feeds_merge
// CHECK:       %[[MM:[0-9]+]]:2 = map.iter.opaque
// CHECK:       map.iter.epilogue
// CHECK:       %[[ADD:[0-9]+]] = map.call.elementwise @{{.*}}.Add
// CHECK:       map.index.slice.select<{{.*}}> (%idx) -> (%[[J:[a-z0-9]+]])
// CHECK-NEXT:  map.call.elementwise @{{.*}}.ReLU(%[[J]], %[[ADD]])
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output %[[MM]]#0, %[[MM]]#1
mo.graph @matmul_feeds_merge(%p: !mo.tensor<[8, 8], f32>, %q: !mo.tensor<[8, 16], f32>, %b: !mo.tensor<[8, 16], f32>) -> (!mo.tensor<[8, 16], f32>, !mo.tensor<[4, 16], f32>) {
  %starts = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stops = mo.constant <#M.dense_array<8, 16> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %steps = mo.constant <#M.dense_array<2, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %mm = mo.matmul(%p, %q) : (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 16], f32>) -> !mo.tensor<[8, 16], f32>
  %x = mo.add(%mm, %b) : !mo.tensor<[8, 16], f32>
  %w = mo.slice(%x, %starts, %stops, %steps) : (!mo.tensor<[8, 16], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[4, 16], f32>
  %y = mo.relu(%w) : !mo.tensor<[4, 16], f32>
  mo.output %x, %y : !mo.tensor<[8, 16], f32>, !mo.tensor<[4, 16], f32>
}

// The adds pair as siblings and the relu joins them through a select before
// output fusion sees the pair as the matmul's one reader; the whole kernel
// still moves into the epilogue, so the matmul's result is never stored.
// PIPE-LABEL:  mo.graph @pair_and_slice
// PIPE:        %[[MM:[0-9]+]]:3 = map.iter.opaque
// PIPE:        map.iter.epilogue
// PIPE-COUNT-2: map.call.elementwise @{{.*}}.Add
// PIPE:        map.index.slice.select<{{.*}}> (%idx)
// PIPE-NOT:    map.iter.elementwise
// PIPE:        mo.output %[[MM]]#0, %[[MM]]#1, %[[MM]]#2
mo.graph @pair_and_slice(%p: !mo.tensor<[8, 8], f32>, %q: !mo.tensor<[8, 16], f32>, %b: !mo.tensor<[8, 16], f32>, %c: !mo.tensor<[8, 16], f32>) -> (!mo.tensor<[8, 16], f32>, !mo.tensor<[8, 16], f32>, !mo.tensor<[4, 16], f32>) {
  %starts = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stops = mo.constant <#M.dense_array<8, 16> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %steps = mo.constant <#M.dense_array<2, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %mm = mo.matmul(%p, %q) : (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 16], f32>) -> !mo.tensor<[8, 16], f32>
  %x = mo.add(%mm, %b) : !mo.tensor<[8, 16], f32>
  %z = mo.add(%mm, %c) : !mo.tensor<[8, 16], f32>
  %w = mo.slice(%x, %starts, %stops, %steps) : (!mo.tensor<[8, 16], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[4, 16], f32>
  %y = mo.relu(%w) : !mo.tensor<[4, 16], f32>
  mo.output %x, %z, %y : !mo.tensor<[8, 16], f32>, !mo.tensor<[8, 16], f32>, !mo.tensor<[4, 16], f32>
}
