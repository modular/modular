// RUN: mo-opt --mlir-print-local-scope %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-kernels | FileCheck %s

// A concat is a loop reading its inputs through `map.index.concat.select`. A
// reduction that is its only reader fuses it like any producer: the select
// moves into the fold or the map phase, at the phase's index, and the concat
// isn't materialized.

// Over the concat axis, the innermost one.
// CHECK-LABEL: mo.graph @sum_of_concat
// CHECK-NOT:   map.iter.elementwise
// CHECK:       map.iter.rowwise<[4, 16], axis = 1>
// CHECK:       map.iter.reduce @{{.*}}ReduceSum index (%[[I:.*]]: !map.index<2>)
// CHECK:       map.index.concat.select<#map.index.concat.spec<[4, 6], [4, 10] -> [4, 16], axis = 1>> (%[[I]])
// CHECK:       mo.output
mo.graph @sum_of_concat(%a: !mo.tensor<[4, 6], f32>, %b: !mo.tensor<[4, 10], f32>) -> !mo.tensor<[4, 1], f32> {
  %c = mo.concat[1] (%a, %b) : (!mo.tensor<[4, 6], f32>, !mo.tensor<[4, 10], f32>) -> !mo.tensor<[4, 16], f32>
  %r = mo.reduce.add(%c) : (!mo.tensor<[4, 16], f32>) -> !mo.tensor<[4, 1], f32>
  mo.output %r : !mo.tensor<[4, 1], f32>
}

// Over the concat axis, an outer one.
// CHECK-LABEL: mo.graph @max_over_outer_concat_axis
// CHECK-NOT:   map.iter.elementwise
// CHECK:       map.iter.rowwise<[8, 8], axis = 0>
// CHECK:       map.iter.reduce @{{.*}}ReduceMax index (%[[I:.*]]: !map.index<2>)
// CHECK:       map.index.concat.select<#map.index.concat.spec<[3, 8], [5, 8] -> [8, 8], axis = 0>> (%[[I]])
mo.graph @max_over_outer_concat_axis(%a: !mo.tensor<[3, 8], f32>, %b: !mo.tensor<[5, 8], f32>) -> !mo.tensor<[1, 8], f32> {
  %c = mo.concat[0] (%a, %b) : (!mo.tensor<[3, 8], f32>, !mo.tensor<[5, 8], f32>) -> !mo.tensor<[8, 8], f32>
  %r = mo.reduce.max(%c) : (!mo.tensor<[8, 8], f32>) -> !mo.tensor<[1, 8], f32>
  mo.output %r : !mo.tensor<[1, 8], f32>
}

// Across the concat axis: each row is one concat row.
// CHECK-LABEL: mo.graph @min_across_concat_axis
// CHECK-NOT:   map.iter.elementwise
// CHECK:       map.iter.rowwise<[8, 8], axis = 1>
// CHECK:       map.index.concat.select
mo.graph @min_across_concat_axis(%a: !mo.tensor<[3, 8], f32>, %b: !mo.tensor<[5, 8], f32>) -> !mo.tensor<[8, 1], f32> {
  %c = mo.concat[0] (%a, %b) : (!mo.tensor<[3, 8], f32>, !mo.tensor<[5, 8], f32>) -> !mo.tensor<[8, 8], f32>
  %r = mo.reduce.min(%c) : (!mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 1], f32>
  mo.output %r : !mo.tensor<[8, 1], f32>
}

// An elementwise op on the concat fuses too, around the select.
// CHECK-LABEL: mo.graph @sum_of_squared_concat
// CHECK-NOT:   map.iter.elementwise
// CHECK:       map.iter.reduce
// CHECK:       map.index.concat.select
// CHECK:       map.call.elementwise @{{.*}}Mul
mo.graph @sum_of_squared_concat(%a: !mo.tensor<[4, 6], f32>, %b: !mo.tensor<[4, 10], f32>) -> !mo.tensor<[4, 1], f32> {
  %c = mo.concat[1] (%a, %b) : (!mo.tensor<[4, 6], f32>, !mo.tensor<[4, 10], f32>) -> !mo.tensor<[4, 16], f32>
  %sq = mo.mul(%c, %c) : !mo.tensor<[4, 16], f32>
  %r = mo.reduce.add(%sq) : (!mo.tensor<[4, 16], f32>) -> !mo.tensor<[4, 1], f32>
  mo.output %r : !mo.tensor<[4, 1], f32>
}

// The concat feeds both the reduction's fold and its map phase (the add): each
// gets a copy of the select, at its own index.
// CHECK-LABEL: mo.graph @concat_plus_its_sum
// CHECK-NOT:   map.iter.elementwise
// CHECK:       map.iter.rowwise<[4, 16], axis = 1>
// CHECK:       map.iter.reduce @{{.*}}ReduceSum index (%[[I:.*]]: !map.index<2>)
// CHECK:       map.index.concat.select<{{.*}}> (%[[I]])
// CHECK:       map.iter.map index (%[[J:.*]]: !map.index<2>)
// CHECK:       map.index.concat.select<{{.*}}> (%[[J]])
// CHECK:       mo.output
mo.graph @concat_plus_its_sum(%a: !mo.tensor<[4, 6], f32>, %b: !mo.tensor<[4, 10], f32>) -> !mo.tensor<[4, 16], f32> {
  %c = mo.concat[1] (%a, %b) : (!mo.tensor<[4, 6], f32>, !mo.tensor<[4, 10], f32>) -> !mo.tensor<[4, 16], f32>
  %s = mo.reduce.add(%c) : (!mo.tensor<[4, 16], f32>) -> !mo.tensor<[4, 1], f32>
  %sb = mo.static.broadcast_to(%s) : !mo.tensor<[4, 1], f32> -> !mo.tensor<[4, 16], f32>
  %y = mo.add(%c, %sb) : !mo.tensor<[4, 16], f32>
  mo.output %y : !mo.tensor<[4, 16], f32>
}

// The fold reads the concat through a transpose, at an index the transpose
// computes rather than the fold's own: the select can't pick its input per
// vector there, so the concat stays a kernel of its own.
// CHECK-LABEL: mo.graph @sum_of_transposed_concat
// CHECK:       %[[C:.*]] = map.iter.elementwise
// CHECK:       map.index.concat.select
// CHECK:       map.iter.rowwise<[16, 4], axis = 1>
// CHECK-NOT:   map.index.concat.select
// CHECK:       mo.output
mo.graph @sum_of_transposed_concat(%a: !mo.tensor<[4, 6], f32>, %b: !mo.tensor<[4, 10], f32>) -> !mo.tensor<[16, 1], f32> {
  %perm = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %c = mo.concat[1] (%a, %b) : (!mo.tensor<[4, 6], f32>, !mo.tensor<[4, 10], f32>) -> !mo.tensor<[4, 16], f32>
  %t = mo.transpose(%c, %perm) : (!mo.tensor<[4, 16], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[16, 4], f32>
  %r = mo.reduce.add(%t) : (!mo.tensor<[16, 4], f32>) -> !mo.tensor<[16, 1], f32>
  mo.output %r : !mo.tensor<[16, 1], f32>
}

// x / sum(x): the same map phase as the add above, as a softmax-like
// normalization writes it.
// CHECK-LABEL: mo.graph @concat_over_its_sum
// CHECK-NOT:   map.iter.elementwise
// CHECK:       map.iter.reduce
// CHECK:       map.index.concat.select
// CHECK:       map.iter.map
// CHECK:       map.index.concat.select
// CHECK:       map.call.elementwise @{{.*}}Div
// CHECK:       mo.output
mo.graph @concat_over_its_sum(%a: !mo.tensor<[4, 6], f32>, %b: !mo.tensor<[4, 10], f32>) -> !mo.tensor<[4, 16], f32> {
  %c = mo.concat[1] (%a, %b) : (!mo.tensor<[4, 6], f32>, !mo.tensor<[4, 10], f32>) -> !mo.tensor<[4, 16], f32>
  %s = mo.reduce.add(%c) : (!mo.tensor<[4, 16], f32>) -> !mo.tensor<[4, 1], f32>
  %sb = mo.static.broadcast_to(%s) : !mo.tensor<[4, 1], f32> -> !mo.tensor<[4, 16], f32>
  %y = mo.div(%c, %sb) : !mo.tensor<[4, 16], f32>
  mo.output %y : !mo.tensor<[4, 16], f32>
}

// A consumer of a per-row result that also reads a concat: the select can't
// run at the row index, so the consumer stays a kernel of its own.
// CHECK-LABEL: mo.graph @per_row_sum_plus_concat
// CHECK:       map.iter.rowwise<[4, 16], axis = 1>
// CHECK-NOT:   map.index.concat.select
// CHECK:       map.iter.elementwise<[4, 1]>
// CHECK:       map.index.concat.select<#map.index.concat.spec<[1, 1], [3, 1] -> [4, 1], axis = 0>>
// CHECK:       mo.output
mo.graph @per_row_sum_plus_concat(%x: !mo.tensor<[4, 16], f32>, %a: !mo.tensor<[1, 1], f32>, %b: !mo.tensor<[3, 1], f32>) -> !mo.tensor<[4, 1], f32> {
  %s = mo.reduce.add(%x) : (!mo.tensor<[4, 16], f32>) -> !mo.tensor<[4, 1], f32>
  %c = mo.concat[0] (%a, %b) : (!mo.tensor<[1, 1], f32>, !mo.tensor<[3, 1], f32>) -> !mo.tensor<[4, 1], f32>
  %y = mo.add(%s, %c) : !mo.tensor<[4, 1], f32>
  mo.output %y : !mo.tensor<[4, 1], f32>
}

// The same with a consumer over the rowwise shape: it joins a map phase,
// select and all.
// CHECK-LABEL: mo.graph @concat_times_broadcast_sum
// CHECK-NOT:   map.iter.elementwise
// CHECK:       map.iter.rowwise<[4, 16], axis = 1>
// CHECK:       map.iter.map index (%[[J:.*]]: !map.index<2>)
// CHECK:       map.index.concat.select<#map.index.concat.spec<[4, 6], [4, 10] -> [4, 16], axis = 1>> (%[[J]])
// CHECK:       mo.output
mo.graph @concat_times_broadcast_sum(%x: !mo.tensor<[4, 16], f32>, %a: !mo.tensor<[4, 6], f32>, %b: !mo.tensor<[4, 10], f32>) -> !mo.tensor<[4, 16], f32> {
  %s = mo.reduce.add(%x) : (!mo.tensor<[4, 16], f32>) -> !mo.tensor<[4, 1], f32>
  %sb = mo.static.broadcast_to(%s) : !mo.tensor<[4, 1], f32> -> !mo.tensor<[4, 16], f32>
  %c = mo.concat[1] (%a, %b) : (!mo.tensor<[4, 6], f32>, !mo.tensor<[4, 10], f32>) -> !mo.tensor<[4, 16], f32>
  %y = mo.mul(%c, %sb) : !mo.tensor<[4, 16], f32>
  mo.output %y : !mo.tensor<[4, 16], f32>
}

// RMS norm of a concat: view fusion gives the square and the scale each a
// copy of the select. The square fuses into the fold, the scale into the map
// phase.
// CHECK-LABEL: mo.graph @rms_norm_of_concat
// CHECK-NOT:   map.iter.elementwise
// CHECK:       map.iter.rowwise<[4, 16], axis = 1>
// CHECK:       map.iter.reduce
// CHECK:       map.index.concat.select
// CHECK:       map.iter.map
// CHECK:       map.index.concat.select<#map.index.concat.spec<[4, 6], [4, 10] -> [4, 16], axis = 1>>
// CHECK:       mo.output
mo.graph @rms_norm_of_concat(%a: !mo.tensor<[4, 6], f32>, %b: !mo.tensor<[4, 10], f32>) -> !mo.tensor<[4, 16], f32> {
  %c = mo.concat[1] (%a, %b) : (!mo.tensor<[4, 6], f32>, !mo.tensor<[4, 10], f32>) -> !mo.tensor<[4, 16], f32>
  %sq = mo.mul(%c, %c) : !mo.tensor<[4, 16], f32>
  %s = mo.reduce.add(%sq) : (!mo.tensor<[4, 16], f32>) -> !mo.tensor<[4, 1], f32>
  %r = mo.rsqrt(%s) : !mo.tensor<[4, 1], f32>
  %rb = mo.static.broadcast_to(%r) : !mo.tensor<[4, 1], f32> -> !mo.tensor<[4, 16], f32>
  %y = mo.mul(%c, %rb) : !mo.tensor<[4, 16], f32>
  mo.output %y : !mo.tensor<[4, 16], f32>
}

// Rotate-half after a reduction, as in RoPE: each element reads its partner in
// the other half of its row. The select's regions read the sum through a
// slice along the reduced axis, which stays in the row, so they read the
// row's value.
// CHECK-LABEL: mo.graph @rotate_half_of_scaled
// CHECK-NOT:   map.iter.elementwise
// CHECK:       map.iter.rowwise<[4, 16], axis = 1>
// CHECK:       %[[ACC:.*]] = map.iter.reduce
// CHECK:       map.iter.map index (%[[J:.*]]: !map.index<2>)
// CHECK:       map.index.concat.select<#map.index.concat.spec<[4, 8], [4, 8] -> [4, 16], axis = 1>> (%[[J]])
// CHECK:       map.call.elementwise @{{.*}}Mul(%{{.*}}, %{{.*}}, %[[ACC]])
// CHECK:       mo.output
mo.graph @rotate_half_of_scaled(%x: !mo.tensor<[4, 16], f32>) -> !mo.tensor<[4, 16], f32> {
  %s = mo.reduce.add(%x) : (!mo.tensor<[4, 16], f32>) -> !mo.tensor<[4, 1], f32>
  %sb = mo.static.broadcast_to(%s) : !mo.tensor<[4, 1], f32> -> !mo.tensor<[4, 16], f32>
  %y = mo.mul(%x, %sb) : !mo.tensor<[4, 16], f32>
  %start1 = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %start2 = mo.constant <#M.dense_array<0, 8> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop1 = mo.constant <#M.dense_array<4, 8> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop2 = mo.constant <#M.dense_array<4, 16> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %y1 = mo.slice(%y, %start1, %stop1, %step) : (!mo.tensor<[4, 16], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[4, 8], f32>
  %y2 = mo.slice(%y, %start2, %stop2, %step) : (!mo.tensor<[4, 16], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[4, 8], f32>
  %ny2 = mo.negative(%y2) : !mo.tensor<[4, 8], f32>
  %r = mo.concat[1] (%ny2, %y1) : (!mo.tensor<[4, 8], f32>, !mo.tensor<[4, 8], f32>) -> !mo.tensor<[4, 16], f32>
  mo.output %r : !mo.tensor<[4, 16], f32>
}

// The same over the other axis: the select's regions read the sums of other
// rows, so the consumer stays a kernel of its own.
// CHECK-LABEL: mo.graph @swap_rows_of_scaled
// CHECK:       map.iter.rowwise<[4, 16], axis = 1>
// CHECK-NOT:   map.iter.map
// CHECK:       map.iter.elementwise<[4, 16]>
// CHECK:       map.index.concat.select<#map.index.concat.spec<[2, 16], [2, 16] -> [4, 16], axis = 0>>
// CHECK:       mo.output
mo.graph @swap_rows_of_scaled(%x: !mo.tensor<[4, 16], f32>) -> !mo.tensor<[4, 16], f32> {
  %s = mo.reduce.add(%x) : (!mo.tensor<[4, 16], f32>) -> !mo.tensor<[4, 1], f32>
  %sb = mo.static.broadcast_to(%s) : !mo.tensor<[4, 1], f32> -> !mo.tensor<[4, 16], f32>
  %y = mo.mul(%x, %sb) : !mo.tensor<[4, 16], f32>
  %start1 = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %start2 = mo.constant <#M.dense_array<2, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop1 = mo.constant <#M.dense_array<2, 16> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop2 = mo.constant <#M.dense_array<4, 16> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %y1 = mo.slice(%y, %start1, %stop1, %step) : (!mo.tensor<[4, 16], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 16], f32>
  %y2 = mo.slice(%y, %start2, %stop2, %step) : (!mo.tensor<[4, 16], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 16], f32>
  %r = mo.concat[0] (%y2, %y1) : (!mo.tensor<[2, 16], f32>, !mo.tensor<[2, 16], f32>) -> !mo.tensor<[4, 16], f32>
  mo.output %r : !mo.tensor<[4, 16], f32>
}

// Rotate-half of a value the rowwise kernel stores per element, x / sum(x),
// also a graph output: the select's regions read that output at the partner
// element, which isn't the stored one, so the consumer stays a kernel of its
// own.
// CHECK-LABEL: mo.graph @rotate_half_of_stored
// CHECK:       %[[R:.*]] = map.iter.rowwise<[4, 16], axis = 1>
// CHECK-NOT:   map.index.concat.select
// CHECK:       map.iter.elementwise<[4, 16]>
// CHECK-NEXT:  inputs (%[[R]]
// CHECK:       map.index.concat.select<#map.index.concat.spec<[4, 8], [4, 8] -> [4, 16], axis = 1>>
// CHECK:       mo.output
mo.graph @rotate_half_of_stored(%x: !mo.tensor<[4, 16], f32>) -> (!mo.tensor<[4, 16], f32>, !mo.tensor<[4, 16], f32>) {
  %s = mo.reduce.add(%x) : (!mo.tensor<[4, 16], f32>) -> !mo.tensor<[4, 1], f32>
  %sb = mo.static.broadcast_to(%s) : !mo.tensor<[4, 1], f32> -> !mo.tensor<[4, 16], f32>
  %y = mo.div(%x, %sb) : !mo.tensor<[4, 16], f32>
  %start1 = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %start2 = mo.constant <#M.dense_array<0, 8> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop1 = mo.constant <#M.dense_array<4, 8> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop2 = mo.constant <#M.dense_array<4, 16> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %y1 = mo.slice(%y, %start1, %stop1, %step) : (!mo.tensor<[4, 16], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[4, 8], f32>
  %y2 = mo.slice(%y, %start2, %stop2, %step) : (!mo.tensor<[4, 16], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[4, 8], f32>
  %ny2 = mo.negative(%y2) : !mo.tensor<[4, 8], f32>
  %r = mo.concat[1] (%ny2, %y1) : (!mo.tensor<[4, 8], f32>, !mo.tensor<[4, 8], f32>) -> !mo.tensor<[4, 16], f32>
  mo.output %y, %r : !mo.tensor<[4, 16], f32>, !mo.tensor<[4, 16], f32>
}
