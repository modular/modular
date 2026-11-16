// RUN: mo-opt --mlir-print-local-scope %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-kernels | FileCheck %s

// A concat is a loop reading its inputs through `map.index.concat.select`. A
// reduction that is its only reader fuses it like any producer: the select
// moves into the fold, at the fold's index, and the concat isn't
// materialized.

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

// The concat feeds both the reduction's fold and its per-element phase (the
// add). A select has no lowering in a map phase yet, so the concat stays a
// kernel of its own and the rowwise reads its result.
// CHECK-LABEL: mo.graph @concat_plus_its_sum
// CHECK:       %[[C:.*]] = map.iter.elementwise
// CHECK:       map.index.concat.select
// CHECK:       map.iter.rowwise<[4, 16], axis = 1>
// CHECK-NEXT:  inputs (%[[C]]
// CHECK-NOT:   map.index.concat.select
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

// x / sum(x): the same per-element phase as the add above, as a softmax-like
// normalization writes it. The concat stays a kernel of its own.
// CHECK-LABEL: mo.graph @concat_over_its_sum
// CHECK:       %[[C:.*]] = map.iter.elementwise
// CHECK:       map.index.concat.select
// CHECK:       map.iter.rowwise<[4, 16], axis = 1>
// CHECK-NEXT:  inputs (%[[C]]
// CHECK-NOT:   map.index.concat.select
// CHECK:       mo.output
mo.graph @concat_over_its_sum(%a: !mo.tensor<[4, 6], f32>, %b: !mo.tensor<[4, 10], f32>) -> !mo.tensor<[4, 16], f32> {
  %c = mo.concat[1] (%a, %b) : (!mo.tensor<[4, 6], f32>, !mo.tensor<[4, 10], f32>) -> !mo.tensor<[4, 16], f32>
  %s = mo.reduce.add(%c) : (!mo.tensor<[4, 16], f32>) -> !mo.tensor<[4, 1], f32>
  %sb = mo.static.broadcast_to(%s) : !mo.tensor<[4, 1], f32> -> !mo.tensor<[4, 16], f32>
  %y = mo.div(%c, %sb) : !mo.tensor<[4, 16], f32>
  mo.output %y : !mo.tensor<[4, 16], f32>
}
