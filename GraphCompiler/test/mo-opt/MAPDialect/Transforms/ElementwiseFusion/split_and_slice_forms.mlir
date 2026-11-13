// RUN: mo-opt %s --mo-pipeline --mo-to-map --map-pipeline -split-input-file | FileCheck %s

// `ops.split` then a cast of each half, and the same halves taken as slices,
// must compile to one kernel. Symbolic optimization moves the split form's
// casts above the split; the cast must fuse through the slices into the
// consumer.

// CHECK-LABEL: mo.graph @split_form
// CHECK:       map.iter.opaque
// CHECK-NEXT:  inputs (%arg0: !mo.tensor<[rows, 8], bf16>)
// CHECK-DAG:   map.index.slice.stride<{{.*}}starts = [0, 0]{{.*}}> (%arg0,
// CHECK-DAG:   map.index.slice.stride<{{.*}}starts = [0, 4]{{.*}}> (%arg0,
// CHECK:       map.iter.elementwise.body<[rows, 4]>
// CHECK-NOT:   map.iter.opaque
// CHECK:       mo.output

mo.graph @split_form<rows>(%x: !mo.tensor<[rows, 8], bf16>) -> (!mo.tensor<[rows, 4], f32>) {
  %sizes = mo.constant <#M.dense_array<4, 4> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %s:2 = mo.split(%x, %sizes) [axis = 1] : (!mo.tensor<[rows, 8], bf16>, !mo.tensor<[2], si64>) -> (!mo.tensor<[rows, 4], bf16>, !mo.tensor<[rows, 4], bf16>)
  %g = mo.cast(%s#0) : (!mo.tensor<[rows, 4], bf16>) -> !mo.tensor<[rows, 4], f32>
  %u = mo.cast(%s#1) : (!mo.tensor<[rows, 4], bf16>) -> !mo.tensor<[rows, 4], f32>
  %sg = mo.sigmoid(%g) : !mo.tensor<[rows, 4], f32>
  %h = mo.mul(%sg, %u) : !mo.tensor<[rows, 4], f32>
  mo.output %h : !mo.tensor<[rows, 4], f32>
}

// -----

// The slice form. Its stops are graph inputs, as when the row count is
// symbolic, so the slices do not become a split before pattern fusion.

// CHECK-LABEL: mo.graph @slice_form
// CHECK:       map.iter.opaque
// CHECK-NEXT:  inputs (%arg0: !mo.tensor<[rows, 8], bf16>)
// CHECK-DAG:   map.index.slice.stride<{{.*}}starts = [0, 0]{{.*}}> (%arg0,
// CHECK-DAG:   map.index.slice.stride<{{.*}}starts = [0, 4]{{.*}}> (%arg0,
// CHECK:       map.iter.elementwise.body<[rows, 4]>
// CHECK-NOT:   map.iter.opaque
// CHECK:       mo.output

mo.graph @slice_form<rows>(%x: !mo.tensor<[rows, 8], bf16>, %stop0: !mo.tensor<[2], si64>, %stop1: !mo.tensor<[2], si64>) -> (!mo.tensor<[rows, 4], f32>) {
  %start0 = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %start1 = mo.constant <#M.dense_array<0, 4> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %s0 = mo.slice(%x, %start0, %stop0, %step) : (!mo.tensor<[rows, 8], bf16>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[rows, 4], bf16>
  %s1 = mo.slice(%x, %start1, %stop1, %step) : (!mo.tensor<[rows, 8], bf16>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[rows, 4], bf16>
  %g = mo.cast(%s0) : (!mo.tensor<[rows, 4], bf16>) -> !mo.tensor<[rows, 4], f32>
  %u = mo.cast(%s1) : (!mo.tensor<[rows, 4], bf16>) -> !mo.tensor<[rows, 4], f32>
  %sg = mo.sigmoid(%g) : !mo.tensor<[rows, 4], f32>
  %h = mo.mul(%sg, %u) : !mo.tensor<[rows, 4], f32>
  mo.output %h : !mo.tensor<[rows, 4], f32>
}
