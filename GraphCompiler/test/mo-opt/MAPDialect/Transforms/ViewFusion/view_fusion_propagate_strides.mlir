// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-fuse-views | FileCheck %s

// The advanced-fusion counterpart to
// `integration/cpu/MODialect/mogg-kernel-api-e2e-fusion-propagate-strides.mlir`,
// whose graphs these are. That test reads the propagated static strides out of
// `mogg-debug-dump-fused-tensor-specs`, a hook only the legacy path emits, so
// it cannot run here. On this path the same information is an index transform
// in the IR: the spec attribute below is what the stride tuple is computed
// from, so checking it holds the view algebra to the same shapes, offsets,
// steps and permutations. The stride tuple itself is derived later in Mojo
// and is not covered.

// CHECK-LABEL: mo.graph @broadcast_to
// CHECK: map.index.broadcast<#map.index.broadcast.spec<[4, 1] -> [5, 4, 3]>>
mo.graph @broadcast_to(%x: !mo.tensor<[4, 1], f32>, %o: !mo.tensor<[5, 4, 3], f32>) -> !mo.tensor<[5, 4, 3], f32> {
  %0 = mo.static.broadcast_to(%x) : !mo.tensor<[4, 1], f32> -> !mo.tensor<[5, 4, 3], f32>
  %1 = mo.add(%0, %o) : !mo.tensor<[5, 4, 3], f32>
  mo.output %1 : !mo.tensor<[5, 4, 3], f32>
}

// CHECK-LABEL: mo.graph @reshape
// CHECK: map.index.reshape<#map.index.reshape.spec<[24] -> [3, 4, 2]>>
mo.graph @reshape(%x: !mo.tensor<[24], f32>, %o: !mo.tensor<[3, 4, 2], f32>) -> !mo.tensor<[3, 4, 2], f32> {
  %0 = mo.static.reshape(%x) : !mo.tensor<[24], f32> -> !mo.tensor<[3, 4, 2], f32>
  %1 = mo.add(%0, %o) : !mo.tensor<[3, 4, 2], f32>
  mo.output %1 : !mo.tensor<[3, 4, 2], f32>
}

// CHECK-LABEL: mo.graph @transpose
// CHECK: map.index.transpose<#map.index.transpose.spec<[2, 3, 4] -> [3, 4, 2], perm = [1, 2, 0]>>
mo.graph @transpose(%x: !mo.tensor<[2, 3, 4], f32>, %o: !mo.tensor<[3, 4, 2], f32>) -> !mo.tensor<[3, 4, 2], f32> {
  %perm = mo.constant <#M.dense_array<1, 2, 0> : tensor<3xsi32>> : !mo.tensor<[3], si32>
  %0 = mo.transpose(%x, %perm) : (!mo.tensor<[2, 3, 4], f32>, !mo.tensor<[3], si32>) -> !mo.tensor<[3, 4, 2], f32>
  %1 = mo.add(%0, %o) : !mo.tensor<[3, 4, 2], f32>
  mo.output %1 : !mo.tensor<[3, 4, 2], f32>
}

// CHECK-LABEL: mo.graph @slice_dim_pos_step
// CHECK: map.index.slice<#map.index.slice.spec<[3, 5] -> [2, 3], starts = [1, 0], stops = [3, 5], steps = [1, 2]>>
mo.graph @slice_dim_pos_step(%x: !mo.tensor<[3, 5], f32>, %o: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> {
  %start = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop = mo.constant <#M.dense_array<3, 5> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %res = mo.slice(%x, %start, %stop, %step) : (
    !mo.tensor<[3, 5], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[2, 3], f32>
  %out = mo.add(%res, %o) : !mo.tensor<[2, 3], f32>
  mo.output %out : !mo.tensor<[2, 3], f32>
}

// A reverse slice reaches the spec with its negative literals already resolved
// against the source dims, which is the normalization the verifier requires.

// CHECK-LABEL: mo.graph @slice_dim_neg_step
// CHECK: map.index.slice<#map.index.slice.spec<[5, 3] -> [2, 2], starts = [4, 2], stops = [4, 2], steps = [-2, -1]>>
mo.graph @slice_dim_neg_step(%x: !mo.tensor<[5, 3], f32>, %o: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %start = mo.constant <#M.dense_array<-1, -1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop = mo.constant <#M.dense_array<-1, -1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<-2, -1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %res = mo.slice(%x, %start, %stop, %step) : (
    !mo.tensor<[5, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[2, 2], f32>
  %out = mo.add(%res, %o) : !mo.tensor<[2, 2], f32>
  mo.output %out : !mo.tensor<[2, 2], f32>
}

// Two hops compose in consumer-to-producer order: the loop index is permuted
// first, then de-linearized onto the flat source.

// CHECK-LABEL: mo.graph @reshape_and_transpose
// CHECK: %[[TIDX:.*]] = map.index.transpose<#map.index.transpose.spec<[2, 3, 4] -> [3, 4, 2], perm = [1, 2, 0]>>
// CHECK: map.index.reshape<#map.index.reshape.spec<[24] -> [2, 3, 4]>> (%[[TIDX]])
mo.graph @reshape_and_transpose(%x: !mo.tensor<[24], f32>, %o: !mo.tensor<[3, 4, 2], f32>) -> !mo.tensor<[3, 4, 2], f32> {
  %0 = mo.static.reshape(%x) : !mo.tensor<[24], f32> -> !mo.tensor<[2, 3, 4], f32>
  %perm = mo.constant <#M.dense_array<1, 2, 0> : tensor<3xsi32>> : !mo.tensor<[3], si32>
  %1 = mo.transpose(%0, %perm) : (!mo.tensor<[2, 3, 4], f32>, !mo.tensor<[3], si32>) -> !mo.tensor<[3, 4, 2], f32>
  %2 = mo.add(%1, %o) : !mo.tensor<[3, 4, 2], f32>
  mo.output %2 : !mo.tensor<[3, 4, 2], f32>
}

// Every start offset is zero, so only the source's row pitch distinguishes the
// view from a contiguous tensor. Regression test for GEX-4058.

// CHECK-LABEL: mo.graph @slice_column_zero_start
// CHECK: map.index.slice<#map.index.slice.spec<[4, 10] -> [4, 8], starts = [0, 0], stops = [4, 8], steps = [1, 1]>>
mo.graph @slice_column_zero_start(%x: !mo.tensor<[4, 10], f32>, %o: !mo.tensor<[4, 8], f32>) -> !mo.tensor<[4, 8], f32> {
  %start = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop = mo.constant <#M.dense_array<4, 8> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %res = mo.slice(%x, %start, %stop, %step) : (
    !mo.tensor<[4, 10], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[4, 8], f32>
  %out = mo.add(%res, %o) : !mo.tensor<[4, 8], f32>
  mo.output %out : !mo.tensor<[4, 8], f32>
}

// Two views into one elementwise consumer, from the `E2E/` sibling of the
// integration test above.

// CHECK-LABEL: mo.graph @fused_broadcast_transpose
// CHECK-DAG: map.index.broadcast<#map.index.broadcast.spec<[4, 1] -> [5, 4, 3]>>
// CHECK-DAG: map.index.transpose<#map.index.transpose.spec<[3, 4, 5] -> [5, 4, 3], perm = [2, 1, 0]>>
mo.graph @fused_broadcast_transpose(%x: !mo.tensor<[4, 1], f32>, %y: !mo.tensor<[3, 4, 5], f32>) -> !mo.tensor<[5, 4, 3], f32> {
  %0 = mo.static.broadcast_to(%x) : !mo.tensor<[4, 1], f32> -> !mo.tensor<[5, 4, 3], f32>
  %perm = mo.constant <#M.dense_array<2, 1, 0> : tensor<3xsi32>> : !mo.tensor<[3], si32>
  %1 = mo.transpose(%y, %perm) : (!mo.tensor<[3, 4, 5], f32>, !mo.tensor<[3], si32>) -> !mo.tensor<[5, 4, 3], f32>
  %2 = mo.add(%0, %1) : !mo.tensor<[5, 4, 3], f32>
  mo.output %2 : !mo.tensor<[5, 4, 3], f32>
}
