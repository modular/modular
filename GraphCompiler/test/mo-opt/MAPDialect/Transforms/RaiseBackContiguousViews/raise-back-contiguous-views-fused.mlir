// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-fuse-views --map-raise-back-contiguous-views | FileCheck %s

// ViewFuser collapses a chain of adjacent views (here two slices) into one
// kernel with several chained map.index transforms. Since the whole chain is
// still a pure sequence of contiguous slices/reshapes, RaiseBackContiguousViews
// raises it back to a sequence of zero-copy ops -- one mogg.slice per hop --
// which MOGGToMGP lowers without a copy. (A single mogg.slice cannot express a
// composed chain, but one op per hop can.)

// CHECK-LABEL: mo.graph @fusedSliceChainRaised
// CHECK-NOT:     map.iter.elementwise
// CHECK-NOT:     map.index.slice
// CHECK:         mogg.slice({{.*}}) : (!mo.tensor<[10, 64], f32>, {{.*}}) -> !mo.tensor<[6, 64], f32>
// CHECK:         mogg.slice({{.*}}) : (!mo.tensor<[6, 64], f32>, {{.*}}) -> !mo.tensor<[3, 64], f32>
mo.graph @fusedSliceChainRaised(%input: !mo.tensor<[10, 64], f32>) -> !mo.tensor<[3, 64], f32> {
  %start1 = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop1 = mo.constant <#M.dense_array<6, 64> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step1 = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %mid = mo.slice(%input, %start1, %stop1, %step1) : (
    !mo.tensor<[10, 64], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[6, 64], f32>
  %start2 = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop2 = mo.constant <#M.dense_array<4, 64> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step2 = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %res = mo.slice(%mid, %start2, %stop2, %step2) : (
    !mo.tensor<[6, 64], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[3, 64], f32>
  mo.output %res : !mo.tensor<[3, 64], f32>
}

// A reshape fused with a contiguous slice -- the shape that arises when block
// scales are reshaped then sliced before an interleave/matmul (grouped_matmul
// swiglu). The chain raises to mo.static.reshape followed by a zero-copy
// mogg.slice, so the interleave reads a view instead of a materialized copy.

// CHECK-LABEL: mo.graph @fusedReshapeSliceChainRaised
// CHECK-NOT:     map.iter.elementwise
// CHECK-NOT:     map.index.reshape
// CHECK-NOT:     map.index.slice
// CHECK:         mo.static.reshape(%arg0) : !mo.tensor<[2, 256, 16], f32> -> !mo.tensor<[512, 16], f32>
// CHECK:         mogg.slice({{.*}}) : (!mo.tensor<[512, 16], f32>, {{.*}}) -> !mo.tensor<[256, 16], f32>
mo.graph @fusedReshapeSliceChainRaised(%input: !mo.tensor<[2, 256, 16], f32>) -> !mo.tensor<[256, 16], f32> {
  %flat = mo.static.reshape(%input) : !mo.tensor<[2, 256, 16], f32> -> !mo.tensor<[512, 16], f32>
  %start = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop = mo.constant <#M.dense_array<256, 16> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %res = mo.slice(%flat, %start, %stop, %step) : (
    !mo.tensor<[512, 16], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[256, 16], f32>
  mo.output %res : !mo.tensor<[256, 16], f32>
}
