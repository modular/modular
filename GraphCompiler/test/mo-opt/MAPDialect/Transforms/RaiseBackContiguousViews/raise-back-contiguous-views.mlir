// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-raise-back-contiguous-views | FileCheck %s

// RaiseBackContiguousViews raises an unfused, contiguous slice back to
// mogg.slice (zero-copy) and an unfused reshape back to mo.static.reshape
// (unconditionally -- MOGGToMGP always turns it into a non-copying
// primitive). A slice that isn't contiguous, or a view kind out of scope
// (broadcast/transpose), is left as a map.iter.elementwise for the ordinary MAP
// pipeline to handle.

// Only the leading dim is sliced; the trailing dim keeps its full extent --
// the common case of "the sliced dim is the first non-1 input dim".

// CHECK-LABEL: mo.graph @contiguousSliceLeadingDim
// CHECK-NOT:     map.iter.elementwise
// CHECK:         [[START:%.*]] = mo.constant <#M.dense_array<1> : !M.array<1xsi64>>
// CHECK:         [[STOP:%.*]] = mo.constant <#M.dense_array<4> : !M.array<1xsi64>>
// CHECK:         mogg.slice(%arg0, [[START]], [[STOP]]) : (!mo.tensor<[10, 64], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[3, 64], f32>
mo.graph @contiguousSliceLeadingDim(%input: !mo.tensor<[10, 64], f32>) -> !mo.tensor<[3, 64], f32> {
  %start = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop = mo.constant <#M.dense_array<4, 64> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %res = mo.slice(%input, %start, %stop, %step) : (
    !mo.tensor<[10, 64], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[3, 64], f32>
  mo.output %res : !mo.tensor<[3, 64], f32>
}

// Leading size-1 dims are skipped when locating "the" sliced dimension: here
// dim 0 has size 1 and is untouched, dim 1 is the one actually sliced.

// CHECK-LABEL: mo.graph @contiguousSliceSkipLeadingOnes
// CHECK-NOT:     map.iter.elementwise
// CHECK:         mogg.slice(%arg0, %{{.*}}, %{{.*}}) : (!mo.tensor<[1, 10, 64], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[1, 3, 64], f32>
mo.graph @contiguousSliceSkipLeadingOnes(%input: !mo.tensor<[1, 10, 64], f32>) -> !mo.tensor<[1, 3, 64], f32> {
  %start = mo.constant <#M.dense_array<0, 1, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %stop = mo.constant <#M.dense_array<1, 4, 64> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %step = mo.constant <#M.dense_array<1, 1, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %res = mo.slice(%input, %start, %stop, %step) : (
    !mo.tensor<[1, 10, 64], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>
  ) -> !mo.tensor<[1, 3, 64], f32>
  mo.output %res : !mo.tensor<[1, 3, 64], f32>
}

// Both dims change (not contiguous along a single dim): left as a
// map.iter.elementwise for the general MAPToMOGG zero-copy strided-view path.

// CHECK-LABEL: mo.graph @nonContiguousMultiDimSlice
// CHECK-NOT:     mogg.slice
// CHECK:         map.iter.elementwise<[3, 32]>
// CHECK:         map.index.slice<#map.index.slice.spec<[10, 64] -> [3, 32], starts = [1, 32], stops = [4, 64], steps = [1, 1]>>
mo.graph @nonContiguousMultiDimSlice(%input: !mo.tensor<[10, 64], f32>) -> !mo.tensor<[3, 32], f32> {
  %start = mo.constant <#M.dense_array<1, 32> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop = mo.constant <#M.dense_array<4, 64> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %res = mo.slice(%input, %start, %stop, %step) : (
    !mo.tensor<[10, 64], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[3, 32], f32>
  mo.output %res : !mo.tensor<[3, 32], f32>
}

// The sliced dim's start/stop aren't compile-time constants: left as a
// map.iter.elementwise (only step needs to be a literal 1 for the fusion
// permission check ViewFuser/ElementwiseFuser use elsewhere, but raising
// back to mogg.slice needs actual scalar values for its start/stop
// operands).

// CHECK-LABEL: mo.graph @nonContiguousDynamicBounds
// CHECK-NOT:     mogg.slice
// CHECK:         map.iter.elementwise<[3, 64]>
// CHECK:         map.index.slice<#map.index.slice.spec<[10, 64] -> [3, 64], starts = [?, ?], stops = [?, ?], steps = [?, ?]>>
mo.graph @nonContiguousDynamicBounds(
    %input: !mo.tensor<[10, 64], f32>,
    %start: !mo.tensor<[2], si64>,
    %stop: !mo.tensor<[2], si64>,
    %step: !mo.tensor<[2], si64>)
    -> !mo.tensor<[3, 64], f32> {
  %res = mo.slice(%input, %start, %stop, %step) : (
    !mo.tensor<[10, 64], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[3, 64], f32>
  mo.output %res : !mo.tensor<[3, 64], f32>
}

// A reshape is always raised back, unconditionally -- no contiguity check.

// CHECK-LABEL: mo.graph @reshapeAlwaysRaised
// CHECK-NOT:     map.iter.elementwise
// CHECK:         mo.static.reshape(%arg0) : !mo.tensor<[4, 6], f32> -> !mo.tensor<[24], f32>
mo.graph @reshapeAlwaysRaised(%input: !mo.tensor<[4, 6], f32>) -> !mo.tensor<[24], f32> {
  %res = mo.static.reshape(%input) : !mo.tensor<[4, 6], f32> -> !mo.tensor<[24], f32>
  mo.output %res : !mo.tensor<[24], f32>
}

// A contiguous slice inside an `mo.if` branch still raises: the pass walks
// rather than iterates the graph's direct ops, so a view kernel nested in
// control flow isn't missed.

// CHECK-LABEL: mo.graph @contiguousSliceInsideIf
// CHECK:         mo.if
// CHECK-NOT:     map.iter.elementwise
// CHECK:         mogg.slice(%arg0, %{{.*}}, %{{.*}}) : (!mo.tensor<[10, 64], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[3, 64], f32>
// CHECK:         } else {
mo.graph @contiguousSliceInsideIf(
    %input: !mo.tensor<[10, 64], f32>,
    %cond: !mo.tensor<[], bool>
) -> !mo.tensor<[3, 64], f32> {
  %start = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop = mo.constant <#M.dense_array<4, 64> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %res = mo.if (%cond : !mo.tensor<[], bool>) -> !mo.tensor<[3, 64], f32> {
    %sliced = mo.slice(%input, %start, %stop, %step) : (
      !mo.tensor<[10, 64], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
    ) -> !mo.tensor<[3, 64], f32>
    mo.yield %sliced : !mo.tensor<[3, 64], f32>
  } else {
    %sliced = mo.slice(%input, %start, %stop, %step) : (
      !mo.tensor<[10, 64], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
    ) -> !mo.tensor<[3, 64], f32>
    mo.yield %sliced : !mo.tensor<[3, 64], f32>
  }
  mo.output %res : !mo.tensor<[3, 64], f32>
}

// Broadcast is out of scope (matches the legacy raise-back's own scope):
// left as a map.iter.elementwise even though it is unfused.

// CHECK-LABEL: mo.graph @broadcastNotRaised
// CHECK:         map.iter.elementwise<[8, 64]>
// CHECK:         map.index.broadcast<#map.index.broadcast.spec<[1, 64] -> [8, 64]>>
mo.graph @broadcastNotRaised(%input: !mo.tensor<[1, 64], f32>) -> !mo.tensor<[8, 64], f32> {
  %res = mo.static.broadcast_to(%input) : !mo.tensor<[1, 64], f32> -> !mo.tensor<[8, 64], f32>
  mo.output %res : !mo.tensor<[8, 64], f32>
}
