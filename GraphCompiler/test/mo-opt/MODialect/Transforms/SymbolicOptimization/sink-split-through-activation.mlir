// RUN: mo-opt %s --mo-symbolic-optimize | FileCheck %s

// When a `mo.split`'s results all feed the same elementwise op,
// `SinkSplitThroughElementwisePattern` rewrites them into a single elementwise
// op on the pre-split input followed by the split. The rebuilt elementwise op
// must keep its inherent attributes -- here the `activation` attribute on
// `mo.relu`. Dropping it produced an op with a null `activation`, which
// crashed downstream canonicalizations (e.g. relu -> max). See GEX.

// CHECK-LABEL: @sink_split_through_activation
// CHECK:         %[[ACT:.*]] = mo.relu(%{{.*}}) : !mo.tensor<[2, 4], f32>
// CHECK:         %[[SPLIT:.*]]:2 = mo.split{{.*}}(%[[ACT]], %{{.*}})
// CHECK:         mo.output %[[SPLIT]]#0, %[[SPLIT]]#1
mo.graph @sink_split_through_activation(%A: !mo.tensor<[2, 4], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %sizes = mo.constant <#M.dense_array<2, 2> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %s0, %s1 = mo.split(%A, %sizes) [axis = 1] : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
  %r0 = mo.relu(%s0) : !mo.tensor<[2, 2], f32>
  %r1 = mo.relu(%s1) : !mo.tensor<[2, 2], f32>
  mo.output %r0, %r1 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}
