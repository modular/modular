// RUN: MAX_GC_USE_ADV_FUSION=1 mo-opt --mlir-print-local-scope %s --mogg-import-packages --mo-to-map --map-fuse-views | FileCheck %s

// Fusing a producer into a consumer keeps the consumer's launch policy, as on
// the old MO->MOGG path. A transpose launches synchronously
// (`MO::kernelRequiresSyncLaunch`, GEX-2569) to bound the memory held by
// in-flight buffers, so absorbing an async view must not make it async.

// CHECK-LABEL: mo.graph @sliceIntoTranspose
// CHECK:       map.iter.elementwise<[32, 3]> dispatch(immediate)
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_view<[#mogg.fusion_leaf<"mo.slice"{{.*}}, #mogg.fusion_leaf<"mo.transpose"{{.*}}]>>
mo.graph @sliceIntoTranspose(%input: !mo.tensor<[10, 64], f32>) -> !mo.tensor<[32, 3], f32> {
  %start = mo.constant <#M.dense_array<1, 32> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop = mo.constant <#M.dense_array<4, 64> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %sliced = mo.slice(%input, %start, %stop, %step) : (
    !mo.tensor<[10, 64], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[3, 32], f32>
  %perm = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %res = mo.transpose(%sliced, %perm) : (!mo.tensor<[3, 32], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[32, 3], f32>
  mo.output %res : !mo.tensor<[32, 3], f32>
}

// The shape that ran async in model init: a reshape feeding a weight
// transpose.

// CHECK-LABEL: mo.graph @reshapeIntoTranspose
// CHECK:       map.iter.elementwise<[6, 4]> dispatch(immediate)
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_view<[#mogg.fusion_leaf<"mo.static.reshape"{{.*}}, #mogg.fusion_leaf<"mo.transpose"{{.*}}]>>
mo.graph @reshapeIntoTranspose(%input: !mo.tensor<[24], f32>) -> !mo.tensor<[6, 4], f32> {
  %reshaped = mo.static.reshape(%input) : !mo.tensor<[24], f32> -> !mo.tensor<[4, 6], f32>
  %perm = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %res = mo.transpose(%reshaped, %perm) : (!mo.tensor<[4, 6], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[6, 4], f32>
  mo.output %res : !mo.tensor<[6, 4], f32>
}

// The other direction: a transpose fused into an async consumer leaves the
// consumer deferred (the default, so `dispatch` is not printed).

// CHECK-LABEL: mo.graph @transposeIntoElementwise
// CHECK:       map.iter.elementwise<[32, 3]>
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_view<[#mogg.fusion_leaf<"mo.transpose"{{.*}}, #mogg.fusion_leaf<"mo.add"{{.*}}]>>
// CHECK-SAME:  hasWriteEffect = false}
mo.graph @transposeIntoElementwise(%input: !mo.tensor<[3, 32], f32>, %other: !mo.tensor<[32, 3], f32>) -> !mo.tensor<[32, 3], f32> {
  %perm = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %t = mo.transpose(%input, %perm) : (!mo.tensor<[3, 32], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[32, 3], f32>
  %res = mo.add(%t, %other) : !mo.tensor<[32, 3], f32>
  mo.output %res : !mo.tensor<[32, 3], f32>
}
