// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-prologue --map-to-mogg | FileCheck %s

// The prologue builds its shape from the dynamic dim.
// CHECK-LABEL: mo.graph @dyn_fused_producer
// CHECK: mogg.routine
// CHECK: dims (%[[D:.*]]: !mogg.dim<batch>)
// CHECK: mogg.shape.create(%[[D]])
// CHECK-SAME: <shape = #mosh<ape[dyn, 4]> : !mosh.ape>
// CHECK-SAME: : !mogg.dim<batch>
// CHECK: %[[PRO:[^ ]+]] = mogg.tensor.create.prologue %{{[^ ]+}}
// CHECK-SAME: (%[[I:[^ ]+]] : !map.index<2>)
// CHECK-SAME: capturing {%[[A0:[^ ]+]] = %[[C0:[^ ]+]] : !mogg.tensor,
// CHECK-SAME: %[[A1:[^ ]+]] = %[[C1:[^ ]+]] : !mogg.tensor}
// CHECK-SAME: device = "cpu", elementType = f32,
// CHECK-SAME: io = Input, mut = false {
// CHECK: %[[L0:.*]] = mogg.tensor.load %[[A0]][%[[I]]]
// CHECK: %[[L1:.*]] = mogg.tensor.load %[[A1]][%[[I]]]
// CHECK: %[[ADD:.*]] = mogg.call.elementwise @{{.*}}Add(%[[L0]], %[[L1]])
// CHECK: mogg.yield %[[ADD]] : !mogg.simd<f32>
// CHECK: mogg.call.opaque @{{.*}}MapLowerPartialFusedInputs
// CHECK-SAME: inputs(%[[PRO]], %{{.*}} : !mogg.tensor, !mogg.tensor)
// CHECK-NOT: map.iter.prologue
mo.graph @dyn_fused_producer(
  %a: !mo.tensor<[batch, 4], f32>,
  %b: !mo.tensor<[batch, 4], f32>,
  %c: !mo.tensor<[batch, 4], f32>
) -> !mo.tensor<[batch, 4], f32> {
  %p = mo.add(%a, %b) : !mo.tensor<[batch, 4], f32>
  %out = mo.custom [symbol = "map_lower_partial_fused_inputs", device = #M.device_ref<"cpu", 0>](%p, %c) : (!mo.tensor<[batch, 4], f32>, !mo.tensor<[batch, 4], f32>) -> !mo.tensor<[batch, 4], f32>
  mo.output %out : !mo.tensor<[batch, 4], f32>
}
