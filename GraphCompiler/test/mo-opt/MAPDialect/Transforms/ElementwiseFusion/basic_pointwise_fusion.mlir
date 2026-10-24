// RUN: mo-opt --mlir-print-local-scope %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_pkg" --mo-to-map --map-fuse-elementwise | FileCheck %s

mo.graph @basic_add_mul(
  %x: !mo.tensor<[8], f32>,
  %y: !mo.tensor<[8], f32>,
  %z: !mo.tensor<[8], f32>
) -> !mo.tensor<[8], f32> {
  %tmp = mo.add(%x, %y) : !mo.tensor<[8], f32>
  %out = mo.mul(%tmp, %z) : !mo.tensor<[8], f32>
  mo.output %out : !mo.tensor<[8], f32>
}

// CHECK-LABEL: mo.graph @basic_add_mul
// CHECK-SAME:  (%[[X:.*]]: !mo.tensor<[8], f32>, %[[Y:.*]]: !mo.tensor<[8], f32>, %[[Z:.*]]: !mo.tensor<[8], f32>)
// CHECK-NEXT:  %[[OUT:.*]] = map.iter.elementwise<[8]>
// CHECK-NEXT:  inputs (%[[X]]: !mo.tensor<[8], f32>, %[[Y]]: !mo.tensor<[8], f32>, %[[Z]]: !mo.tensor<[8], f32>)
// CHECK:       %[[XV:.*]] = map.load %[[X]][%{{.*}}] : !mo.tensor<[8], f32>
// CHECK-NEXT:  %[[YV:.*]] = map.load %[[Y]][%{{.*}}] : !mo.tensor<[8], f32>
// CHECK-NEXT:  %[[ADD:.*]] = map.call.elementwise @{{.*}}.Add(%{{.*}}, %[[XV]], %[[YV]])
// CHECK-NEXT:  %[[ZV:.*]] = map.load %[[Z]][%{{.*}}] : !mo.tensor<[8], f32>
// CHECK-NEXT:  map.call.elementwise @{{.*}}.Mul(%{{.*}}, %[[ADD]], %[[ZV]])
// CHECK-NEXT:  map.store
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_elementwise<[#mogg.fusion_leaf<"mo.add"{{.*}}, #mogg.fusion_leaf<"mo.mul"{{.*}}]>>
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output %[[OUT]] : !mo.tensor<[8], f32>


mo.graph @add_mul_duplicate_operands(
  %x: !mo.tensor<[8], f32>
) -> !mo.tensor<[8], f32> {
  %t0 = mo.add(%x, %x) : !mo.tensor<[8], f32>
  %t1 = mo.mul(%t0, %x) : !mo.tensor<[8], f32>
  %out = mo.add(%x, %t1) : !mo.tensor<[8], f32>
  mo.output %out : !mo.tensor<[8], f32>
}

// Duplicate %x operand slots are uniqued after fusion; %x is a single input.
// CHECK-LABEL: mo.graph @add_mul_duplicate_operands
// CHECK-SAME:  (%[[X:.*]]: !mo.tensor<[8], f32>)
// CHECK-NEXT:  %[[OUT2:.*]] = map.iter.elementwise<[8]>
// CHECK-NEXT:  inputs (%[[X]]: !mo.tensor<[8], f32>)
// CHECK:       %[[V1:.*]] = map.load %[[X]][%{{.*}}] : !mo.tensor<[8], f32>
// CHECK-NEXT:  %[[V2:.*]] = map.load %[[X]][%{{.*}}] : !mo.tensor<[8], f32>
// CHECK-NEXT:  %[[V3:.*]] = map.load %[[X]][%{{.*}}] : !mo.tensor<[8], f32>
// CHECK-NEXT:  %[[ADD1:.*]] = map.call.elementwise @{{.*}}.Add(%{{.*}}, %[[V2]], %[[V3]])
// CHECK-NEXT:  %[[V4:.*]] = map.load %[[X]][%{{.*}}] : !mo.tensor<[8], f32>
// CHECK-NEXT:  %[[MUL:.*]] = map.call.elementwise @{{.*}}.Mul(%{{.*}}, %[[ADD1]], %[[V4]])
// CHECK-NEXT:  map.call.elementwise @{{.*}}.Add(%{{.*}}, %[[V1]], %[[MUL]])
// CHECK-NEXT:  map.store
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_elementwise<[#mogg.fusion_leaf<"mo.add"{{.*}}, #mogg.fusion_leaf<"mo.mul"{{.*}}, #mogg.fusion_leaf<"mo.add"{{.*}}]>>
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output %[[OUT2]] : !mo.tensor<[8], f32>


mo.graph @fusion_diamond(
  %x: !mo.tensor<[8], f32>,
  %y: !mo.tensor<[8], f32>
) -> !mo.tensor<[8], f32> {
  %t0 = mo.add(%x, %y) : !mo.tensor<[8], f32>
  %t1 = mo.mul(%t0, %t0) : !mo.tensor<[8], f32>
  %t2 = mo.relu(%t0) : !mo.tensor<[8], f32>
  %out = mo.add(%t1, %t2) : !mo.tensor<[8], f32>
  mo.output %out : !mo.tensor<[8], f32>
}

// CHECK-LABEL: mo.graph @fusion_diamond
// CHECK-SAME:  (%[[X:.*]]: !mo.tensor<[8], f32>, %[[Y:.*]]: !mo.tensor<[8], f32>)
// CHECK-NEXT:  %[[OUT3:.*]] = map.iter.elementwise<[8]>
// Duplicate (%x, %y) operand slots are uniqued after fusion: one input each.
// CHECK-NEXT:  inputs (%[[X]]: !mo.tensor<[8], f32>, %[[Y]]: !mo.tensor<[8], f32>)
// `%t0` is read three times at one index, so it is computed once.
// CHECK:       %[[XV:.*]] = map.load %[[X]][%{{.*}}] : !mo.tensor<[8], f32>
// CHECK-NEXT:  %[[YV:.*]] = map.load %[[Y]][%{{.*}}] : !mo.tensor<[8], f32>
// CHECK-NEXT:  %[[T0:.*]] = map.call.elementwise @{{.*}}.Add(%{{.*}}, %[[XV]], %[[YV]])
// CHECK-NEXT:  %[[MUL:.*]] = map.call.elementwise @{{.*}}.Mul(%{{.*}}, %[[T0]], %[[T0]])
// CHECK-NEXT:  %[[RELU:.*]] = map.call.elementwise @{{.*}}.ReLU(%{{.*}}, %[[T0]])
// CHECK-NEXT:  map.call.elementwise @{{.*}}.Add(%{{.*}}, %[[MUL]], %[[RELU]])
// CHECK-NEXT:  map.store
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_elementwise<[#mogg.fusion_leaf<"mo.add"{{.*}}, #mogg.fusion_leaf<"mo.relu"{{.*}}, #mogg.fusion_leaf<"mo.mul"{{.*}}, #mogg.fusion_leaf<"mo.add"{{.*}}]>>
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output %[[OUT3]] : !mo.tensor<[8], f32>


mo.graph @add_with_params_fused(%x: !mo.tensor<[4, 3], f32>) -> !mo.tensor<[4, 3], f32> {
  %0 = mo.custom [symbol = "elemwise_add_with_param", device = #M.device_ref<"cpu", 0>, parameters = {"bias" = 1}](%x) : (!mo.tensor<[4, 3], f32>) -> !mo.tensor<[4, 3], f32>
  %1 = mo.custom [symbol = "elemwise_add_with_param", device = #M.device_ref<"cpu", 0>, parameters = {"bias" = 2}](%0) : (!mo.tensor<[4, 3], f32>) -> !mo.tensor<[4, 3], f32>
  mo.output %1 : !mo.tensor<[4, 3], f32>
}

// CHECK-LABEL: mo.graph @add_with_params_fused
// CHECK-SAME:  (%[[X4:.*]]: !mo.tensor<[4, 3], f32>)
// CHECK-NEXT:  %[[OUT4:.*]] = map.iter.elementwise<[4, 3]>
// CHECK-NEXT:  inputs (%[[X4]]: !mo.tensor<[4, 3], f32>)
// CHECK:       %[[X4V:.*]] = map.load %[[X4]][%{{.*}}] : !mo.tensor<[4, 3], f32>
// CHECK-NEXT:  %[[ADD1:.*]] = map.call.elementwise @{{.*}}ElemwiseAddWithParam[bias = 1 : i64](%{{.*}}, %[[X4V]])
// CHECK-NEXT:  map.call.elementwise @{{.*}}ElemwiseAddWithParam[bias = 2 : i64](%{{.*}}, %[[ADD1]])
// CHECK-NEXT:  map.store
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_elementwise<[#mogg.fusion_leaf<"custom__elemwise_add_with_param"{{.*}}, #mogg.fusion_leaf<"custom__elemwise_add_with_param"{{.*}}]>>
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output %[[OUT4]] : !mo.tensor<[4, 3], f32>


mo.graph @basic_add_mul_dynamic<D0>(
  %x: !mo.tensor<[D0], f32>,
  %y: !mo.tensor<[D0], f32>,
  %z: !mo.tensor<[D0], f32>
) -> !mo.tensor<[D0], f32> {
  %tmp = mo.add(%x, %y) : !mo.tensor<[D0], f32>
  %out = mo.mul(%tmp, %z) : !mo.tensor<[D0], f32>
  mo.output %out : !mo.tensor<[D0], f32>
}

// CHECK-LABEL: mo.graph @basic_add_mul_dynamic
// CHECK-SAME:  (%[[X5:.*]]: !mo.tensor<[D0], f32>, %[[Y5:.*]]: !mo.tensor<[D0], f32>, %[[Z5:.*]]: !mo.tensor<[D0], f32>)
// CHECK-NEXT:  %[[OUT5:.*]] = map.iter.elementwise<[D0]>
// CHECK-NEXT:  inputs (%[[X5]]: !mo.tensor<[D0], f32>, %[[Y5]]: !mo.tensor<[D0], f32>, %[[Z5]]: !mo.tensor<[D0], f32>)
// CHECK:       %[[X5V:.*]] = map.load %[[X5]][%{{.*}}] : !mo.tensor<[D0], f32>
// CHECK-NEXT:  %[[Y5V:.*]] = map.load %[[Y5]][%{{.*}}] : !mo.tensor<[D0], f32>
// CHECK-NEXT:  %[[ADD5:.*]] = map.call.elementwise @{{.*}}.Add(%{{.*}}, %[[X5V]], %[[Y5V]])
// CHECK-NEXT:  %[[Z5V:.*]] = map.load %[[Z5]][%{{.*}}] : !mo.tensor<[D0], f32>
// CHECK-NEXT:  map.call.elementwise @{{.*}}.Mul(%{{.*}}, %[[ADD5]], %[[Z5V]])
// CHECK-NEXT:  map.store
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_elementwise<[#mogg.fusion_leaf<"mo.add"{{.*}}, #mogg.fusion_leaf<"mo.mul"{{.*}}]>>
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output %[[OUT5]] : !mo.tensor<[D0], f32>


mo.graph @add_mul_dynamic_multidim<D0, D1>(
  %x: !mo.tensor<[D0, D1], f32>,
  %y: !mo.tensor<[D0, D1], f32>,
  %z: !mo.tensor<[D0, D1], f32>
) -> !mo.tensor<[D0, D1], f32> {
  %tmp = mo.add(%x, %y) : !mo.tensor<[D0, D1], f32>
  %out = mo.mul(%tmp, %z) : !mo.tensor<[D0, D1], f32>
  mo.output %out : !mo.tensor<[D0, D1], f32>
}

// CHECK-LABEL: mo.graph @add_mul_dynamic_multidim
// CHECK-SAME:  (%[[X6:.*]]: !mo.tensor<[D0, D1], f32>, %[[Y6:.*]]: !mo.tensor<[D0, D1], f32>, %[[Z6:.*]]: !mo.tensor<[D0, D1], f32>)
// CHECK-NEXT:  %[[OUT6:.*]] = map.iter.elementwise<[D0, D1]>
// CHECK-NEXT:  inputs (%[[X6]]: !mo.tensor<[D0, D1], f32>, %[[Y6]]: !mo.tensor<[D0, D1], f32>, %[[Z6]]: !mo.tensor<[D0, D1], f32>)
// CHECK:       %[[X6V:.*]] = map.load %[[X6]][%{{.*}}] : !mo.tensor<[D0, D1], f32>
// CHECK-NEXT:  %[[Y6V:.*]] = map.load %[[Y6]][%{{.*}}] : !mo.tensor<[D0, D1], f32>
// CHECK-NEXT:  %[[ADD6:.*]] = map.call.elementwise @{{.*}}.Add(%{{.*}}, %[[X6V]], %[[Y6V]])
// CHECK-NEXT:  %[[Z6V:.*]] = map.load %[[Z6]][%{{.*}}] : !mo.tensor<[D0, D1], f32>
// CHECK-NEXT:  map.call.elementwise @{{.*}}.Mul(%{{.*}}, %[[ADD6]], %[[Z6V]])
// CHECK-NEXT:  map.store
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_elementwise<[#mogg.fusion_leaf<"mo.add"{{.*}}, #mogg.fusion_leaf<"mo.mul"{{.*}}]>>
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output %[[OUT6]] : !mo.tensor<[D0, D1], f32>
