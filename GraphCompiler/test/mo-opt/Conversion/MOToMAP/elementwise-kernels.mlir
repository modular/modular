// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map | FileCheck %s

// CHECK-LABEL: mo.graph @rank1
// CHECK-SAME:  (%[[X:.*]]: !mo.tensor<[8], f32>, %[[Y:.*]]: !mo.tensor<[8], f32>)
// CHECK:       %[[OUT:.*]] = map.iter.elementwise<[8]>
// CHECK-NEXT:  inputs  (%[[X]]: !mo.tensor<[8], f32>, %[[Y]]: !mo.tensor<[8], f32>)
// CHECK:       index   (%[[IDX:.*]]: !map.index<1>) -> !mo.tensor<[8], f32> {
// CHECK-NEXT:  %[[XV:.*]] = map.load %[[X]][%[[IDX]]] : !mo.tensor<[8], f32>
// CHECK-NEXT:  %[[YV:.*]] = map.load %[[Y]][%[[IDX]]] : !mo.tensor<[8], f32>
// CHECK-NEXT:  map.call.elementwise @{{.*}}.Add(%[[IDX]], %[[XV]], %[[YV]]) : !map.index<1>, f32, f32 -> f32
// CHECK-NEXT:  map.store
mo.graph @rank1(%x: !mo.tensor<[8], f32>, %y: !mo.tensor<[8], f32>) -> !mo.tensor<[8], f32> {
  %output = mo.add(%x, %y) : !mo.tensor<[8], f32>
  mo.output %output : !mo.tensor<[8], f32>
}

// CHECK-LABEL: mo.graph @symbolic
// CHECK-SAME:  (%[[X2:.*]]: !mo.tensor<[D0, D1], f32>, %[[Y2:.*]]: !mo.tensor<[D0, D1], f32>)
// CHECK:       %[[OUT2:.*]] = map.iter.elementwise<[D0, D1]>
// CHECK-NEXT:  inputs  (%[[X2]]: !mo.tensor<[D0, D1], f32>, %[[Y2]]: !mo.tensor<[D0, D1], f32>)
// CHECK:       index   (%[[IDX2:.*]]: !map.index<2>) -> !mo.tensor<[D0, D1], f32> {
// CHECK-NEXT:  %[[X2V:.*]] = map.load %[[X2]][%[[IDX2]]] : !mo.tensor<[D0, D1], f32>
// CHECK-NEXT:  %[[Y2V:.*]] = map.load %[[Y2]][%[[IDX2]]] : !mo.tensor<[D0, D1], f32>
// CHECK-NEXT:  map.call.elementwise @{{.*}}.Add(%[[IDX2]], %[[X2V]], %[[Y2V]]) : !map.index<2>, f32, f32 -> f32
// CHECK-NEXT:  map.store
mo.graph @symbolic<D0, D1>(%x: !mo.tensor<[D0, D1], f32>, %y: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
  %output = mo.add(%x, %y) : !mo.tensor<[D0, D1], f32>
  mo.output %output : !mo.tensor<[D0, D1], f32>
}

// CHECK-LABEL: mo.graph @custom_kernel
// CHECK-SAME:  (%[[X3:.*]]: !mo.tensor<[8, 8], f32>, %[[Y3:.*]]: !mo.tensor<[8, 8], f32>)
// CHECK:       %[[OUT3:.*]] = map.iter.elementwise<[8, 8]>
// CHECK-NEXT:  inputs  (%[[X3]]: !mo.tensor<[8, 8], f32>, %[[Y3]]: !mo.tensor<[8, 8], f32>)
// CHECK:       index   (%[[IDX3:.*]]: !map.index<2>) -> !mo.tensor<[8, 8], f32> {
// CHECK-NEXT:  %[[X3V:.*]] = map.load %[[X3]][%[[IDX3]]] : !mo.tensor<[8, 8], f32>
// CHECK-NEXT:  %[[Y3V:.*]] = map.load %[[Y3]][%[[IDX3]]] : !mo.tensor<[8, 8], f32>
// CHECK-NEXT:  map.call.elementwise @{{.*}}AddElementwise(%[[IDX3]], %[[X3V]], %[[Y3V]]) : !map.index<2>, f32, f32 -> f32
// CHECK-NEXT:  map.store
mo.graph @custom_kernel(%x: !mo.tensor<[8, 8], f32>, %y: !mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 8], f32> {
  %output = mo.custom [symbol = "imposter_add_elementwise", device = #M.device_ref<"cpu", 0>](%x, %y) : (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 8], f32>
  mo.output %output : !mo.tensor<[8, 8], f32>
}

// CHECK-LABEL: mo.graph @gpu_device
// CHECK-SAME:  (%[[X4:.*]]: !mo.tensor<[8, 8], f32, gpu:0>, %[[Y4:.*]]: !mo.tensor<[8, 8], f32, gpu:0>)
// CHECK:       %[[OUT4:.*]] = map.iter.elementwise<[8, 8]>
// CHECK-NEXT:  inputs  (%[[X4]]: !mo.tensor<[8, 8], f32, gpu:0>, %[[Y4]]: !mo.tensor<[8, 8], f32, gpu:0>)
// CHECK:       index   (%[[IDX4:.*]]: !map.index<2>) -> !mo.tensor<[8, 8], f32, gpu:0> {
// CHECK-NEXT:  %[[X4V:.*]] = map.load %[[X4]][%[[IDX4]]] : !mo.tensor<[8, 8], f32, gpu:0>
// CHECK-NEXT:  %[[Y4V:.*]] = map.load %[[Y4]][%[[IDX4]]] : !mo.tensor<[8, 8], f32, gpu:0>
// CHECK-NEXT:  map.call.elementwise @{{.*}}.Add(%[[IDX4]], %[[X4V]], %[[Y4V]]) : !map.index<2>, f32, f32 -> f32
// CHECK-NEXT:  map.store %{{.*}}[%[[IDX4]]] = %{{.*}} : !mo.tensor<[8, 8], f32, gpu:0>
mo.graph @gpu_device(%x: !mo.tensor<[8, 8], f32, gpu:0>, %y: !mo.tensor<[8, 8], f32, gpu:0>) -> !mo.tensor<[8, 8], f32, gpu:0> {
  %output = mo.add(%x, %y) : !mo.tensor<[8, 8], f32, gpu:0>
  mo.output %output : !mo.tensor<[8, 8], f32, gpu:0>
}

// CHECK-LABEL: mo.graph @relu_unary
// CHECK-SAME:  (%[[X5:.*]]: !mo.tensor<[8, 8], f16>)
// CHECK:       %[[OUT5:.*]] = map.iter.elementwise<[8, 8]>
// CHECK-NEXT:  inputs  (%[[X5]]: !mo.tensor<[8, 8], f16>)
// CHECK:       index   (%[[IDX5:.*]]: !map.index<2>) -> !mo.tensor<[8, 8], f16> {
// CHECK-NEXT:  %[[X5V:.*]] = map.load %[[X5]][%[[IDX5]]] : !mo.tensor<[8, 8], f16>
// CHECK-NEXT:  map.call.elementwise @{{.*}}.ReLU(%[[IDX5]], %[[X5V]]) : !map.index<2>, f16 -> f16
// CHECK-NEXT:  map.store
mo.graph @relu_unary(%x: !mo.tensor<[8, 8], f16>) -> !mo.tensor<[8, 8], f16> {
  %output = mo.relu(%x) : !mo.tensor<[8, 8], f16>
  mo.output %output : !mo.tensor<[8, 8], f16>
}

// CHECK-LABEL: mo.graph @select_ternary
// CHECK-SAME:  (%[[COND:.*]]: !mo.tensor<[8, 8], bool>, %[[X6:.*]]: !mo.tensor<[8, 8], f32>, %[[Y6:.*]]: !mo.tensor<[8, 8], f32>)
// CHECK:       %[[OUT6:.*]] = map.iter.elementwise<[8, 8]>
// CHECK-NEXT:  inputs  (%[[COND]]: !mo.tensor<[8, 8], bool>, %[[X6]]: !mo.tensor<[8, 8], f32>, %[[Y6]]: !mo.tensor<[8, 8], f32>)
// CHECK:       index   (%[[IDX6:.*]]: !map.index<2>) -> !mo.tensor<[8, 8], f32> {
// CHECK-NEXT:  %[[CONDV:.*]] = map.load %[[COND]][%[[IDX6]]] : !mo.tensor<[8, 8], bool>
// CHECK-NEXT:  %[[X6V:.*]] = map.load %[[X6]][%[[IDX6]]] : !mo.tensor<[8, 8], f32>
// CHECK-NEXT:  %[[Y6V:.*]] = map.load %[[Y6]][%[[IDX6]]] : !mo.tensor<[8, 8], f32>
// CHECK-NEXT:  map.call.elementwise @{{.*}}.Select(%[[IDX6]], %[[CONDV]], %[[X6V]], %[[Y6V]]) : !map.index<2>, i1, f32, f32 -> f32
// CHECK-NEXT:  map.store
mo.graph @select_ternary(%cond: !mo.tensor<[8, 8], bool>, %x: !mo.tensor<[8, 8], f32>, %y: !mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 8], f32> {
  %output = mo.select(%cond, %x, %y) : (!mo.tensor<[8, 8], bool>, !mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 8], f32>
  mo.output %output : !mo.tensor<[8, 8], f32>
}

// CHECK-LABEL: mo.graph @cast_unary
// CHECK-SAME:  (%[[X7:.*]]: !mo.tensor<[8], f32>)
// CHECK:       %[[OUT7:.*]] = map.iter.elementwise<[8]>
// CHECK-NEXT:  inputs  (%[[X7]]: !mo.tensor<[8], f32>)
// CHECK:       index   (%[[IDX7:.*]]: !map.index<1>) -> !mo.tensor<[8], si32> {
// CHECK-NEXT:  %[[X7V:.*]] = map.load %[[X7]][%[[IDX7]]] : !mo.tensor<[8], f32>
// CHECK-NEXT:  map.call.elementwise @{{.*}}.Cast(%[[IDX7]], %[[X7V]]) : !map.index<1>, f32 -> si32
// CHECK-NEXT:  map.store
mo.graph @cast_unary(%x: !mo.tensor<[8], f32>) -> !mo.tensor<[8], si32> {
  %output = mo.cast(%x) : (!mo.tensor<[8], f32>) -> !mo.tensor<[8], si32>
  mo.output %output : !mo.tensor<[8], si32>
}

// CHECK-LABEL: mo.graph @custom_kernel_with_param
// CHECK-SAME:  (%[[X8:.*]]: !mo.tensor<[4, 3], f32>)
// CHECK:       %[[OUT8:.*]] = map.iter.elementwise<[4, 3]>
// CHECK-NEXT:  inputs  (%[[X8]]: !mo.tensor<[4, 3], f32>)
// CHECK:       index   (%[[IDX8:.*]]: !map.index<2>) -> !mo.tensor<[4, 3], f32> {
// CHECK-NEXT:  %[[X8V:.*]] = map.load %[[X8]][%[[IDX8]]] : !mo.tensor<[4, 3], f32>
// CHECK-NEXT:  map.call.elementwise @{{.*}}ElemwiseAddWithParam[bias = 1 : i64](%[[IDX8]], %[[X8V]]) : !map.index<2>, f32 -> f32
// CHECK-NEXT:  map.store
mo.graph @custom_kernel_with_param(%x: !mo.tensor<[4, 3], f32>) -> !mo.tensor<[4, 3], f32> {
  %output = mo.custom [symbol = "elemwise_add_with_param", device = #M.device_ref<"cpu", 0>, parameters = {"bias" = 1}](%x) : (!mo.tensor<[4, 3], f32>) -> !mo.tensor<[4, 3], f32>
  mo.output %output : !mo.tensor<[4, 3], f32>
}

// CHECK-LABEL: mo.graph @mutable_store
// CHECK-SAME:  (%[[BUF9:.*]]: !mo.buffer<[2, 2], f32>, %[[A9:.*]]: !mo.tensor<[2, 2], f32>, %[[B9:.*]]: !mo.tensor<[2, 2], f32>)
// CHECK:       %[[ADD9:.*]] = map.iter.elementwise<[2, 2]>
// CHECK:       map.call.elementwise @{{.*}}.Add(%{{.*}}, %{{.*}}, %{{.*}})
// CHECK:       %[[OUT9:.*]] = map.iter.elementwise<[2, 2]>
// CHECK-NEXT:  inputs  (%[[ADD9]]: !mo.tensor<[2, 2], f32>, %[[BUF9]]: !mo.buffer<[2, 2], f32>)
// CHECK-NEXT:  outputs ()
// CHECK:       index   (%[[IDX9:.*]]: !map.index<2>) chain(%{{.*}}) -> !mo.chain {
// CHECK-NEXT:  %[[ADD9V:.*]] = map.load %[[ADD9]][%[[IDX9]]] : !mo.tensor<[2, 2], f32>
// CHECK-NEXT:  %[[STORED9:.*]] = map.call.elementwise @{{.*}}MutableStore(%[[IDX9]], %[[ADD9V]]) : !map.index<2>, f32 -> f32
// CHECK-NEXT:  map.store %[[BUF9]][%[[IDX9]]] = %[[STORED9]] : !mo.buffer<[2, 2], f32>
// CHECK:       hasWriteEffect = true
mo.graph @mutable_store(%buf: !mo.buffer<[2, 2], f32>, %a: !mo.tensor<[2, 2], f32>, %b: !mo.tensor<[2, 2], f32>) {
  %chain = mo.chain.create()
  %res = mo.add(%a, %b) : !mo.tensor<[2, 2], f32>
  %out_chain = mo.mutable.store(%buf, %res, %chain) : (!mo.buffer<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.chain) -> !mo.chain
  mo.output
}
