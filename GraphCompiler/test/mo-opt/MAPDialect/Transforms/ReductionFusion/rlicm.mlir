// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_pkg" --mo-to-map --map-fuse-kernels --map-rlicm | FileCheck %s

// After fusion, the work of a map phase that is the same for the whole row
// moves to row level: an op on row values, and a load whose index ignores the
// reduced axis. A kernel that reads its element's index stays in the phase.

// The rsqrt of the broadcast row value is the same for the whole row, so it
// runs once per row.
// CHECK-LABEL: mo.graph @hoisted
// CHECK:       row     (%[[ROW5:.*]]: !map.index<2>)
// CHECK:         %[[ACC5:.*]] = map.iter.reduce
// CHECK:         %[[INV5:.*]] = map.call.elementwise @builtin_kernels.elementwise.Rsqrt(%[[ROW5]], %[[ACC5]])
// CHECK-NEXT:    map.iter.map index (%[[I5:.*]]: !map.index<2>) {
// CHECK-NEXT:      %[[XV5:.*]] = map.load %{{.*}}[%[[I5]]]
// CHECK-NEXT:      map.call.elementwise @builtin_kernels.elementwise.Mul(%[[I5]], %[[XV5]], %[[INV5]])
mo.graph @hoisted(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 128], f32> {
  %s = mo.reduce.add(%x) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  %sb = mo.static.broadcast_to(%s) : !mo.tensor<[8, 1], f32> -> !mo.tensor<[8, 128], f32>
  %inv = mo.rsqrt(%sb) : !mo.tensor<[8, 128], f32>
  %y = mo.mul(%x, %inv) : !mo.tensor<[8, 128], f32>
  mo.output %y : !mo.tensor<[8, 128], f32>
}

// A per-row side input, read through a broadcast, and the product with the
// sum move to row level: the phase only scales by the result.
// CHECK-LABEL: mo.graph @rowScale
// CHECK-SAME:  (%{{.*}}: !mo.tensor<[8, 128], f32>, %[[B7:.*]]: !mo.tensor<[8, 1], f32>)
// CHECK:       row     (%[[ROW7:.*]]: !map.index<2>)
// CHECK:         %[[ACC7:.*]] = map.iter.reduce
// CHECK:         %[[BI7:.*]] = map.index.broadcast<#map.index.broadcast.spec<[8, 1] -> [8, 128]>> (%[[ROW7]])
// CHECK-NEXT:    %[[BV7:.*]] = map.load %[[B7]][%[[BI7]]]
// CHECK-NEXT:    %[[T7:.*]] = map.call.elementwise @builtin_kernels.elementwise.Mul(%[[ROW7]], %[[ACC7]], %[[BV7]])
// CHECK-NEXT:    map.iter.map index (%[[I7:.*]]: !map.index<2>) {
// CHECK-NEXT:      %[[XV7:.*]] = map.load %{{.*}}[%[[I7]]]
// CHECK-NEXT:      map.call.elementwise @builtin_kernels.elementwise.Mul(%[[I7]], %[[XV7]], %[[T7]])
mo.graph @rowScale(%x: !mo.tensor<[8, 128], f32>, %b: !mo.tensor<[8, 1], f32>) -> !mo.tensor<[8, 128], f32> {
  %s = mo.reduce.add(%x) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  %sb = mo.static.broadcast_to(%s) : !mo.tensor<[8, 1], f32> -> !mo.tensor<[8, 128], f32>
  %bb = mo.static.broadcast_to(%b) : !mo.tensor<[8, 1], f32> -> !mo.tensor<[8, 128], f32>
  %t = mo.mul(%sb, %bb) : !mo.tensor<[8, 128], f32>
  %y = mo.mul(%x, %t) : !mo.tensor<[8, 128], f32>
  mo.output %y : !mo.tensor<[8, 128], f32>
}

// A kernel that reads its element's index stays in the phase, even when its
// operands are row values.
// CHECK-LABEL: mo.graph @indexedCall
// CHECK:         map.iter.reduce
// CHECK-NOT:     map.call.elementwise
// CHECK:         map.iter.map index (%[[I8:.*]]: !map.index<2>) {
// CHECK:           map.call.elementwise @{{.*}}(%[[I8]],
mo.graph @indexedCall(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 128], f32> {
  %s = mo.reduce.add(%x) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  %sb = mo.static.broadcast_to(%s) : !mo.tensor<[8, 1], f32> -> !mo.tensor<[8, 128], f32>
  %z = mo.custom [symbol = "elemwise_add_with_index", device = #M.device_ref<"cpu", 0>](%sb) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 128], f32>
  %y = mo.mul(%x, %z) : !mo.tensor<[8, 128], f32>
  mo.output %y : !mo.tensor<[8, 128], f32>
}

// The same in a concat select's regions, which read their row: rotate-half
// scales each partner by the row's rsqrt, which runs once per row.
// CHECK-LABEL: mo.graph @hoistedFromSelect
// CHECK:         map.iter.reduce
// CHECK:         map.call.elementwise @builtin_kernels.elementwise.Rsqrt
// CHECK:         map.iter.map
// CHECK:           map.index.concat.select
// CHECK-NOT:       Rsqrt
// CHECK:       mo.output
mo.graph @hoistedFromSelect(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 128], f32> {
  %s = mo.reduce.add(%x) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  %sb = mo.static.broadcast_to(%s) : !mo.tensor<[8, 1], f32> -> !mo.tensor<[8, 128], f32>
  %inv = mo.rsqrt(%sb) : !mo.tensor<[8, 128], f32>
  %y = mo.mul(%x, %inv) : !mo.tensor<[8, 128], f32>
  %start1 = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %start2 = mo.constant <#M.dense_array<0, 64> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop1 = mo.constant <#M.dense_array<8, 64> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop2 = mo.constant <#M.dense_array<8, 128> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %y1 = mo.slice(%y, %start1, %stop1, %step) : (!mo.tensor<[8, 128], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[8, 64], f32>
  %y2 = mo.slice(%y, %start2, %stop2, %step) : (!mo.tensor<[8, 128], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[8, 64], f32>
  %ny2 = mo.negative(%y2) : !mo.tensor<[8, 64], f32>
  %r = mo.concat[1] (%ny2, %y1) : (!mo.tensor<[8, 64], f32>, !mo.tensor<[8, 64], f32>) -> !mo.tensor<[8, 128], f32>
  mo.output %r : !mo.tensor<[8, 128], f32>
}
