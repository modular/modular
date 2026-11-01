// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map | FileCheck %s

// A reduction with a registered monoid lowers to a `map.iter.rowwise`: one
// fold over the input, then a per-row store of each output field. The axis is
// the one input dimension that becomes 1.

// CHECK-LABEL: mo.graph @sumInnerAxis
// CHECK-SAME:  (%[[X:.*]]: !mo.tensor<[8, 128], f32>)
// CHECK:       %[[R:.*]] = map.iter.rowwise<[8, 128], axis = 1>
// CHECK-NEXT:  inputs  (%[[X]]: !mo.tensor<[8, 128], f32>)
// CHECK-NEXT:  outputs (%[[OUT:.*]]: !mo.tensor<[8, 1], f32>)
// CHECK-NEXT:  row     (%[[ROW:.*]]: !map.index<2>) -> !mo.tensor<[8, 1], f32> {
// CHECK-NEXT:    %[[ACC:.*]] = map.iter.reduce @builtin_kernels.monoids.ReduceSum index (%[[I:.*]]: !map.index<2>) -> f32 {
// CHECK-NEXT:      %[[V:.*]] = map.load %[[X]][%[[I]]] : !mo.tensor<[8, 128], f32>
// CHECK-NEXT:      map.yield %[[V]] : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    map.store %[[OUT]][%[[ROW]]] = %[[ACC]] : !mo.tensor<[8, 1], f32>
// CHECK-NEXT:  } {fusionInfo = #{{.*}}}
// CHECK-NEXT:  mo.output %[[R]]
mo.graph @sumInnerAxis(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32> {
  %r = mo.reduce.add(%x) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  mo.output %r : !mo.tensor<[8, 1], f32>
}

// A non-inner axis, with symbolic dims.
// CHECK-LABEL: mo.graph @maxOuterAxis
// CHECK:       map.iter.rowwise<[M, N], axis = 0>
// CHECK:       outputs (%[[OUT2:.*]]: !mo.tensor<[1, N], f32>)
// CHECK:       map.iter.reduce @builtin_kernels.monoids.ReduceMax index ({{.*}}: !map.index<2>) -> f32 {
mo.graph @maxOuterAxis<M, N>(%x: !mo.tensor<[M, N], f32>) -> !mo.tensor<[1, N], f32> {
  %r = mo.reduce.max(%x) : (!mo.tensor<[M, N], f32>) -> !mo.tensor<[1, N], f32>
  mo.output %r : !mo.tensor<[1, N], f32>
}

// A middle axis of a rank-3 tensor.
// CHECK-LABEL: mo.graph @minMiddleAxis
// CHECK:       map.iter.rowwise<[2, 16, 4], axis = 1>
// CHECK:       row     ({{.*}}: !map.index<3>) -> !mo.tensor<[2, 1, 4], f16> {
mo.graph @minMiddleAxis(%x: !mo.tensor<[2, 16, 4], f16>) -> !mo.tensor<[2, 1, 4], f16> {
  %r = mo.reduce.min(%x) : (!mo.tensor<[2, 16, 4], f16>) -> !mo.tensor<[2, 1, 4], f16>
  mo.output %r : !mo.tensor<[2, 1, 4], f16>
}

// A monoid with several output fields has one result per field, each stored
// at the row index. ArgMax's index output is int64.
// CHECK-LABEL: mo.graph @argMax
// CHECK:       %[[R4:.*]]:2 = map.iter.rowwise<[4, 16], axis = 1>
// CHECK:       outputs (%[[OV:.*]]: !mo.tensor<[4, 1], bf16>, %[[OI:.*]]: !mo.tensor<[4, 1], si64>)
// CHECK:       %[[VALS:.*]], %[[IDXS:.*]] = map.iter.reduce @builtin_kernels.monoids.ArgMax index ({{.*}}: !map.index<2>) -> (bf16, si64) {
// CHECK:       map.yield {{.*}} : bf16
// CHECK:       map.store %[[OV]][{{.*}}] = %[[VALS]] : !mo.tensor<[4, 1], bf16>
// CHECK-NEXT:  map.store %[[OI]][{{.*}}] = %[[IDXS]] : !mo.tensor<[4, 1], si64>
// CHECK:       mo.output %[[R4]]#0, %[[R4]]#1
mo.graph @argMax(%x: !mo.tensor<[4, 16], bf16>) -> (!mo.tensor<[4, 1], bf16>, !mo.tensor<[4, 1], si64>) {
  %v, %i = mo.custom [symbol = "mo.reduce.arg_max.tmp", device = #M.device_ref<"cpu", 0>](%x) : (!mo.tensor<[4, 16], bf16>) -> (!mo.tensor<[4, 1], bf16>, !mo.tensor<[4, 1], si64>)
  mo.output %v, %i : !mo.tensor<[4, 1], bf16>, !mo.tensor<[4, 1], si64>
}

// CHECK-LABEL: mo.graph @welford
// CHECK:       map.iter.reduce @builtin_kernels.monoids.Welford index ({{.*}}: !map.index<2>) -> (f32, f32, f32) {
// CHECK-COUNT-3: map.store
mo.graph @welford(%x: !mo.tensor<[4, 16], f32>) -> (!mo.tensor<[4, 1], f32>, !mo.tensor<[4, 1], f32>, !mo.tensor<[4, 1], f32>) {
  %count, %mean, %m2 = mo.custom [symbol = "mo.reduce.mean_var.tmp", device = #M.device_ref<"cpu", 0>](%x) : (!mo.tensor<[4, 16], f32>) -> (!mo.tensor<[4, 1], f32>, !mo.tensor<[4, 1], f32>, !mo.tensor<[4, 1], f32>)
  mo.output %count, %mean, %m2 : !mo.tensor<[4, 1], f32>, !mo.tensor<[4, 1], f32>, !mo.tensor<[4, 1], f32>
}
