// RUN: mo-opt %s --mo-symbolic-optimize --mo-propagate-shapes --canonicalize -cse | FileCheck %s

// CHECK-LABEL: mo.graph @static_shape
// CHECK-SAME: <() -> [[M:.*]]>(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[[[M]], 1], si64> {
mo.graph @static_shape<() -> M>(%arg0: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[M, 1], si64> {
  // CHECK-DAG: %[[RANGE:.*]] = mo.constant <#M.dense_array<0, 6, 12, 18, 24> : tensor<5x1xsi64>> : !mo.tensor<[5, 1], si64>
  // CHECK-DAG: %[[PERM:.*]] = mo.constant <#M.dense_array<1, 0> : !M.array<2xsi64>> : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[FLATARG:.*]] = mo.static.reshape(%[[ARG0]]) : !mo.tensor<[2, 3], f32> -> !mo.tensor<[6], f32>
  // CHECK-DAG: %[[WHERE:.*]] = mo.arg_nonzero<() -> [[W:.*]]>(%[[FLATARG]]) : (!mo.tensor<[6], f32>) -> !mo.tensor<[[[W]], 1], si64>
  // CHECK-DAG: %[[VAL5:.*]] = mo.static.broadcast_to(%[[WHERE]]) : !mo.tensor<[[[W]], 1], si64> -> !mo.tensor<[[[W]], 5], si64>
  // CHECK-DAG: %[[VAL6:.*]] = mo.static.broadcast_to(%[[RANGE]]) : !mo.tensor<[5, 1], si64> -> !mo.tensor<[5, [[W]]], si64>
  // CHECK-DAG: %[[WHERETP:.*]] = mo.transpose(%[[VAL5]], %[[PERM]]) : (!mo.tensor<[[[W]], 5], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[5, [[W]]], si64>
  // CHECK-DAG: %[[RES:.*]] = mo.add(%[[WHERETP]], %[[VAL6]]) : !mo.tensor<[5, [[W]]], si64>
  // CHECK-DAG: %[[OLDRES:.*]] = mo.static.reshape(%[[RES]]) : !mo.tensor<[5, [[W]]], si64> -> !mo.tensor<[mul_no_wrap([[W]], 5), 1], si64>
  %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[2, 3], f32> -> !mo.tensor<[5, 2, 3], f32>
  %1 = mo.static.reshape(%0) : !mo.tensor<[5, 2, 3], f32> -> !mo.tensor<[30], f32>
  %2 = mo.arg_nonzero<() -> K>(%1) : (!mo.tensor<[30], f32>) -> !mo.tensor<[K, 1], si64>
  // CHECK: mo.output<mul_no_wrap([[W]], 5)> %[[OLDRES]] : !mo.tensor<[mul_no_wrap([[W]], 5), 1], si64>
  mo.output<K> %2 : !mo.tensor<[K, 1], si64>
}

// CHECK-LABEL: mo.graph @static_shape_with_device
// CHECK-SAME: <() -> [[M:.*]]>(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32, dev:0>) -> !mo.tensor<[[[M]], 1], si64, dev:0> {
mo.graph @static_shape_with_device<() -> M>(%arg0: !mo.tensor<[2, 3], f32, dev:0>) -> !mo.tensor<[M, 1], si64, dev:0> {
  // CHECK-DAG: %[[RANGE:.*]] = mo.constant <#M.dense_array<0, 6, 12, 18, 24> : tensor<5x1xsi64>> : !mo.tensor<[5, 1], si64, dev:0>
  // CHECK-DAG: %[[PERM:.*]] = mo.constant <#M.dense_array<1, 0> : !M.array<2xsi64>> : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[FLATARG:.*]] = mo.static.reshape(%[[ARG0]]) : !mo.tensor<[2, 3], f32, dev:0> -> !mo.tensor<[6], f32, dev:0>
  // CHECK-DAG: %[[WHERE:.*]] = mo.arg_nonzero<() -> [[W:.*]]>(%[[FLATARG]]) : (!mo.tensor<[6], f32, dev:0>) -> !mo.tensor<[[[W]], 1], si64, dev:0>
  // CHECK-DAG: %[[VAL5:.*]] = mo.static.broadcast_to(%[[WHERE]]) : !mo.tensor<[[[W]], 1], si64, dev:0> -> !mo.tensor<[[[W]], 5], si64, dev:0>
  // CHECK-DAG: %[[VAL6:.*]] = mo.static.broadcast_to(%[[RANGE]]) : !mo.tensor<[5, 1], si64, dev:0> -> !mo.tensor<[5, [[W]]], si64, dev:0>
  // CHECK-DAG: %[[WHERETP:.*]] = mo.transpose(%[[VAL5]], %[[PERM]]) : (!mo.tensor<[[[W]], 5], si64, dev:0>, !mo.tensor<[2], si64>) -> !mo.tensor<[5, [[W]]], si64, dev:0>
  // CHECK-DAG: %[[RES:.*]] = mo.add(%[[WHERETP]], %[[VAL6]]) : !mo.tensor<[5, [[W]]], si64, dev:0>
  // CHECK-DAG: %[[OLDRES:.*]] = mo.static.reshape(%[[RES]]) : !mo.tensor<[5, [[W]]], si64, dev:0> -> !mo.tensor<[mul_no_wrap([[W]], 5), 1], si64, dev:0>
  %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[2, 3], f32, dev:0> -> !mo.tensor<[5, 2, 3], f32, dev:0>
  %1 = mo.static.reshape(%0) : !mo.tensor<[5, 2, 3], f32, dev:0> -> !mo.tensor<[30], f32, dev:0>
  %2 = mo.arg_nonzero<() -> K>(%1) : (!mo.tensor<[30], f32, dev:0>) -> !mo.tensor<[K, 1], si64, dev:0>
  // CHECK: mo.output<mul_no_wrap([[W]], 5)> %[[OLDRES]] : !mo.tensor<[mul_no_wrap([[W]], 5), 1], si64, dev:0>
  mo.output<K> %2 : !mo.tensor<[K, 1], si64, dev:0>
}

// CHECK-LABEL: mo.graph @dynamic_shape
// CHECK-SAME: <[[A:.*]], [[B:.*]] -> [[M:.*]]>(%[[ARG:.*]]: !mo.tensor<[2, [[A]]], f32>) -> !mo.tensor<[[[M]], 1], si64> {
mo.graph @dynamic_shape<A, B -> M>(%arg0: !mo.tensor<[2, A], f32>) -> !mo.tensor<[M, 1], si64> {
  // CHECK-DAG: %[[C0:.*]] = mosh.param.to_value = <mul_no_wrap([[A]], [[A]], [[B]], 2)>
  // CHECK-DAG: %[[C1:.*]] = mosh.param.to_value = <mul_no_wrap([[A]], 2)>
  // CHECK-DAG: %[[START:.*]] = mo.constant <#M.dense_array<0> : !M.array<1xsi64>> : !mo.tensor<[], si64>
  // CHECK-DAG: %[[LIMIT:.*]] = mo.index.to_tensor(%[[C0]]) -> !mo.tensor<[], si64>
  // CHECK-DAG: %[[STEP:.*]] = mo.index.to_tensor(%[[C1]]) -> !mo.tensor<[], si64>
  // CHECK-DAG: %[[RANGE:.*]] = mo.range(%[[START]], %[[LIMIT]], %[[STEP]]) : (!mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[mul_no_wrap([[A]], [[B]])], si64>
  // CHECK-DAG: %[[PERM:.*]] = mo.constant <#M.dense_array<1, 0> : !M.array<2xsi64>> : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[FLATARG:.*]] = mo.static.reshape(%[[ARG0]]) : !mo.tensor<[2, [[A]]], f32> -> !mo.tensor<[mul_no_wrap([[A]], 2)], f32>
  // CHECK-DAG: %[[WHERE:.*]] = mo.arg_nonzero<() -> [[W:.*]]>(%[[FLATARG]]) : (!mo.tensor<[mul_no_wrap([[A]], 2)], f32>) -> !mo.tensor<[[[W]], 1], si64>
  // CHECK-DAG: %[[RNG2:.*]] = mo.static.reshape(%[[RANGE]]) : !mo.tensor<[mul_no_wrap([[A]], [[B]])], si64> -> !mo.tensor<[mul_no_wrap([[A]], [[B]]), 1], si64>
  // CHECK-DAG: %[[VAL5:.*]] = mo.static.broadcast_to(%[[WHERE]]) : !mo.tensor<[[[W]], 1], si64> -> !mo.tensor<[[[W]], mul_no_wrap([[A]], [[B]])], si64>
  // CHECK-DAG: %[[VAL6:.*]] = mo.static.broadcast_to(%[[RNG2]]) : !mo.tensor<[mul_no_wrap([[A]], [[B]]), 1], si64> -> !mo.tensor<[mul_no_wrap([[A]], [[B]]), [[W]]], si64>
  // CHECK-DAG: %[[WHERETP:.*]] = mo.transpose(%[[VAL5]], %[[PERM]]) : (!mo.tensor<[[[W]], mul_no_wrap([[A]], [[B]])], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[mul_no_wrap([[A]], [[B]]), [[W]]], si64>
  // CHECK-DAG: %[[RES:.*]] = mo.add(%[[WHERETP]], %[[VAL6]]) : !mo.tensor<[mul_no_wrap([[A]], [[B]]), [[W]]], si64>
  // CHECK-DAG: %[[OLDRES:.*]] = mo.static.reshape(%[[RES]]) : !mo.tensor<[mul_no_wrap([[A]], [[B]]), [[W]]], si64> -> !mo.tensor<[mul_no_wrap([[A]], [[B]], [[W]]), 1], si64>
  %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[2, A], f32> -> !mo.tensor<[B, A, 2, A], f32>
  kgen.param.declare D: scalar<si64> = <mul_no_wrap(B, A, 2, A)>
  %1 = mo.static.reshape(%0) : !mo.tensor<[B, A, 2, A], f32> -> !mo.tensor<[D], f32>
  %2 = mo.arg_nonzero<() -> K>(%1) : (!mo.tensor<[D], f32>) -> !mo.tensor<[K, 1], si64>
  // CHECK: mo.output<mul_no_wrap([[A]], [[B]], [[W]])> %[[OLDRES]] : !mo.tensor<[mul_no_wrap([[A]], [[B]], [[W]]), 1], si64>
  mo.output<K> %2 : !mo.tensor<[K, 1], si64>
}

// CHECK-LABEL: mo.graph @non_flat_input
// CHECK-SAME: <() -> M>(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[M, 2], si64> {
mo.graph @non_flat_input<() -> M>(%arg0: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[M, 2], si64> {
  // CHECK-NEXT: %[[BCASTARG0:.*]] = mo.static.broadcast_to(%[[ARG0]]) : !mo.tensor<[2, 3], f32> -> !mo.tensor<[5, 2, 3], f32>
  // CHECK-NEXT: %[[RESHAPE:.*]] = mo.static.reshape(%[[BCASTARG0]]) : !mo.tensor<[5, 2, 3], f32> -> !mo.tensor<[5, 6], f32>
  %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[2, 3], f32> -> !mo.tensor<[5, 2, 3], f32>
  %1 = mo.static.reshape(%0) : !mo.tensor<[5, 2, 3], f32> -> !mo.tensor<[5, 6], f32>
  // CHECK-NEXT: %[[RES:.*]] = mo.arg_nonzero<() -> [[K:.*]]>(%[[RESHAPE]]) : (!mo.tensor<[5, 6], f32>) -> !mo.tensor<[[[K]], 2], si64>
  %2 = mo.arg_nonzero<() -> K>(%1) : (!mo.tensor<[5, 6], f32>) -> !mo.tensor<[K, 2], si64>
  // CHECK-NEXT: mo.output<[[K]]> %[[RES]] : !mo.tensor<[[[K]], 2], si64>
  mo.output<K> %2 : !mo.tensor<[K, 2], si64>
}

// CHECK-LABEL: mo.graph @no_reshape
// CHECK-SAME: <[[A:.*]], [[B:.*]] -> [[M:.*]]>(%[[ARG0:.*]]: !mo.tensor<[[[A]]], f32>) -> !mo.tensor<[[[M]], 1], si64> {
mo.graph @no_reshape<A, B -> M>(%arg0: !mo.tensor<[A], f32>) -> !mo.tensor<[M, 1], si64> {
  // CHECK-NEXT: %[[BCASTARG0:.*]] = mo.static.broadcast_to(%[[ARG0]]) : !mo.tensor<[[[A]]], f32> -> !mo.tensor<[[[B]]], f32>
  %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[A], f32> -> !mo.tensor<[B], f32>
  // CHECK-NEXT: %[[RES:.*]] = mo.arg_nonzero<() -> [[K:.*]]>(%[[BCASTARG0]]) : (!mo.tensor<[[[B]]], f32>) -> !mo.tensor<[[[K]], 1], si64>
  %1 = mo.arg_nonzero<() -> K>(%0) : (!mo.tensor<[B], f32>) -> !mo.tensor<[K, 1], si64>
  // CHECK-NEXT: mo.output<[[K]]> %[[RES]] : !mo.tensor<[[[K]], 1], si64>
  mo.output<K> %1 : !mo.tensor<[K, 1], si64>
}
