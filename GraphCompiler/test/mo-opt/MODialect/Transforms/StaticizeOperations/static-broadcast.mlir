// RUN: mo-opt %s --mo-propagate-shapes --canonicalize | FileCheck %s

// CHECK-LABEL: mo.graph @static_broadcast0
//  CHECK-SAME:   [[INPUT:%.*]]: !mo.tensor<[], si64>
//  CHECK-SAME: -> !mo.tensor<[1], si64> {
mo.graph @static_broadcast0<() -> out>(
  %input: !mo.tensor<[], si64>
) -> !mo.tensor<[out], si64> {

  %one = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>
  %shape = mo.static.broadcast_to(%one): !mo.tensor<[], si64> -> !mo.tensor<[1], si64>

  // CHECK-NEXT: [[RES:%.*]] = mo.static.reshape([[INPUT]]) : {{.*}} -> !mo.tensor<[1], si64>
  %res = mo.broadcast_to<()->D0>(%input, %shape) : (
    !mo.tensor<[], si64>, !mo.tensor<[1], si64>
  ) -> !mo.tensor<[D0], si64>

  // CHECK-NEXT: mo.output [[RES]]
  mo.output<D0> %res : !mo.tensor<[D0], si64>
}


// CHECK-LABEL: mo.graph @static_broadcast1
//  CHECK-SAME:   [[INPUT:%.*]]: !mo.tensor<[], si64>
//  CHECK-SAME: -> !mo.tensor<[1, 1], si64> {
mo.graph @static_broadcast1<() -> out0,out1>(
  %input: !mo.tensor<[], si64>
) -> !mo.tensor<[out0,out1], si64> {

  %one = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>
  %shape = mo.static.broadcast_to(%one): !mo.tensor<[], si64> -> !mo.tensor<[2], si64>

  // CHECK-NEXT: [[RES:%.*]] = mo.static.reshape([[INPUT]]) : {{.*}} -> !mo.tensor<[1, 1], si64>
  %res = mo.broadcast_to<()->D0,D1>(%input, %shape) : (
    !mo.tensor<[], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[D0,D1], si64>

  // CHECK-NEXT: mo.output [[RES]]
  mo.output<D0,D1> %res : !mo.tensor<[D0,D1], si64>
}


// CHECK-LABEL: mo.graph @static_broadcast2
//  CHECK-SAME:   [[INPUT:%.*]]: !mo.tensor<[], si64>
//  CHECK-SAME: -> !mo.tensor<[5, 5], si64> {
mo.graph @static_broadcast2<() -> out0,out1>(
  %input: !mo.tensor<[], si64>
) -> !mo.tensor<[out0,out1], si64> {

  %five = mo.constant <#M.dense_array<5> : tensor<si64>> : !mo.tensor<[], si64>
  %shape = mo.static.broadcast_to(%five): !mo.tensor<[], si64> -> !mo.tensor<[2], si64>

  // CHECK-NEXT: [[RES:%.*]] = mo.static.broadcast_to([[INPUT]]) : {{.*}} -> !mo.tensor<[5, 5], si64>
  %res = mo.broadcast_to<()->D0,D1>(%input, %shape) : (
    !mo.tensor<[], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[D0,D1], si64>

  // CHECK-NEXT: mo.output [[RES]]
  mo.output<D0,D1> %res : !mo.tensor<[D0,D1], si64>
}


// CHECK-LABEL: mo.graph @static_broadcast_1d_input
//  CHECK-SAME:   [[INPUT:%.*]]: !mo.tensor<[], si64>
//  CHECK-SAME: -> !mo.tensor<[3, 3], si64> {
mo.graph @static_broadcast_1d_input<() -> out0, out1>(
  %input: !mo.tensor<[], si64>
) -> !mo.tensor<[out0, out1], si64> {

  %three = mo.constant <#M.dense_array<3> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %shape = mo.static.broadcast_to(%three): !mo.tensor<[1], si64> -> !mo.tensor<[2], si64>

  // CHECK-NEXT: [[RES:%.*]] = mo.static.broadcast_to([[INPUT]]) : {{.*}} -> !mo.tensor<[3, 3], si64>
  %res = mo.broadcast_to<()-> D0, D1>(%input, %shape) : (
    !mo.tensor<[], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[D0, D1], si64>

  // CHECK-NEXT: mo.output [[RES]]
  mo.output<D0, D1> %res : !mo.tensor<[D0, D1], si64>
}
