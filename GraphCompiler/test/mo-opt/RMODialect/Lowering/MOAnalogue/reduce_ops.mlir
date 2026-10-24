// RUN: mo-opt --split-input-file --mo-legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @mean_dyn_dims<D0, D1, D2, D3 -> D4, D5, D6, D7>(%arg0: !mo.tensor<[D0, D1, D2, D3], f32>, %len: !mo.tensor<[], f32>) -> !mo.tensor<[D4, D5, D6, D7], f32> {
  %sum = rmo.mo.reduce.add<() -> D8, D9, D10, D11>(%arg0) [axis = 1] : (!mo.tensor<[D0, D1, D2, D3], f32>) -> !mo.tensor<[D8, D9, D10, D11], f32>
  %1 = rmo.div(%sum, %len) : (!mo.tensor<[D8, D9, D10, D11], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[D8, D9, D10, D11], f32>
  mo.output<D8, D9, D10, D11> %1 : !mo.tensor<[D8, D9, D10, D11], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @mean_dyn_dims<[[D0:[A-Z0-9]*]], [[D1:[A-Z0-9]*]], [[D2:[A-Z0-9]*]], [[D3:[A-Z0-9]*]] -> [[D4:[A-Z0-9]*]], [[D5:[A-Z0-9]*]], [[D6:[A-Z0-9]*]], [[D7:[A-Z0-9]*]]>([[ARG0:%.*]]: !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>, [[LEN:%.*]]: !mo.tensor<[], f32>) -> !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.reduce.add([[ARG0]]) : (!mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>) -> !mo.tensor<[[[D0]], 1, [[D2]], [[D3]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[D8:[A-Za-z0-9_]*]], [[D9:[A-Za-z0-9_]*]], [[D10:[A-Za-z0-9_]*]], [[D11:[A-Za-z0-9_]*]]>([[V0]]) [message = ""] : (!mo.tensor<[[[D0]], 1, [[D2]], [[D3]]], f32>) -> !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>
//       CHECK:   [[SUM_B:%.*]] = mo.broadcast_to([[V1]], {{.*}}) : (!mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>
//  CHECK-NEXT:   [[LEN_B:%.*]] = mo.broadcast_to([[LEN]], {{.*}}) : (!mo.tensor<[], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>
//  CHECK-NEXT:   [[MEAN:%.*]] = mo.div([[SUM_B]], [[LEN_B]]) : !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>
//  CHECK-NEXT:   mo.output<[[D8]], [[D9]], [[D10]], [[D11]]> [[MEAN]] : !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>

// -----

mo.graph @mean_dyn_const_axis<D0, D1, D2, D3 -> D4, D5, D6, D7>(%arg0: !mo.tensor<[D0, D1, D2, D3], f32>, %len: !mo.tensor<[], f32>) -> !mo.tensor<[D4, D5, D6, D7], f32> {
  %sum = rmo.mo.reduce.add<() -> D8, D9, D10, D11>(%arg0) [axis = 1] : (!mo.tensor<[D0, D1, D2, D3], f32>) -> !mo.tensor<[D8, D9, D10, D11], f32>
  %1 = rmo.div(%sum, %len) : (!mo.tensor<[D8, D9, D10, D11], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[D8, D9, D10, D11], f32>
  mo.output<D8, D9, D10, D11> %1 : !mo.tensor<[D8, D9, D10, D11], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @mean_dyn_const_axis<[[D0:[A-Z0-9]*]], [[D1:[A-Z0-9]*]], [[D2:[A-Z0-9]*]], [[D3:[A-Z0-9]*]] -> [[D4:[A-Z0-9]*]], [[D5:[A-Z0-9]*]], [[D6:[A-Z0-9]*]], [[D7:[A-Z0-9]*]]>([[ARG0:%.*]]: !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>, [[LEN:%.*]]: !mo.tensor<[], f32>) -> !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.reduce.add([[ARG0]]) : (!mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>) -> !mo.tensor<[[[D0]], 1, [[D2]], [[D3]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[D8:[A-Za-z0-9_]*]], [[D9:[A-Za-z0-9_]*]], [[D10:[A-Za-z0-9_]*]], [[D11:[A-Za-z0-9_]*]]>([[V0]]) [message = ""] : (!mo.tensor<[[[D0]], 1, [[D2]], [[D3]]], f32>) -> !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>
//       CHECK:   [[SUM_B:%.*]] = mo.broadcast_to([[V1]], {{.*}}) : (!mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>
//  CHECK-NEXT:   [[LEN_B:%.*]] = mo.broadcast_to([[LEN]], {{.*}}) : (!mo.tensor<[], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>
//  CHECK-NEXT:   [[MEAN:%.*]] = mo.div([[SUM_B]], [[LEN_B]]) : !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>
//  CHECK-NEXT:   mo.output<[[D8]], [[D9]], [[D10]], [[D11]]> [[MEAN]] : !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>

// -----


// Testing all reduce ops

mo.graph @reduce_mul_dyn_dims<D0, D1, D2, D3 -> D4, D5, D6, D7>(%arg0: !mo.tensor<[D0, D1, D2, D3], f32>) -> !mo.tensor<[D4, D5, D6, D7], f32> {
  %1 = rmo.mo.reduce.mul<() -> D8, D9, D10, D11>(%arg0) [axis = 1] : (!mo.tensor<[D0, D1, D2, D3], f32>) -> !mo.tensor<[D8, D9, D10, D11], f32>
  mo.output<D8, D9, D10, D11> %1 : !mo.tensor<[D8, D9, D10, D11], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @reduce_mul_dyn_dims<[[D0:[A-Z0-9]*]], [[D1:[A-Z0-9]*]], [[D2:[A-Z0-9]*]], [[D3:[A-Z0-9]*]] -> [[D4:[A-Z0-9]*]], [[D5:[A-Z0-9]*]], [[D6:[A-Z0-9]*]], [[D7:[A-Z0-9]*]]>([[ARG0:%.*]]: !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>) -> !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.reduce.mul<() -> [[D8:[A-Za-z0-9_]*]], [[D9:[A-Za-z0-9_]*]], [[D10:[A-Za-z0-9_]*]], [[D11:[A-Za-z0-9_]*]]>([[ARG0]]) [axis = 1] : (!mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>) -> !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>
//  CHECK-NEXT:   mo.output<[[D8]], [[D9]], [[D10]], [[D11]]> [[V0]] : !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>

// -----

mo.graph @reduce_add_dyn_dims<D0, D1, D2, D3 -> D4, D5, D6, D7>(%arg0: !mo.tensor<[D0, D1, D2, D3], f32>) -> !mo.tensor<[D4, D5, D6, D7], f32> {
  %1 = rmo.mo.reduce.add<() -> D8, D9, D10, D11>(%arg0) [axis = 1] : (!mo.tensor<[D0, D1, D2, D3], f32>) -> !mo.tensor<[D8, D9, D10, D11], f32>
  mo.output<D8, D9, D10, D11> %1 : !mo.tensor<[D8, D9, D10, D11], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @reduce_add_dyn_dims<[[D0:[A-Z0-9]*]], [[D1:[A-Z0-9]*]], [[D2:[A-Z0-9]*]], [[D3:[A-Z0-9]*]] -> [[D4:[A-Z0-9]*]], [[D5:[A-Z0-9]*]], [[D6:[A-Z0-9]*]], [[D7:[A-Z0-9]*]]>([[ARG0:%.*]]: !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>) -> !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.reduce.add([[ARG0]]) : (!mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>) -> !mo.tensor<[[[D0]], 1, [[D2]], [[D3]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[D8:[A-Za-z0-9_]*]], [[D9:[A-Za-z0-9_]*]], [[D10:[A-Za-z0-9_]*]], [[D11:[A-Za-z0-9_]*]]>([[V0]]) [message = ""] : (!mo.tensor<[[[D0]], 1, [[D2]], [[D3]]], f32>) -> !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>
//  CHECK-NEXT:   mo.output<[[D8]], [[D9]], [[D10]], [[D11]]> [[V1]] : !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>

// -----

mo.graph @reduce_min_dyn_dims<D0, D1, D2, D3 -> D4, D5, D6, D7>(%arg0: !mo.tensor<[D0, D1, D2, D3], f32>) -> !mo.tensor<[D4, D5, D6, D7], f32> {
  %1 = rmo.mo.reduce.min<() -> D8, D9, D10, D11>(%arg0) [axis = 1] : (!mo.tensor<[D0, D1, D2, D3], f32>) -> !mo.tensor<[D8, D9, D10, D11], f32>
  mo.output<D8, D9, D10, D11> %1 : !mo.tensor<[D8, D9, D10, D11], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @reduce_min_dyn_dims<[[D0:[A-Z0-9]*]], [[D1:[A-Z0-9]*]], [[D2:[A-Z0-9]*]], [[D3:[A-Z0-9]*]] -> [[D4:[A-Z0-9]*]], [[D5:[A-Z0-9]*]], [[D6:[A-Z0-9]*]], [[D7:[A-Z0-9]*]]>([[ARG0:%.*]]: !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>) -> !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.reduce.min([[ARG0]]) : (!mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>) -> !mo.tensor<[[[D0]], 1, [[D2]], [[D3]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[D8:[A-Za-z0-9_]*]], [[D9:[A-Za-z0-9_]*]], [[D10:[A-Za-z0-9_]*]], [[D11:[A-Za-z0-9_]*]]>([[V0]]) [message = ""] : (!mo.tensor<[[[D0]], 1, [[D2]], [[D3]]], f32>) -> !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>
//  CHECK-NEXT:   mo.output<[[D8]], [[D9]], [[D10]], [[D11]]> [[V1]] : !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>

// -----

mo.graph @reduce_max_dyn_dims<D0, D1, D2, D3 -> D4, D5, D6, D7>(%arg0: !mo.tensor<[D0, D1, D2, D3], f32>) -> !mo.tensor<[D4, D5, D6, D7], f32> {
  %1 = rmo.mo.reduce.max<() -> D8, D9, D10, D11>(%arg0) [axis = 1] : (!mo.tensor<[D0, D1, D2, D3], f32>) -> !mo.tensor<[D8, D9, D10, D11], f32>
  mo.output<D8, D9, D10, D11> %1 : !mo.tensor<[D8, D9, D10, D11], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @reduce_max_dyn_dims<[[D0:[A-Z0-9]*]], [[D1:[A-Z0-9]*]], [[D2:[A-Z0-9]*]], [[D3:[A-Z0-9]*]] -> [[D4:[A-Z0-9]*]], [[D5:[A-Z0-9]*]], [[D6:[A-Z0-9]*]], [[D7:[A-Z0-9]*]]>([[ARG0:%.*]]: !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>) -> !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.reduce.max([[ARG0]]) : (!mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>) -> !mo.tensor<[[[D0]], 1, [[D2]], [[D3]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[D8:[A-Za-z0-9_]*]], [[D9:[A-Za-z0-9_]*]], [[D10:[A-Za-z0-9_]*]], [[D11:[A-Za-z0-9_]*]]>([[V0]]) [message = ""] : (!mo.tensor<[[[D0]], 1, [[D2]], [[D3]]], f32>) -> !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>
//  CHECK-NEXT:   mo.output<[[D8]], [[D9]], [[D10]], [[D11]]> [[V1]] : !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>

// -----

mo.graph @argmin_dyn_dims<D0, D1, D2, D3 -> D4, D5, D6, D7>(%arg0: !mo.tensor<[D0, D1, D2, D3], f32>) -> !mo.tensor<[D4, D5, D6, D7], si32> {
  %1 = rmo.mo.reduce.arg_min<() -> D8, D9, D10, D11>(%arg0) [axis = 1] : (!mo.tensor<[D0, D1, D2, D3], f32>) -> !mo.tensor<[D8, D9, D10, D11], si32>
  mo.output<D8, D9, D10, D11> %1 : !mo.tensor<[D8, D9, D10, D11], si32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @argmin_dyn_dims<[[D0:[A-Z0-9]*]], [[D1:[A-Z0-9]*]], [[D2:[A-Z0-9]*]], [[D3:[A-Z0-9]*]] -> [[D4:[A-Z0-9]*]], [[D5:[A-Z0-9]*]], [[D6:[A-Z0-9]*]], [[D7:[A-Z0-9]*]]>([[ARG0:%.*]]: !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>) -> !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], si32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.reduce.arg_min<() -> [[D8:[A-Za-z0-9_]*]], [[D9:[A-Za-z0-9_]*]], [[D10:[A-Za-z0-9_]*]], [[D11:[A-Za-z0-9_]*]]>([[ARG0]]) [axis = 1] : (!mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>) -> !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], si32>
//  CHECK-NEXT:   mo.output<[[D8]], [[D9]], [[D10]], [[D11]]> [[V0]] : !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], si32>

// -----

mo.graph @argmax_dyn_dims<D0, D1, D2, D3 -> D4, D5, D6, D7>(%arg0: !mo.tensor<[D0, D1, D2, D3], f32>) -> !mo.tensor<[D4, D5, D6, D7], si32> {
  %1 = rmo.mo.reduce.arg_max<() -> D8, D9, D10, D11>(%arg0) [axis = 1] : (!mo.tensor<[D0, D1, D2, D3], f32>) -> !mo.tensor<[D8, D9, D10, D11], si32>
  mo.output<D8, D9, D10, D11> %1 : !mo.tensor<[D8, D9, D10, D11], si32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @argmax_dyn_dims<[[D0:[A-Z0-9]*]], [[D1:[A-Z0-9]*]], [[D2:[A-Z0-9]*]], [[D3:[A-Z0-9]*]] -> [[D4:[A-Z0-9]*]], [[D5:[A-Z0-9]*]], [[D6:[A-Z0-9]*]], [[D7:[A-Z0-9]*]]>([[ARG0:%.*]]: !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>) -> !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], si32> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.reduce.arg_max<() -> [[D8:[A-Za-z0-9_]*]], [[D9:[A-Za-z0-9_]*]], [[D10:[A-Za-z0-9_]*]], [[D11:[A-Za-z0-9_]*]]>([[ARG0]]) [axis = 1] : (!mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>) -> !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], si32>
//  CHECK-NEXT:   mo.output<[[D8]], [[D9]], [[D10]], [[D11]]> [[V0]] : !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], si32>
