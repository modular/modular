// RUN: mo-opt --canonicalize %s | mo-opt | FileCheck %s

//===----------------------------------------------------------------------===//
// IndexToTensorOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: mo.graph @index_to_tensor(
// CHECK-SAME: %[[ARG0:.*]]: !kgen.scalar<si64>) -> !mo.tensor<[], si64> {
mo.graph @index_to_tensor(%arg0: !kgen.scalar<si64>) -> !mo.tensor<[], si64> {
  // CHECK: %[[RES:.*]] = mo.index.to_tensor(%[[ARG0]]) -> !mo.tensor<[], si64>
  %0 = mo.index.to_tensor(%arg0) -> !mo.tensor<[], si64>
  // CHECK: mo.output %[[RES]] : !mo.tensor<[], si64>
  mo.output %0: !mo.tensor<[], si64>
}

// CHECK-LABEL: mo.graph @index_to_tensor_fold() -> !mo.tensor<[], si64> {
mo.graph @index_to_tensor_fold() -> !mo.tensor<[], si64> {
  %0 = mosh.param.to_value = <42>
  // CHECK-NEXT: %[[RES:.*]] = mo.constant <#M.dense_array<42> : tensor<si64>> : !mo.tensor<[], si64>
  %1 = mo.index.to_tensor(%0) -> !mo.tensor<[], si64>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[], si64>
  mo.output %1: !mo.tensor<[], si64>
}

//===----------------------------------------------------------------------===//
// GraphOp and TensorType
//===----------------------------------------------------------------------===//

// CHECK-LABEL: mo.graph @nullary_parametric_graph<
// CHECK-SAME: () -> [[R1:.*]]>(%[[ARG:.*]]: !mo.tensor<[], f32>) -> !mo.tensor<[], f32> {
mo.graph @nullary_parametric_graph<() -> R1>(%arg: !mo.tensor<[], f32>) -> (!mo.tensor<[], f32>) {
  // CHECK-NEXT: mo.output<42> %[[ARG]] : !mo.tensor<[], f32>
  mo.output<42> %arg : !mo.tensor<[], f32>
}

// CHECK-LABEL: mo.graph @unary_void_parametric_graph<
// CHECK-SAME: [[N:.*]]>(%[[ARG:.*]]: !mo.tensor<[[[N]]], f32>) -> !mo.tensor<[[[N]]], f32> {
mo.graph @unary_void_parametric_graph<N>(%arg: !mo.tensor<[N], f32>) -> (!mo.tensor<[N], f32>) {
  // CHECK-NEXT: mo.output %[[ARG]] : !mo.tensor<[[[N]]], f32>
  mo.output %arg : !mo.tensor<[N], f32>
}

// CHECK-LABEL: mo.graph @binary_parametric_graph<
// CHECK-SAME: [[A:.*]], [[B:.*]] -> [[R1:.*]]>(%[[ARG:.*]]: !mo.tensor<[[[A]], [[B]]], f32>) -> !mo.tensor<[[[A]], [[B]]], f32> {
mo.graph @binary_parametric_graph<A, B-> R1>(%arg: !mo.tensor<[A, B], f32>) -> (!mo.tensor<[A, B], f32>) {
  // CHECK-NEXT: mo.output<42> %[[ARG]] : !mo.tensor<[[[A]], [[B]]], f32>
  mo.output<42> %arg : !mo.tensor<[A, B], f32>
}

// CHECK-LABEL: mo.graph @parametric_result<
// CHECK-SAME: () -> [[D2:.*]]>() {
mo.graph @parametric_result<() -> D2>() {
  // CHECK-NEXT: kgen.param.declare [[D1:.*]]: si64 = <42>
  kgen.param.declare D1: si64 = <42>
  // CHECK-NEXT: mo.output<[[D1]]>
  mo.output<D1>
}

// CHECK-LABEL: mo.graph @output_with_bindings<
// CHECK-SAME: [[D1:.*]] -> [[D2:.*]]>(%[[ARG:.*]]: !mo.tensor<[[[D1]]], f32>) -> !mo.tensor<[[[D2]]], f32> {
mo.graph @output_with_bindings<D1 -> D2>(%arg: !mo.tensor<[D1], f32>) -> (!mo.tensor<[D2], f32>) {
  // CHECK-NEXT: mo.output<D1> %[[ARG]] : !mo.tensor<[[[D1]]], f32>
  mo.output<D1> %arg : !mo.tensor<[D1], f32>
}

// CHECK-LABEL: mo.graph @empty_dims(
// CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[2, 0, 1], f32>) -> !mo.tensor<[2, 0, 1], f32> {
mo.graph @empty_dims(%arg0: !mo.tensor<[2, 0, 1], f32>) -> (!mo.tensor<[2, 0, 1], f32>) {
  mo.output %arg0 : !mo.tensor<[2, 0, 1], f32>
}

// CHECK-LABEL: mo.graph @scalar(
// CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[], f32>) -> !mo.tensor<[], f32> {
mo.graph @scalar(%arg0: !mo.tensor<[], f32>) -> (!mo.tensor<[], f32>) {
  mo.output %arg0 : !mo.tensor<[], f32>
}

// CHECK-LABEL: mo.graph @invalid_type(
// CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[2, 3], invalid>) -> !mo.tensor<[2, 3], invalid> {
mo.graph @invalid_type(%arg0: !mo.tensor<[2, 3], invalid>) -> (!mo.tensor<[2, 3], invalid>) {
  mo.output %arg0 : !mo.tensor<[2, 3], invalid>
}

// CHECK-LABEL: mo.graph @tensor_int_metadata(
// CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[], si32, {foo = 42 : i64}>) -> !mo.tensor<[], si32, {foo = 42 : i64}> {
mo.graph @tensor_int_metadata(%arg0: !mo.tensor<[], si32, {foo = 42}>) -> !mo.tensor<[], si32, {foo = 42}> {
  mo.output %arg0 : !mo.tensor<[], si32, {foo = 42}>
}

// CHECK-LABEL: mo.graph @tensor_str_metadata(
// CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[], si32, {bar = "str"}>) -> !mo.tensor<[], si32, {bar = "str"}> {
  mo.graph @tensor_str_metadata(%arg0: !mo.tensor<[], si32, {bar = "str"}>) -> !mo.tensor<[], si32, {bar = "str"}> {
  mo.output %arg0 : !mo.tensor<[], si32, {bar = "str"}>
}

// CHECK-LABEL: mo.graph @tensor_float_metadata(
// CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[], si32, {fp = 4.200000e+00 : f64}>) -> !mo.tensor<[], si32, {fp = 4.200000e+00 : f64}> {
  mo.graph @tensor_float_metadata(%arg0: !mo.tensor<[], si32, {fp = 4.2}>) -> !mo.tensor<[], si32, {fp = 4.2}> {
  mo.output %arg0 : !mo.tensor<[], si32, {fp = 4.2}>
}

//===----------------------------------------------------------------------===//
// IfOp and YieldOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: mo.graph @if_op(
// CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[2, 1], f32>, %[[ARG1:.*]]: !mo.tensor<[2, 1], f32>, %[[ARG2:.*]]: !mo.tensor<[], bool>) -> !mo.tensor<[2, 1], f32> {
mo.graph @if_op(%arg0: !mo.tensor<[2, 1], f32>, %arg1: !mo.tensor<[2, 1], f32>, %cond : !mo.tensor<[], bool>) -> (!mo.tensor<[2, 1], f32>) {
  // CHECK: %0 = mo.if(%[[ARG2]] : !mo.tensor<[], bool>) -> !mo.tensor<[2, 1], f32> {
  %0 = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.tensor<[2, 1], f32> {
    // CHECK: mo.yield %[[ARG0]] : !mo.tensor<[2, 1], f32>
    mo.yield %arg0 : !mo.tensor<[2, 1], f32>
  // CHECK: } else {
  } else {
    // CHECK: mo.yield %[[ARG1]] : !mo.tensor<[2, 1], f32>
    mo.yield %arg1 : !mo.tensor<[2, 1], f32>
  }
  // CHECK: mo.output %0 : !mo.tensor<[2, 1], f32>
  mo.output %0 : !mo.tensor<[2, 1], f32>
}

// CHECK-LABEL: mo.graph @empty_if_op(
// CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[2, 1], f32>, %[[ARG2:.*]]: !mo.tensor<[], bool>) -> !mo.tensor<[2, 1], f32> {
mo.graph @empty_if_op(%arg0: !mo.tensor<[2, 1], f32>, %cond : !mo.tensor<[], bool>) -> (!mo.tensor<[2, 1], f32>) {
  // CHECK: mo.output %[[ARG0]] : !mo.tensor<[2, 1], f32>
  mo.if(%cond : !mo.tensor<[], bool>) {
    mo.yield
  } else {
    mo.yield
  }
  mo.output %arg0 : !mo.tensor<[2, 1], f32>
}

// CHECK-LABEL: mo.graph @constant_fold_if_tensor(
// CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[2], f32>, %[[ARG1:.*]]: !mo.tensor<[2], f32>) -> (!mo.tensor<[2], f32>, !mo.tensor<[2], f32>) {
mo.graph @constant_fold_if_tensor(%arg0: !mo.tensor<[2], f32>, %arg1: !mo.tensor<[2], f32>) -> (!mo.tensor<[2], f32>, !mo.tensor<[2], f32>) {
  %true = mo.constant <#M.dense_array<1> : tensor<i1>> : !mo.tensor<[], bool>
  %false = mo.constant <#M.dense_array<0> : tensor<i1>> : !mo.tensor<[], bool>

  %0 = mo.if(%true : !mo.tensor<[], bool>) -> !mo.tensor<[2], f32> {
    mo.yield %arg0 : !mo.tensor<[2], f32>
  } else {
    mo.yield %arg1 : !mo.tensor<[2], f32>
  }

  %1 = mo.if(%false : !mo.tensor<[], bool>) -> !mo.tensor<[2], f32> {
    mo.yield %arg0 : !mo.tensor<[2], f32>
  } else {
    mo.yield %arg1 : !mo.tensor<[2], f32>
  }

  // CHECK-NEXT: mo.output %[[ARG0]], %[[ARG1]]
  mo.output %0, %1: !mo.tensor<[2], f32>, !mo.tensor<[2], f32>
}

// CHECK-LABEL: mo.graph @constant_fold_if_scalar(
// CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[2], f32>, %[[ARG1:.*]]: !mo.tensor<[2], f32>) -> (!mo.tensor<[2], f32>, !mo.tensor<[2], f32>) {
mo.graph @constant_fold_if_scalar(%arg0: !mo.tensor<[2], f32>, %arg1: !mo.tensor<[2], f32>) -> (!mo.tensor<[2], f32>, !mo.tensor<[2], f32>) {
  %true = mo.constant.scalar <true> : !mo.scalar<bool>
  %false = mo.constant.scalar <false> : !mo.scalar<bool>

  %0 = mo.if(%true : !mo.scalar<bool>) -> !mo.tensor<[2], f32> {
    mo.yield %arg0 : !mo.tensor<[2], f32>
  } else {
    mo.yield %arg1 : !mo.tensor<[2], f32>
  }

  %1 = mo.if(%false : !mo.scalar<bool>) -> !mo.tensor<[2], f32> {
    mo.yield %arg0 : !mo.tensor<[2], f32>
  } else {
    mo.yield %arg1 : !mo.tensor<[2], f32>
  }

  // CHECK-NEXT: mo.output %[[ARG0]], %[[ARG1]]
  mo.output %0, %1: !mo.tensor<[2], f32>, !mo.tensor<[2], f32>
}

// CHECK-LABEL: mo.graph @remove_unused_results(
// CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[2], f32>, %[[ARG1:.*]]: !mo.tensor<[2], f32>, %[[COND:.*]]: !mo.tensor<[], bool>) -> !mo.tensor<[2], f32> {
mo.graph @remove_unused_results(%arg0: !mo.tensor<[2], f32>, %arg1: !mo.tensor<[2], f32>, %cond: !mo.tensor<[], bool>) -> (!mo.tensor<[2], f32>) {

  // CHECK: %[[V0:.*]] = mo.if(%[[COND]] : !mo.tensor<[], bool>) -> !mo.tensor<[2], f32> {
  %0:2 = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.tensor<[2], f32>, !mo.tensor<[2], f32> {
    // CHECK: mo.yield %[[ARG1]]
    mo.yield %arg0, %arg1 : !mo.tensor<[2], f32>, !mo.tensor<[2], f32>
  } else {
    // CHECK: mo.yield %[[ARG0]]
    mo.yield %arg1, %arg0 : !mo.tensor<[2], f32>, !mo.tensor<[2], f32>
  }

  // CHECK: mo.output %[[V0]]
  mo.output %0#1 : !mo.tensor<[2], f32>
}

// CHECK-LABEL: mo.graph @condition_propagation(
// CHECK-SAME: %[[COND:.*]]: !mo.tensor<[], bool>) -> !mo.tensor<[], bool> {
mo.graph @condition_propagation(%cond: !mo.tensor<[], bool>) -> (!mo.tensor<[], bool>) {

  // CHECK-DAG: %[[FALSE:.*]] = mo.constant <#M.dense_array<0> : tensor<1xi8>> : !mo.tensor<[], bool>
  // CHECK-DAG: %[[TRUE:.*]] = mo.constant <#M.dense_array<1> : tensor<1xi8>> : !mo.tensor<[], bool>

  // CHECK: %[[V0:.*]] = mo.if(%[[COND]] : !mo.tensor<[], bool>) -> !mo.tensor<[], bool> {
  %0 = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.tensor<[], bool> {
    // CHECK: mo.yield %[[TRUE]]
    mo.yield %cond : !mo.tensor<[], bool>
  } else {
    // CHECK: mo.yield %[[FALSE]]
    mo.yield %cond : !mo.tensor<[], bool>
  }

  // CHECK: mo.output %[[V0]]
  mo.output %0 : !mo.tensor<[], bool>
}

// CHECK-LABEL: mo.graph @condition_propagation_scalars(
// CHECK-SAME: %[[COND:.*]]: !mo.scalar<bool>) -> !mo.scalar<bool> {
mo.graph @condition_propagation_scalars(%cond: !mo.scalar<bool>) -> (!mo.scalar<bool>) {

  // CHECK-DAG: %[[FALSE:.*]] = mo.constant.scalar <false> : !mo.scalar<bool>
  // CHECK-DAG: %[[TRUE:.*]] = mo.constant.scalar <true> : !mo.scalar<bool>

  // CHECK: %[[V0:.*]] = mo.if(%[[COND]] : !mo.scalar<bool>) -> !mo.scalar<bool> {
  %0 = mo.if(%cond : !mo.scalar<bool>) -> !mo.scalar<bool> {
    // CHECK: mo.yield %[[TRUE]]
    mo.yield %cond : !mo.scalar<bool>
  } else {
    // CHECK: mo.yield %[[FALSE]]
    mo.yield %cond : !mo.scalar<bool>
  }

  // CHECK: mo.output %[[V0]]
  mo.output %0 : !mo.scalar<bool>
}

//===----------------------------------------------------------------------===//
// WhileOp, WhileConditionOp, and YieldOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: mo.graph @while_op(
// CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[2], f32>, %[[COND:.*]]: !mo.tensor<[], bool>) -> !mo.tensor<[2], f32> {
mo.graph @while_op(%arg0: !mo.tensor<[2], f32>, %cond : !mo.tensor<[], bool>) -> (!mo.tensor<[2], f32>) {
  // CHECK: %[[RES:.*]]:2 = mo.while (%[[ARG0]] as %[[ARGI0:.*]]: !mo.tensor<[2], f32>, %[[COND]] as %[[CONDI:.*]]: !mo.tensor<[], bool>) {
  %0:2 = mo.while (%arg0 as %argi0: !mo.tensor<[2], f32>, %cond as %condi: !mo.tensor<[], bool>) {
    // CHECK: %[[NEG:.*]] = mo.not(%[[CONDI]]) : !mo.tensor<[], bool>
    %neg = mo.not(%condi) : !mo.tensor<[], bool>
    // CHECK: mo.while.condition(%[[CONDI]]) %[[ARGI0]], %[[NEG]] : !mo.tensor<[2], f32>, !mo.tensor<[], bool>
    mo.while.condition(%condi) %argi0, %neg : !mo.tensor<[2], f32>, !mo.tensor<[], bool>
  // CHECK: } do {
  } do {
    // CHECK: %[[NEW_ARG0:.*]] = mo.add(%[[ARGI0]], %[[ARGI0]]) : !mo.tensor<[2], f32>
    %new_arg0 = mo.add (%argi0, %argi0) : !mo.tensor<[2], f32>
    // CHECK: mo.yield %[[NEW_ARG0]], %[[CONDI]] : !mo.tensor<[2], f32>, !mo.tensor<[], bool>
    mo.yield %new_arg0, %condi : !mo.tensor<[2], f32>, !mo.tensor<[], bool>
  }
  // CHECK: mo.output %[[RES]]#0 : !mo.tensor<[2], f32>
  mo.output %0#0 : !mo.tensor<[2], f32>
}

// CHECK-LABEL: mo.graph @while_op_remove_invariant_values
// CHECK-SAME: (%[[ARG0:.+]]: !mo.tensor<[2], f32>, %[[COND:.*]]: !mo.tensor<[], bool>) -> !mo.tensor<[], bool> {
mo.graph @while_op_remove_invariant_values(%arg0: !mo.tensor<[2], f32>, %cond : !mo.tensor<[], bool>) -> (!mo.tensor<[], bool>) {
  // CHECK-NOT: mo.while
  %0:2 = mo.while (%arg0 as %argi0: !mo.tensor<[2], f32>, %cond as %condi: !mo.tensor<[], bool>) {
    mo.while.condition(%condi) %argi0, %condi : !mo.tensor<[2], f32>, !mo.tensor<[], bool>
  } do {
    mo.yield %argi0, %condi : !mo.tensor<[2], f32>, !mo.tensor<[], bool>
  }
  // CHECK: mo.output %[[ARG1]] : !mo.tensor<[], bool>
  mo.output %0#1 : !mo.tensor<[], bool>
}

// CHECK-LABEL: mo.graph @while_op_remove_duplicate_operands
// CHECK-SAME: (%[[ARG0:.+]]: !mo.tensor<[2], f32>, %[[COND:.*]]: !mo.tensor<[], bool>) -> (!mo.tensor<[2], f32>, !mo.tensor<[2], f32>) {
mo.graph @while_op_remove_duplicate_operands(%arg0: !mo.tensor<[2], f32>, %cond : !mo.tensor<[], bool>) -> (!mo.tensor<[2], f32>, !mo.tensor<[2], f32>) {
  // CHECK: %[[RES:.*]] = mo.while (%[[ARG0]] as %[[ARGI:.*]]: !mo.tensor<[2], f32>) {
  %0:2 = mo.while (%arg0 as %argi0: !mo.tensor<[2], f32>, %arg0 as %argi1: !mo.tensor<[2], f32>) {
    // CHECK: mo.while.condition(%[[COND]]) %[[ARGI]] : !mo.tensor<[2], f32>
    mo.while.condition(%cond) %argi0, %argi1 : !mo.tensor<[2], f32>, !mo.tensor<[2], f32>
  // CHECK: } do {
  } do {
    // CHECK: %[[NEW_VAL:.*]] = mo.add(%[[ARGI]], %[[ARGI]]) : !mo.tensor<[2], f32>
    %new_val = mo.add (%argi0, %argi1) : !mo.tensor<[2], f32>
    // CHECK: mo.yield %[[NEW_VAL]] : !mo.tensor<[2], f32>
    mo.yield %new_val, %new_val : !mo.tensor<[2], f32>, !mo.tensor<[2], f32>
  }
  // CHECK: mo.output %[[RES]], %[[RES]] : !mo.tensor<[2], f32>, !mo.tensor<[2], f32>
  mo.output %0#0, %0#1 : !mo.tensor<[2], f32>, !mo.tensor<[2], f32>
}

// CHECK-LABEL: mo.graph @while_op_remove_duplicate_operands_no_change
// CHECK-SAME: (%[[ARG0:.+]]: !mo.tensor<[2], f32>, %[[COND:.*]]: !mo.tensor<[], bool>) -> (!mo.tensor<[2], f32>, !mo.tensor<[2], f32>) {
mo.graph @while_op_remove_duplicate_operands_no_change(%arg0: !mo.tensor<[2], f32>, %cond : !mo.tensor<[], bool>) -> (!mo.tensor<[2], f32>, !mo.tensor<[2], f32>) {
  // CHECK: %[[RES:.*]]:2 = mo.while (%[[ARG0]] as %[[ARGI0:.*]]: !mo.tensor<[2], f32>, %[[ARG0]] as %[[ARGI1:.*]]: !mo.tensor<[2], f32>) {
  %0:2 = mo.while (%arg0 as %argi0: !mo.tensor<[2], f32>, %arg0 as %argi1: !mo.tensor<[2], f32>) {
    // CHECK: mo.while.condition(%[[COND]]) %[[ARGI0]], %[[ARGI1]] : !mo.tensor<[2], f32>, !mo.tensor<[2], f32>
    mo.while.condition(%cond) %argi0, %argi1 : !mo.tensor<[2], f32>, !mo.tensor<[2], f32>
  // CHECK: } do {
  } do {
    // CHECK: %[[NEW_VAL1:.*]] = mo.add(%[[ARGI0]], %[[ARGI1]]) : !mo.tensor<[2], f32>
    %new_val1 = mo.add (%argi0, %argi1) : !mo.tensor<[2], f32>

    // CHECK: %[[NEW_VAL2:.*]] = mo.mul(%[[ARGI0]], %[[ARGI1]]) : !mo.tensor<[2], f32>
    %new_val2 = mo.mul (%argi0, %argi1) : !mo.tensor<[2], f32>

    // CHECK: mo.yield %[[NEW_VAL1]], %[[NEW_VAL2]] : !mo.tensor<[2], f32>
    mo.yield %new_val1, %new_val2 : !mo.tensor<[2], f32>, !mo.tensor<[2], f32>
  }
  // CHECK: mo.output %[[RES]]#0, %[[RES]]#1 : !mo.tensor<[2], f32>, !mo.tensor<[2], f32>
  mo.output %0#0, %0#1 : !mo.tensor<[2], f32>, !mo.tensor<[2], f32>
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: mo.graph @constant() -> !mo.tensor<[2, 3], f32>
mo.graph @constant() -> !mo.tensor<[2, 3], f32> {
  // CHECK: %[[VAL_0:.*]] = mo.constant <#M.dense_array<1.{{0+}}e+00, 2.{{0+}}e+00, 3.{{0+}}e+00, 4.{{0+}}e+00, 5.{{0+}}e+00, 6.{{0+}}e+00> : tensor<2x3xf32>> : !mo.tensor<[2, 3], f32>
  %0 = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0> : tensor<2x3xf32>> : !mo.tensor<[2, 3], f32>
  // CHECK: mo.output %[[VAL_0]] : !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// CHECK-LABEL: mo.graph @constant_cse() -> (!mo.tensor<[2], ui32>, !mo.tensor<[2], ui32>)
mo.graph @constant_cse() -> (!mo.tensor<[2], ui32>, !mo.tensor<[2], ui32>) {
  // CHECK: %[[VAL0:.*]] = mo.constant <#M.dense_array<1, 2> : tensor<2xui32>> : !mo.tensor<[2], ui32>
  %0 = mo.constant <#M.dense_array<1, 2> : tensor<2xui32>> : !mo.tensor<[2], ui32>
  %1 = mo.constant <#M.dense_array<1, 2> : tensor<2xui32>> : !mo.tensor<[2], ui32>
  // CHECK: mo.output %[[VAL0]], %[[VAL0]] : !mo.tensor<[2], ui32>, !mo.tensor<[2], ui32>
  mo.output %0, %1 : !mo.tensor<[2], ui32>, !mo.tensor<[2], ui32>
}

// CHECK-LABEL: mo.graph @two_constants() -> (!mo.tensor<[2, 3], f32>, !mo.tensor<[4], si8>)
mo.graph @two_constants() -> (!mo.tensor<[2, 3], f32>, !mo.tensor<[4], si8>) {
  // CHECK-DAG: %[[VAL_0:.*]] = mo.constant <#M.dense_array<0.{{0+}}e+00, 1.{{0+}}e+00, 2.{{0+}}e+00, 3.{{0+}}e+00, 4.{{0+}}e+00, 5.{{0+}}e+00> : tensor<2x3xf32>> : !mo.tensor<[2, 3], f32>
  // CHECK-DAG: %[[VAL_1:.*]] = mo.constant <dense_resource<some_buffer> : tensor<4xsi8>> : !mo.tensor<[4], si8>
  %0 = mo.constant <#M.dense_array<0.0, 1.0, 2.0, 3.0, 4.0, 5.0> : tensor<2x3xf32>> : !mo.tensor<[2, 3], f32>
  %1 = mo.constant <dense_resource<some_buffer> : tensor<4xsi8>> : !mo.tensor<[4], si8>
  // CHECK: mo.output %[[VAL_0]], %[[VAL_1]] : !mo.tensor<[2, 3], f32>, !mo.tensor<[4], si8>
  mo.output %0, %1 : !mo.tensor<[2, 3], f32>, !mo.tensor<[4], si8>
}

//===----------------------------------------------------------------------===//
// ConstantExternalOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: mo.graph @external_constant_control_flow
// CHECK:       (%[[ARG0:.*]]: !mo.tensor<[], bool>) -> !mo.tensor<[5, 10], si64>
mo.graph @external_constant_control_flow(%arg0: !mo.tensor<[], bool>) -> !mo.tensor<[5, 10], si64> {
  // CHECK-NEXT: %[[C1:.*]] = mo.constant.external [name = "random_weight", align = 8, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[5, 10], si64>
  // CHECK-NEXT: %[[C0:.*]] = mo.constant.external [name = "random_weight", align = 8, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[5, 10], si64>

  %0 = mo.if(%arg0 : !mo.tensor<[], bool>) -> !mo.tensor<[5, 10], si64> {
    %4 = mo.constant.external [name = "random_weight", align = 8, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[5, 10], si64>

    // CHECK: mo.yield %[[C0]]
    mo.yield %4 : !mo.tensor<[5, 10], si64>
  } else {
    %4 = mo.constant.external [name = "random_weight", align = 8, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[5, 10], si64>

    // CHECK: mo.yield %[[C1]]
    mo.yield %4 : !mo.tensor<[5, 10], si64>
  }
  mo.output %0 : !mo.tensor<[5, 10], si64>
}


//===----------------------------------------------------------------------===//
// TransferOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: mo.graph @transfer_to_explicit
mo.graph @transfer_to_explicit(%arg0 : !mo.tensor<[3, 2], f32, gpu:3>) -> !mo.tensor<[3, 2], f32> {
  // CHECK: %{{.*}} = mo.chain.create
  // CHECK: %[[T:.*]], %[[CH:.*]] = mo.transfer[%{{.*}}] %{{.*}} : !mo.tensor<[3, 2], f32, gpu:3> to <"cpu", 0>
  %ch0 = mo.chain.create()
  %0, %ch1 = mo.transfer[%ch0] %arg0 : !mo.tensor<[3, 2], f32, gpu:3> to <"cpu", 0>
  mo.output %0 : !mo.tensor<[3, 2], f32>
}

// CHECK-LABEL: mo.graph @transfer_fold_explicit
mo.graph @transfer_fold_explicit(%arg0 : !mo.tensor<[7], f32, gpu:3>) -> !mo.tensor<[7], f32, gpu:3> {
  // CHECK-NEXT: mo.output
  %ch0 = mo.chain.create()
  %t0, %ch1 = mo.transfer[%ch0] %arg0 : !mo.tensor<[7], f32, gpu:3> to <"cpu", 0>
  %t1, %ch2 = mo.transfer[%ch1] %t0 : !mo.tensor<[7], f32> to <"gpu", 3>
  mo.output %t1 : !mo.tensor<[7], f32, gpu:3>
}

// Checks that double transfers are only folded if the first transfer has a
// single use.
// CHECK-LABEL: mo.graph @transfer_fold_multi_uses
// CHECK-SAME: ([[ARG0:%.*]]: !mo.tensor{{.*}} ->
mo.graph @transfer_fold_multi_uses(%arg0 : !mo.tensor<[7], f32, gpu:3>) -> (!mo.tensor<[7], f32>, !mo.tensor<[7], f32, gpu:3>) {
  // CHECK-NEXT: mo.chain.create
  // CHECK-NEXT: mo.transfer{{.*}}[[ARG0]]
  // CHECK-NEXT: mo.output
  %ch0 = mo.chain.create()
  %t0, %ch1 = mo.transfer[%ch0] %arg0 : !mo.tensor<[7], f32, gpu:3> to <"cpu", 0>
  %t1, %ch2 = mo.transfer[%ch1] %t0 : !mo.tensor<[7], f32> to <"gpu", 3>
  mo.output %t0, %t1 : !mo.tensor<[7], f32>, !mo.tensor<[7], f32, gpu:3>
}

// CHECK-LABEL: mo.graph @transfer_elide_copy
mo.graph @transfer_elide_copy(%arg0 : !mo.tensor<[4, 5], f32, gpu:3>) -> !mo.tensor<[4, 5], f32, gpu:3> {
  // CHECK-NEXT: mo.output
  %ch0 = mo.chain.create()
  %t0, %ch1 = mo.transfer[%ch0] %arg0 : !mo.tensor<[4, 5], f32, gpu:3> to <"gpu", 3> [alwaysElideSameDeviceCopy = true]
  mo.output %t0 : !mo.tensor<[4, 5], f32, gpu:3>
}

// CHECK-LABEL: mo.graph @transfer_no_elide_with_flag
mo.graph @transfer_no_elide_with_flag(%arg0 : !mo.tensor<[4, 5], f32, gpu:3>) -> !mo.tensor<[4, 5], f32> {
  // CHECK: %[[T:.*]], %[[C:.*]] = mo.transfer[%[[IN:.*]]] %{{.*}} : !mo.tensor<[4, 5], f32, gpu:3> to <"cpu", 0>
  %ch0 = mo.chain.create()
  %t0, %ch1 = mo.transfer[%ch0] %arg0 : !mo.tensor<[4, 5], f32, gpu:3> to <"cpu", 0> [alwaysElideSameDeviceCopy = true]
  // CHECK-NEXT: mo.output %[[T]]
  mo.output %t0 : !mo.tensor<[4, 5], f32>
}

// CHECK-LABEL: mo.graph @transfer_no_elide_copy
mo.graph @transfer_no_elide_copy(%arg0 : !mo.tensor<[4, 5], f32, gpu:3>) -> !mo.tensor<[4, 5], f32, gpu:3> {
  // CHECK: %[[T:.*]], %[[C:.*]] = mo.transfer[%[[IN:.*]]] %{{.*}} : !mo.tensor<[4, 5], f32, gpu:3> to <"gpu", 3>
  %ch0 = mo.chain.create()
  %t0, %ch1 = mo.transfer[%ch0] %arg0 : !mo.tensor<[4, 5], f32, gpu:3> to <"gpu", 3> [alwaysElideSameDeviceCopy = false]
  mo.output %t0 : !mo.tensor<[4, 5], f32, gpu:3>
}

// CHECK-LABEL: mo.graph @transfer_cast_rotate
// CHECK-SAME: (%[[ARG0:.*]]: !mo.tensor<[4096], f32>) ->
// CHECK: %[[CAST:.*]] = mo.cast(%[[ARG0]]) : (!mo.tensor<[4096], f32>) -> !mo.tensor<[4096], bf16>
// CHECK-NEXT: %[[GPU:.*]], %[[CH:.*]] = mo.transfer[%[[IN:.*]]] %[[CAST]] : !mo.tensor<[4096], bf16> to <"gpu", 0>
// CHECK-NEXT: mo.output %[[GPU]]
mo.graph @transfer_cast_rotate(%arg0 : !mo.tensor<[4096], f32>) -> !mo.tensor<[4096], bf16, gpu:0> {
  %ch0 = mo.chain.create()
  %0 = mo.cast(%arg0) : (!mo.tensor<[4096], f32>) -> !mo.tensor<[4096], bf16>
  %1, %ch1 = mo.transfer[%ch0] %0 : !mo.tensor<[4096], bf16> to <"gpu", 0>
  mo.output %1 : !mo.tensor<[4096], bf16, gpu:0>
}

// CHECK-LABEL: mo.graph @transfer_cast_no_rotate1
// CHECK-SAME: (%[[ARG0:.*]]: !mo.tensor<[4096], bf16>) ->
// CHECK-NEXT: %[[CH:.*]] = mo.chain.create
// CHECK-NEXT: %[[CPU:.*]], %[[OCH:.*]] = mo.transfer[%[[CH]]]
// CHECK-SAME: %[[ARG0]] : !mo.tensor<[4096], bf16> to <"gpu", 0>
// CHECK-NEXT: %[[OUT:.*]] = mo.cast(%[[CPU]])
// CHECK-NEXT: mo.output %[[OUT]]
mo.graph @transfer_cast_no_rotate1(%arg0 : !mo.tensor<[4096], bf16>) -> !mo.tensor<[4096], f32, gpu:0> {
  %ch0 = mo.chain.create()
  %0, %c1 = mo.transfer[%ch0] %arg0 : !mo.tensor<[4096], bf16> to <"gpu", 0>
  %1 = mo.cast(%0) : (!mo.tensor<[4096], bf16, gpu:0>) -> !mo.tensor<[4096], f32, gpu:0>
  mo.output %1 : !mo.tensor<[4096], f32, gpu:0>
}

// CHECK-LABEL: mo.graph @transfer_cast_no_rotate_2
// CHECK-SAME: (%[[ARG0:.*]]: !mo.tensor<[4096], f32, gpu:0>) ->
// CHECK-NEXT: %[[CH:.*]] = mo.chain.create
// CHECK-NEXT: %[[CPU:.*]], %[[OCH:.*]] = mo.transfer[%[[CH]]]
// CHECK-SAME: %[[ARG0]] : !mo.tensor<[4096], f32, gpu:0> to <"cpu", 0>
// CHECK-NEXT: %[[OUT:.*]] = mo.cast(%[[CPU]])
// CHECK-NEXT: mo.output %[[OUT]]
mo.graph @transfer_cast_no_rotate_2(%arg0 : !mo.tensor<[4096], f32, gpu:0>) -> !mo.tensor<[4096], bf16> {
  %ch0 = mo.chain.create()
  %0, %c1 = mo.transfer[%ch0] %arg0 : !mo.tensor<[4096], f32, gpu:0> to <"cpu", 0>
  %1 = mo.cast(%0) : (!mo.tensor<[4096], f32>) -> !mo.tensor<[4096], bf16>
  mo.output %1 : !mo.tensor<[4096], bf16>
}

{-#
  dialect_resources: {
    builtin: {
      some_buffer: "0x4000000087654321"
    }
  }
#-}

//===----------------------------------------------------------------------===//
// StableCreateOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: mo.graph @stable_create
mo.graph @stable_create(%arg0: !mo.tensor<[3, 2], f32, gpu:1>) -> !mo.tensor<[3, 2], f32, gpu:1, {stable}> {
  // CHECK: %[[S:.*]] = mo.stable.create %{{.*}} : !mo.tensor<[3, 2], f32, gpu:1, {stable}>
  // CHECK: mo.output %[[S]] : !mo.tensor<[3, 2], f32, gpu:1, {stable}>
  %0 = mo.stable.create %arg0 : !mo.tensor<[3, 2], f32, gpu:1, {stable}>
  mo.output %0 : !mo.tensor<[3, 2], f32, gpu:1, {stable}>
}
