// RUN: mo-opt %s --mo-propagate-shapes --canonicalize | FileCheck %s

// CHECK-LABEL: mo.graph @graph_with_param_decls_in_ops
// CHECK-SAME: <[[D0:.*]] -> [[D2:.*]]>(
mo.graph @graph_with_param_decls_in_ops<D0 -> D1, D2>(
  // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[[[D0]], 20], f32>,
  %arg0: !mo.tensor<[D0, 20], f32>,
  // CHECK-SAME: %[[SHAPE:.*]]: !mo.tensor<[2], si32>
  %shape: !mo.tensor<[2], si32>
// CHECK-SAME: ) -> !mo.tensor<[[[D0]], [[D2]]], f32> {
) -> !mo.tensor<[D1, D2], f32> {
  kgen.param.declare D3: scalar<si64> = <D0>
  kgen.param.declare D4: scalar<si64> = <20>


  %0 = mo.rebind(%arg0) : !mo.tensor<[D0, 20], f32> -> !mo.tensor<[D3, D4], f32>
  // CHECK-NEXT: %[[VAL1:.*]] = mo.reshape<() -> [[D6:.*]], [[D7:.*]]>(%[[ARG0]], %[[SHAPE]]) : (
  // CHECK-SAME:   !mo.tensor<[[[D0]], 20], f32>, !mo.tensor<[2], si32>
  // CHECK-SAME: ) -> !mo.tensor<[[[D6]], [[D7]]], f32>
  %1 = mo.reshape<() -> D6, D7>(%0, %shape) : (
    !mo.tensor<[D3, D4], f32>, !mo.tensor<[2], si32>
  ) -> !mo.tensor<[D6, D7], f32>
  kgen.param.declare D8: scalar<si64> = <D3>
  kgen.param.declare D9: scalar<si64> = <D7>
  // CHECK-NEXT: %[[VAL2:.*]] = mo.rebind(%[[VAL1]]) :
  // CHECK-SAME: !mo.tensor<[[[D6]], [[D7]]], f32> -> !mo.tensor<[[[D0]], [[D7]]], f32>
  %2 = mo.rebind(%1) : !mo.tensor<[D6, D7], f32> -> !mo.tensor<[D8, D9], f32>
  // CHECK-NEXT: mo.output<[[D7]]> %[[VAL2]] : !mo.tensor<[[[D0]], [[D7]]], f32>
  mo.output<D8, D9> %2 : !mo.tensor<[D8, D9], f32>
}

// CHECK-LABEL: mo.graph @nested_div_mul_simplification
// CHECK-SAME: <[[D0:.*]], [[D1:.*]]>(
mo.graph @nested_div_mul_simplification<D0, D1 -> D2>(
  // CHECK-SAME: %[[ARG:.*]]: !mo.tensor<[[[D0]], [[D1]], 20], f32>
  %arg: !mo.tensor<[D0, D1, 20], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[D0]], [[D1]], 4, 5], f32> {
) -> !mo.tensor<[D0, D2, 4, 5], f32> {
  kgen.param.declare NUM: scalar<si64> = <mul_no_wrap(D0, D1, 20)>
  kgen.param.declare DENOM: scalar<si64> = <mul_no_wrap(D0, 4, 5)>
  kgen.param.declare DIV: scalar<si64> = <div(NUM, DENOM)>
  // CHECK-NEXT: %[[RES:.*]] = mo.static.reshape(%[[ARG]])
  // CHECK-SAME: : !mo.tensor<[[[D0]], [[D1]], 20], f32> -> !mo.tensor<[[[D0]], [[D1]], 4, 5], f32>
  %res = mo.static.reshape(%arg) : !mo.tensor<[D0, D1, 20], f32> -> !mo.tensor<[D0, DIV, 4, 5], f32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[[[D0]], [[D1]], 4, 5], f32>
  mo.output<DIV> %res : !mo.tensor<[D0, DIV, 4, 5], f32>
}


// CHECK-LABEL: mo.graph @param_expr_dce
// CHECK-SAME:   <[[D0:.*]]>(%[[INPUT:.*]]: !mo.tensor<[[[D0]]], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[D0]]], f32> {
mo.graph @param_expr_dce<D0 -> D1>(
  %input: !mo.tensor<[D0], f32>
) -> !mo.tensor<[D1], f32> {

  // These are trivial and DCEed away
  kgen.param.declare D2: scalar<si64> = <add(D0, 4)>
  kgen.param.declare D3: scalar<si64> = <sub(D2, 4)>
  %res = mo.static.reshape(%input) : !mo.tensor<[D0], f32> -> !mo.tensor<[D3], f32>

  // CHECK-NEXT: mo.output %[[INPUT]] : !mo.tensor<[[[D0]]], f32>
  mo.output<D3> %res : !mo.tensor<[D3], f32>
}

// CHECK-LABEL: mo.graph @fold_nested_max
// CHECK-SAME: <[[D0:.*]], [[BS:.*]] -> [[OUT:.*]]>(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[[[D0]]], f32>,
// CHECK-SAME:   %[[DUMMY:.*]]: !mo.tensor<[[[BS]]], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[OUT]], 768], f32> {
mo.graph @fold_nested_max<D0, bs -> out>(
  %input: !mo.tensor<[D0], f32>,
  %dummy: !mo.tensor<[bs], f32> // this is used solely to help define `bs`
) -> !mo.tensor<[out, 768], f32> {

  kgen.param.declare D1: scalar<si64> = <max(bs, 1)>
  kgen.param.declare D2: scalar<si64> = <max(D1, bs)>

  // CHECK-NEXT: %[[INPUT_RS:.*]] = mo.static.reshape(%[[INPUT]]) : !mo.tensor<[[[D0]]], f32> -> !mo.tensor<[max([[BS]], 1), 768], f32>
  %input_rs = mo.static.reshape(%input) : !mo.tensor<[D0], f32> -> !mo.tensor<[D1, 768], f32>

  // Simplified and removed.
  %res = mo.static.broadcast_to(%input_rs) : !mo.tensor<[D1, 768], f32> -> !mo.tensor<[D2, 768], f32>

  // CHECK-NEXT: mo.output<max([[BS]], 1)> %[[INPUT_RS]] : !mo.tensor<[max([[BS]], 1), 768], f32>
  mo.output<D2> %res : !mo.tensor<[D2, 768], f32>
}

// CHECK-LABEL: mo.graph @non_mo_tensor_input
mo.graph @non_mo_tensor_input<D0 -> D1>(
  // CHECK-SAME: <[[D0:.*]]>(%[[INPUT:.*]]: !mo.tensor<[[[D0]]], f32>, %[[SHAPE:.*]]: !mosh.ape
  %input: !mo.tensor<[D0], f32>, %shape: !mosh.ape
// CHECK-SAME: ) -> (!mo.tensor<[[[D0]]], f32>, !mosh.ape) {
) -> (!mo.tensor<[D1], f32>, !mosh.ape) {

  // CHECK-NEXT: mo.output %[[INPUT]], %[[SHAPE]] : !mo.tensor<[[[D0]]], f32>, !mosh.ape
  mo.output<D0> %input, %shape : !mo.tensor<[D0], f32>, !mosh.ape
}

// CHECK-LABEL: mo.graph @out_of_order_instructions
// CHECK-SAME:  <[[D0:.*]] -> [[OUT:.*]]>(
// CHECK-SAME:   %[[SCALAR:.*]]: !mo.tensor<[], f32>,
// CHECK-SAME:   %[[DUMMY:.*]]: !mo.tensor<[[[D0]]], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[OUT]]], f32> {
mo.graph @out_of_order_instructions<D0 -> out>(
  %scalar: !mo.tensor<[], f32>,
  %dummy: !mo.tensor<[D0], f32> // this is used solely to help define `D0`
) -> !mo.tensor<[out], f32> {

  kgen.param.declare D3: scalar<si64> = <div(D2, D0)>
  kgen.param.declare D2: scalar<si64> = <mul(D1, D1)>

  // CHECK-NEXT: %[[RES:.*]] = mo.static.broadcast_to(%[[SCALAR]]) : !mo.tensor<[], f32> -> !mo.tensor<[div(add(mul([[D0]], [[D0]]), mul([[D0]], 2), 1), [[D0]])], f32>
  %res = mo.static.broadcast_to(%scalar) : !mo.tensor<[], f32> -> !mo.tensor<[D3], f32>

  kgen.param.declare D1: scalar<si64> = <add(D0, 1)>

  // CHECK-NEXT: mo.output<div(add(mul([[D0]], [[D0]]), mul([[D0]], 2), 1), [[D0]])> %[[RES]] : !mo.tensor<[div(add(mul([[D0]], [[D0]]), mul([[D0]], 2), 1), [[D0]])], f32>
  mo.output<D3> %res : !mo.tensor<[D3], f32>
}

// CHECK-LABEL: mo.graph @nested_mul
mo.graph @nested_mul<D0>() {

  // These KGEN expressions are all folded away

  // NOTE that this gets canonicalized into `mul(D0, 2)`
  kgen.param.declare P0: scalar<si64> = <add(D0, D0)>
  kgen.param.declare P1: scalar<si64> = <mod(P0, D0)>
  kgen.param.declare P2: scalar<bool> = <identical(:scalar<si64> P1, 0)>
  kgen.param.declare P3: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "error" : !kgen.string}, P2)>

  // CHECK-NEXT: mo.output
  mo.output
}

// CHECK-LABEL: mo.graph @range_based_folding
// CHECK-SAME:  <[[D0:.*]] -> [[O3:.*]]>(
// CHECK-SAME:    %[[IN:.*]]: !mo.tensor<[], f32>
// CHECK-SAME:  ) -> !mo.tensor<[[[D0]], 0, 0, [[O3]]], f32> {
mo.graph @range_based_folding<D0 -> O0, O1, O2, O3>(
  %in: !mo.tensor<[], f32>
) -> !mo.tensor<[O0, O1, O2, O3], f32> {

  kgen.param.declare P0: scalar<si64> = <max(D0, 0)> // D0
  kgen.param.declare P1: scalar<si64> = <max(mul_no_wrap(D0, -1), 0)> // 0
  kgen.param.declare P2: scalar<si64> = <min(D0, 0)> // 0
  kgen.param.declare P3: scalar<si64> = <min(mul_no_wrap(D0, -1), 0)> // mul_no_wrap(D0, -1)

  // CHECK-NEXT: %[[OUT:.*]] = mo.static.broadcast_to(%[[IN]])
  // CHECK-SAME: : !mo.tensor<[], f32> -> !mo.tensor<[[[D0]], 0, 0, mul_no_wrap([[D0]], -1)], f32>
  %out = mo.static.broadcast_to(%in) : !mo.tensor<[], f32> -> !mo.tensor<[P0, P1, P2, P3], f32>

  // CHECK-NEXT: mo.output<mul_no_wrap([[D0]], -1)> %[[OUT]] : !mo.tensor<[[[D0]], 0, 0, mul_no_wrap([[D0]], -1)], f32>
  mo.output<P0, P1, P2, P3> %out : !mo.tensor<[P0, P1, P2, P3], f32>
}


// This optimization simplifies statements of the form `<max(add(A, B), B)>`
// where we know `A, B >= 0`. We know `A, B >= 0` because they appear in a shape.
// As a result `A + B >= B` and `<max(add(A, B), B)> = add(A, B)`

// CHECK-LABEL: mo.graph @propagate_max_add_optimization
// CHECK-SAME: <
// CHECK-SAME: [[A:[a-zA-Z0-9_]*]], [[B:[a-zA-Z0-9_]*]] ->
// CHECK-SAME: [[D1:[a-zA-Z0-9_]*]], [[D2:[a-zA-Z0-9_]*]], [[D3:[a-zA-Z0-9_]*]], [[D4:[a-zA-Z0-9_]*]]
// CHECK-SAME: >
// CHECK-SAME: (
// CHECK-SAME: [[IN:%[a-zA-Z0-9_]*]]: !mo.tensor<[[[A]], [[B]]], f32>
// CHECK-SAME: ) ->
// CHECK-SAME: (!mo.tensor<[[[D1]], 1], f32>, !mo.tensor<[[[D2]], 1], f32>,
// CHECK-SAME: !mo.tensor<[[[D3]], 1], f32>, !mo.tensor<[[[D4]], 1], f32>)
mo.graph @propagate_max_add_optimization<A, B -> D1, D2, D3, D4>
  (%arg0: !mo.tensor<[A, B], f32>) ->
  (!mo.tensor<[D1, 1], f32>, !mo.tensor<[D2, 1], f32>, !mo.tensor<[D3, 1], f32>, !mo.tensor<[D4, 1], f32>)
{
  kgen.param.declare expr1: scalar<si64> = <max(add(A, B), B)>
  kgen.param.declare expr2: scalar<si64> = <max(add(A, B), A)>
  kgen.param.declare expr3: scalar<si64> = <max(A, add(A, B))>
  kgen.param.declare expr4: scalar<si64> = <max(B, add(A, B))>

  // NOTE: no A and B exist which fulfill this rebind safely, but we just have it
  // to easily test the simplification of this MAX property.
  // CHECK-NEXT: [[V0:%.*]] = mo.rebind([[IN]]) : !mo.tensor<[[[A]], [[B]]], f32> -> !mo.tensor<[add([[A]], [[B]]), 1], f32>
  // CHECK-NEXT: [[V1:%.*]] = mo.rebind([[IN]]) : !mo.tensor<[[[A]], [[B]]], f32> -> !mo.tensor<[add([[A]], [[B]]), 1], f32>
  // CHECK-NEXT: [[V2:%.*]] = mo.rebind([[IN]]) : !mo.tensor<[[[A]], [[B]]], f32> -> !mo.tensor<[add([[A]], [[B]]), 1], f32>
  // CHECK-NEXT: [[V3:%.*]] = mo.rebind([[IN]]) : !mo.tensor<[[[A]], [[B]]], f32> -> !mo.tensor<[add([[A]], [[B]]), 1], f32>
  %a = mo.rebind(%arg0): !mo.tensor<[A, B], f32> -> !mo.tensor<[expr1, 1], f32>
  %b = mo.rebind(%arg0): !mo.tensor<[A, B], f32> -> !mo.tensor<[expr2, 1], f32>
  %c = mo.rebind(%arg0): !mo.tensor<[A, B], f32> -> !mo.tensor<[expr3, 1], f32>
  %d = mo.rebind(%arg0): !mo.tensor<[A, B], f32> -> !mo.tensor<[expr4, 1], f32>

  // CHECK-NEXT: mo.output
  // CHECK-SAME: [[V0]], [[V1]], [[V2]], [[V3]]
  mo.output<expr1, expr2, expr3, expr4> %a, %b, %c, %d : !mo.tensor<[expr1, 1], f32>, !mo.tensor<[expr2, 1], f32>, !mo.tensor<[expr3, 1], f32>, !mo.tensor<[expr4, 1], f32>
}

// CHECK-LABEL: mo.graph @update_param_declare_rhs
// CHECK-SAME:  <[[D1:.*]]>() {
mo.graph @update_param_declare_rhs<D1>() {
  // In cases like this, where we do some folding in the RHS of a
  // `kgen.param.declare`, we still want to update the RHS to the simplified
  // version in the IR (not just in the evaluator during the pass).

  // CHECK-NEXT: kgen.param.declare {{.*}}: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "test" : !kgen.string}, identical(:scalar<si64> [[D1]], 1))>
  // CHECK-NEXT: mo.output
  kgen.param.declare CH: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "test" : !kgen.string}, identical(:scalar<si64> cond(identical(:scalar<si64> D1, -1), 1, D1), 1))>
  mo.output
}

// NOTE this ensures the pass can handle out-of-order ops. The test is somewhat
// contrived, but once we have general SSA->Parameter conversion (e.g., scalar
// input in llama2 that gets used in shape computation), the pass would have to
// deal with such out-of-order cases in practice.
// CHECK-LABEL: mo.graph @out_of_order_ops
// CHECK-SAME:   <() -> [[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]]>(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[], f32>,
// CHECK-SAME:   %[[SHAPE1:.*]]: !mo.tensor<[2], si32>,
// CHECK-SAME:   %[[SHAPE2:.*]]: !mo.tensor<[2], si32>
// CHECK-SAME: ) -> (!mo.tensor<[[[D0]], [[D1]]], f32>, !mo.tensor<[[[D2]], [[D3]]], f32>) {
mo.graph @out_of_order_ops<() -> D0, D1, D2, D3>(
  %input: !mo.tensor<[], f32>,
  %shape1: !mo.tensor<[2], si32>,
  %shape2: !mo.tensor<[2], si32>
) -> (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D2, D3], f32>) {

  // NOTE that SSA dependencies are all preserved, but not parameter
  // dependencies. Moreover, there's a transitive dependency to `%b1` as well
  // (due to the shape function call below). The pass must handle such
  // transitive dependencies across SSA/Param boundaries.
  %b2 = mo.static.broadcast_to(%input): !mo.tensor<[], f32> -> !mo.tensor<[D6, D7], f32>

  mo.invoke_shape_func["mo.broadcast_to"]<() -> D4, D5>(%input, %shape1)
    [dataDeptTensors = [1]] :
    (!mo.tensor<[], f32>, !mo.tensor<[2], si32>)
  %b1 = mo.static.broadcast_to(%input): !mo.tensor<[], f32> -> !mo.tensor<[D4, D5], f32>

  mo.invoke_shape_func["mo.broadcast_to"]<() -> D6, D7>(%b1, %shape2)
    [dataDeptTensors = [1]] :
    (!mo.tensor<[D4, D5], f32>, !mo.tensor<[2], si32>)

  // NOTE that these got sorted based on SSA _and_ parameter dependencies.
  // CHECK-NEXT: mo.invoke_shape_func["mo.broadcast_to"]<() -> [[D4:.*]], [[D5:.*]]>(%[[INPUT]], %[[SHAPE1]])
  // CHECK-SAME:   [dataDeptTensors = [1]] :
  // CHECK-SAME:   (!mo.tensor<[], f32>, !mo.tensor<[2], si32>)
  //
  // CHECK-NEXT: %[[B1:.*]] = mo.static.broadcast_to(%[[INPUT]]) : !mo.tensor<[], f32> -> !mo.tensor<[[[D4]], [[D5]]], f32>

  // CHECK-NEXT: mo.invoke_shape_func["mo.broadcast_to"]<() -> [[D6:.*]], [[D7:.*]]>(%[[B1]], %[[SHAPE2]])
  // CHECK-SAME:   [dataDeptTensors = [1]] :
  // CHECK-SAME:   (!mo.tensor<[[[D4]], [[D5]]], f32>, !mo.tensor<[2], si32>)
  //
  // CHECK-NEXT: %[[B2:.*]] = mo.static.broadcast_to(%[[INPUT]]) : !mo.tensor<[], f32> -> !mo.tensor<[[[D6]], [[D7]]], f32>

  // CHECK-NEXT: mo.output<[[D4]], [[D5]], [[D6]], [[D7]]> %[[B1]], %[[B2]] : !mo.tensor<[[[D4]], [[D5]]], f32>, !mo.tensor<[[[D6]], [[D7]]], f32>
  mo.output<D4, D5, D6, D7> %b1, %b2 : !mo.tensor<[D4, D5], f32>, !mo.tensor<[D6, D7], f32>
}

// CHECK-LABEL: mo.graph @broadcast_to_parametric_shape
// CHECK-SAME:   <[[D0:.*]], [[D1:.*]]>(
// CHECK-SAME:    %[[INPUT:.*]]: !mo.tensor<[[[D0]]], f32>,
// CHECK-SAME:    %[[DUMMY:.*]]: !mo.tensor<[[[D1]]], f32>
// CHECK-SAME:  ) -> !mo.tensor<[[[D0]], [[D1]]], f32> {
mo.graph @broadcast_to_parametric_shape<D0, D1 -> D2, D3>(
  %input: !mo.tensor<[D0], f32>,
  %dummy: !mo.tensor<[D1], f32>
) -> !mo.tensor<[D2, D3], f32> {

  %param_shape = mosh.param.to_value: !mosh.ape = <[D0, D1]>
  %target_shape = mo.shape.to_tensor(%param_shape) -> !mo.tensor<[2], si32>

  // CHECK-NEXT: kgen.param.declare {{.*}}: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"
  // CHECK-SAME:  {message = "[broadcast_to] input dimension at index 0 ([[D0]]) must be either 1 or equal to corresponding output dimension at index 1 ([[D1]])" : !kgen.string},
  // CHECK-SAME:  or(identical(:scalar<si64> D0, D1), identical(:scalar<si64> D0, 1)))>
  //
  // CHECK-NEXT: %[[RES:.*]] = mo.static.broadcast_to(%[[INPUT]])
  // CHECK-SAME:   : !mo.tensor<[[[D0]]], f32> -> !mo.tensor<[[[D0]], [[D1]]], f32>
  %res = mo.broadcast_to<() -> D4, D5>(%input, %target_shape) : (
    !mo.tensor<[D0], f32>, !mo.tensor<[2], si32>
  ) -> !mo.tensor<[D4, D5], f32>

  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[[[D0]], [[D1]]], f32>
  mo.output<D4, D5> %res : !mo.tensor<[D4, D5], f32>
}

// CHECK-LABEL: mo.graph @reshape_parametric_shape
// CHECK-SAME:   <[[D0:.*]], [[D1:.*]]>(
// CHECK-SAME:    %[[INPUT:.*]]: !mo.tensor<[[[D0]], [[D1]]], f32>
// CHECK-SAME:  ) -> !mo.tensor<[[[D0]], 4, 5], f32> {
mo.graph @reshape_parametric_shape<D0, D1 -> D2, D3, D4>(
  %input: !mo.tensor<[D0, D1], f32>
) -> !mo.tensor<[D2, D3, D4], f32> {

  %param_shape = mosh.param.to_value: !mosh.ape = <[D0, 4, 5]>
  %target_shape = mo.shape.to_tensor(%param_shape) -> !mo.tensor<[3], si32>

  // CHECK-NEXT: kgen.param.declare {{.*}}: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"
  // CHECK-SAME:  {message = "[reshape] input and output number of elements must match:
  // CHECK-SAME: #mosh<ape[D0, D1]> : !mosh.ape != #mosh<ape[D0, 4, 5]> : !mosh.ape[[REST:.*]]" : !kgen.string},
  // CHECK-SAME: identical(:scalar<si64> mul_no_wrap(D0, D1), mul_no_wrap(D0, 20)))>
  //
  // CHECK-NEXT: %[[RES:.*]] = mo.static.reshape(%[[INPUT]])
  // CHECK-SAME:   : !mo.tensor<[[[D0]], [[D1]]], f32> -> !mo.tensor<[[[D0]], 4, 5], f32>
  %res = mo.reshape<() -> D5, D6, D7>(%input, %target_shape) : (
    !mo.tensor<[D0, D1], f32>, !mo.tensor<[3], si32>
  ) -> !mo.tensor<[D5, D6, D7], f32>

  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[[[D0]], 4, 5], f32>
  mo.output<D5, D6, D7> %res : !mo.tensor<[D5, D6, D7], f32>
}

// COM: Makes sure we handle the case where we are reshaping scalar tensors
// CHECK-LABEL: mo.graph
// CHECK-SAME: @reshape_scalar_tensor([[VARG0:[^ ]*]]: !mo.tensor<[], f32>) -> !mo.tensor<[1, 1, 1], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.static.reshape([[VARG0]]) : !mo.tensor<[], f32> -> !mo.tensor<[1, 1, 1], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[1, 1, 1], f32>
mo.graph @reshape_scalar_tensor<() -> D2, D3, D4>(
  %input: !mo.tensor<[], f32>
) -> !mo.tensor<[D2, D3, D4], f32> {

  %param_shape = mosh.param.to_value: !mosh.ape = <[1, 1, 1]>
  %target_shape = mo.shape.to_tensor(%param_shape) -> !mo.tensor<[3], si32>
  %res = mo.reshape<() -> D5, D6, D7>(%input, %target_shape) : (
    !mo.tensor<[], f32>, !mo.tensor<[3], si32>
  ) -> !mo.tensor<[D5, D6, D7], f32>

  mo.output<D5, D6, D7> %res : !mo.tensor<[D5, D6, D7], f32>
}

// CHECK-LABEL: mo.graph @fold_abs_on_non_negative_dim
//  CHECK-SAME:  <[[D0:.*]]>(
mo.graph @fold_abs_on_non_negative_dim<D0 -> D1>(
  %dummy: !mo.tensor<[D0], f32>,
  %scalar: !mo.tensor<[], f32>
) -> !mo.tensor<[D1], f32> {

  // CHECK-NEXT: %[[RESULT:.*]] = mo.static.broadcast_to{{.*}} -> !mo.tensor<[D0], f32>
  kgen.param.declare D2: scalar<si64> = <cond(ge(:scalar<si64> D0, 0), D0, mul_no_wrap(D0, -1))>
  %result = mo.static.broadcast_to(%scalar) : !mo.tensor<[], f32> -> !mo.tensor<[D2], f32>

  // CHECK-NEXT: mo.output %[[RESULT]]
  mo.output<D2> %result : !mo.tensor<[D2], f32>
}

// This test case is extracted from ONNX text encoder.
// CHECK-LABEL: mo.graph @select_fold_during_staticization
//  CHECK-SAME:   <[[SEQ_LEN:.*]], [[BATCH_SIZE:.*]]>(
//  CHECK-SAME:   [[INPUT:%.*]]: !mo.tensor<[[[SEQ_LEN:.*]], [[BATCH_SIZE:.*]]], si32>
mo.graph @select_fold_during_staticization<seq_len, batch_size -> O1, O2>(
  %input: !mo.tensor<[seq_len, batch_size], si32>
) -> !mo.tensor<[O1, O2], si32> {

  %zero = mo.constant <#M.dense_array<0> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %zeros = mo.constant <#M.dense_array<0, 0> : !M.array<2xsi64>> : !mo.tensor<[2], si64>
  %one = mo.constant <#M.dense_array<1> : tensor<1xsi64>> : !mo.tensor<[1], si64>

  // Essentially constructing `%target_shape` to represent
  //   `[0 == 0 ? seq_len : 0,  batch_size == 0 ? batch_size : batch_size]
  // which folds to
  //   `[seq_len, batch_size]`
  %input_shape = mo.shape_of(%input) : (!mo.tensor<[seq_len, batch_size], si32>) -> !mo.tensor<[2], si64>
  %batch_size = mo.gather(%input_shape, %one) [axis = 0] : (!mo.tensor<[2], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[1], si64>
  %zero_and_batch_size = mo.concat[0] (%zero, %batch_size) : (!mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[2], si64>
  %cond = mo.equal(%zero_and_batch_size, %zeros) : (!mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2], bool>
  %target_shape = mo.select(%cond, %input_shape, %zero_and_batch_size) : (!mo.tensor<[2], bool>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2], si64>

  // reshape(%input, [seq_len, batch_size]) == %input
  %res = mo.reshape<() -> D1, D2>(%input, %target_shape) : (
    !mo.tensor<[seq_len, batch_size], si32>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[D1, D2], si32>

  // CHECK: mo.output [[INPUT]]
  mo.output<D1, D2> %res : !mo.tensor<[D1, D2], si32>
}

// CHECK-LABEL: mo.graph @range_gather
//  CHECK-SAME:   <[[N1:.*]], [[N2:.*]], [[N3:.*]]>(
//  CHECK-SAME:   [[INPUT:%.*]]: !mo.tensor<[], si32>
mo.graph @range_gather<N1, N2, N3 -> O1, O2>(
  %input: !mo.tensor<[], si32>,
  %dummy: !mo.tensor<[N1, N2, N3], si32>
) -> !mo.tensor<[O1, O2], si32> {

  %one = mo.constant <#M.dense_array<1> : tensor<si32>> : !mo.tensor<[], si32>
  %dummy_shape = mo.shape_of(%dummy) : (!mo.tensor<[N1, N2, N3], si32>) -> !mo.tensor<[3], si32>
  %three_idx = mosh.param.to_value = <3>
  %three = mo.index.to_tensor(%three_idx) -> !mo.tensor<[], si32>
  %range = mo.range(%one, %three, %one) : (!mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[2], si32>
  %shape = mo.gather(%dummy_shape, %range) [axis = 0] : (!mo.tensor<[3], si32>, !mo.tensor<[2], si32>) -> !mo.tensor<[2], si32>

  // CHECK: [[RES:%.*]] = mo.static.broadcast_to([[INPUT]]) : {{.*}} -> !mo.tensor<[[[N2]], [[N3]]], si32>
  %res = mo.broadcast_to<() -> D1, D2>(%input, %shape) : (!mo.tensor<[], si32>, !mo.tensor<[2], si32>) -> !mo.tensor<[D1, D2], si32>

  // CHECK: mo.output [[RES]]
  mo.output<D1, D2> %res : !mo.tensor<[D1, D2], si32>
}

// This test ensures that staticized input actually gets materialized via
// `mo.shape.to_tensor`, so that we don't have to run all the shape computation
// via kernels. This is important for ops which can be shape inferred and have
// non-trivial data dependent input.
// CHECK-LABEL: mo.graph @slice_with_staticized_input
//  CHECK-SAME: <[[D0:.*]]>(
//  CHECK-SAME: [[INPUT:%.*]]: !mo.tensor<[[[D0]], 2304], f32>
//   CHECK-DAG:   [[STEP:%.*]] = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
//   CHECK-DAG:   [[START:%.*]] = mo.constant <#M.dense_array<0, 768> : tensor<2xsi64>> : !mo.tensor<[2], si64>
//   CHECK-DAG:   [[STOP_MOSH:%.*]] = mosh.param.to_value: !mosh.ape = <[[[D0]], 1536]>
//  CHECK-NEXT:   [[STOP:%.*]] = mo.shape.to_tensor([[STOP_MOSH]]) -> !mo.tensor<[2], si64>
//  CHECK-NEXT:   [[RES:%.*]] = mo.slice([[INPUT]], [[START]], [[STOP]], [[STEP]]) : {{.*}} -> !mo.tensor<[D0, 768], f32>
//  CHECK-NEXT:   mo.output [[RES]]
mo.graph @slice_with_staticized_input<D0 -> D1, D2>(
  %input: !mo.tensor<[D0, 2304], f32>
) -> !mo.tensor<[D1, D2], f32> {
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %two = mo.constant <#M.dense_array<2> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %zero = mo.constant <#M.dense_array<0> : !M.array<1xsi64>> : !mo.tensor<[1], si64>
  %neg_one = mo.constant <#M.dense_array<-1> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %three = mo.constant <#M.dense_array<3> : tensor<1xsi64>> : !mo.tensor<[1], si64>

  // The following is a generic way to compute N // 3 (where N == 2304 in our
  // case). It can't be constant folded because `mo.shape_of(%input)` isn't a
  // constant op as it's formulated (the 1st input dimension is dynamic).
  %in_shape = mo.shape_of(%input) : (!mo.tensor<[D0, 2304], f32>) -> !mo.tensor<[2], si64>
  %last_dim = mo.gather(%in_shape, %neg_one) [axis = 0] : (!mo.tensor<[2], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[1], si64>
  %add = mo.add(%last_dim, %two) : !mo.tensor<[1], si64>
  %div = mo.div(%add, %three) : !mo.tensor<[1], si64>
  %mul = mo.mul(%div, %two) : !mo.tensor<[1], si64>

  // Prepare the start/stop bounds.
  %d0 = mo.gather(%in_shape, %zero) [axis = 0] : (!mo.tensor<[2], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[1], si64>
  %start = mo.concat[0] (%zero, %div) : (!mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[2], si64>
  %stop = mo.concat[0] (%d0, %mul) : (!mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[2], si64>

  %res = mo.slice<() -> D3, D4>(%input, %start, %stop, %step) : (!mo.tensor<[D0, 2304], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[D3, D4], f32>
  mo.output<D3, D4> %res : !mo.tensor<[D3, D4], f32>
}

// Ensure shape computations related in a general slice operation are
// propagated.

// CHECK-LABEL: mo.graph @slice
// CHECK:       (%[[ARG0:.*]]: !mo.tensor<[1], si32>) -> !mo.tensor<[2, 2], si32>
// CHECK:       %[[OUT:.*]] = mo.static.broadcast_to(%[[ARG0]]) : !mo.tensor<[1], si32> -> !mo.tensor<[2, 2], si32>
// CHECK:       mo.output %[[OUT]] : !mo.tensor<[2, 2], si32>
mo.graph @slice<() -> O0, O1>(%arg0: !mo.tensor<[1], si32>) -> !mo.tensor<[O0, O1], si32> {
  %0 = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %1 = mo.constant <#M.dense_array<0> : !M.array<1xsi64>> : !mo.tensor<[], si64>
  %c2 = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %c3 = mo.constant <#M.dense_array<3, 3> : tensor<2xsi64>> : !mo.tensor<[2], si64>

  %ones = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %zeros = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>

  %6 = mo.greater(%zeros, %ones) : (!mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2], bool>
  %7 = mo.not(%6) : !mo.tensor<[2], bool>
  %8 = mo.select(%7, %c2, %c3) : (!mo.tensor<[2], bool>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2], si64>

  %out = mo.broadcast_to<() -> D0, D1>(%arg0, %8) : (!mo.tensor<[1], si32>, !mo.tensor<[2], si64>) -> !mo.tensor<[D0, D1], si32>
  mo.output<D0, D1> %out : !mo.tensor<[D0, D1], si32>
}

