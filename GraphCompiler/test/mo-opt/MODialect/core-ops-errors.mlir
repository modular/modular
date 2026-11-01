// RUN: mo-opt %s --allow-unregistered-dialect --split-input-file --verify-diagnostics

//===----------------------------------------------------------------------===//
// GraphOp and TensorType
//===----------------------------------------------------------------------===//

mo.graph @tensor_incorrect_syntax(%arg0: !mo.tensor<[], f32>) {
  // expected-error@+1 {{couldn't parse shape parameter: expected literal shape reference (i.e. `[...]`)}}
  %0 = mo.rebind(%arg0) : !mo.tensor<[], f32> -> !mo.tensor<*, f32>
  mo.output
}

// -----

// expected-error @below {{invalid dimension, expect all dimensions >= 0 got: -1}}
mo.graph @tensor_type_neg_dim(%arg0: !mo.tensor<[-1, 10], f32>) -> !mo.tensor<[-1, 10], f32> {
  mo.output %arg0: !mo.tensor<[-1, 10], f32>
}

// -----

mo.graph @graph_early_output(%arg0: !mo.tensor<[], f32>) -> !mo.tensor<[], f32> {
  // expected-error @below {{'mo.output' op must be the last operation in the parent block}}
  mo.output %arg0 : !mo.tensor<[], f32>
  %0 = mo.constant <#M.dense_array<1, 2> : tensor<2xui32>> : !mo.tensor<[2], ui32>
}

// -----

// expected-error @below {{'mo.graph' op expects regions to end with 'mo.output', found 'graph.return'}}
// expected-note @below {{in custom textual format, the absence of terminator implies 'mo.output'}}
mo.graph @graph_kgen_return(%arg0: !mo.tensor<[], f32>) -> !mo.tensor<[], f32> {
  "graph.return"(%arg0) : (!mo.tensor<[], f32>) -> ()
}

// -----

mo.graph @graph_multiple_output(%arg0: !mo.tensor<[], f32>) -> !mo.tensor<[], f32> {
  // expected-error @below {{'mo.output' op must be the last operation in the parent block}}
  mo.output %arg0 : !mo.tensor<[], f32>
  mo.output %arg0 : !mo.tensor<[], f32>
}

// -----

// expected-error @below {{cyclic reference between expressions defining and using parameters}}
mo.graph @cycle<D0 -> O0, O1, O2, O3>(
  %input1:    !mo.tensor<[D0, 3], f32>,
  %input2:    !mo.tensor<[], f32>
) -> (!mo.tensor<[O0, O1], f32>, !mo.tensor<[O2, O3], f32>) {

  %temp1 = mo.static.broadcast_to(%input1) : !mo.tensor<[D0, 3], f32> -> !mo.tensor<[D5, D6], f32>
  // expected-note @below {{parameter "D3" is defined here, which references the expression:}}
  mo.invoke_shape_func["arg_nonzero_shape"]<() -> D3, D4>(%temp1)
    [dataDeptTensors = [0 : i32]] : (!mo.tensor<[D5, D6], f32>)

  %temp2 = mo.static.broadcast_to(%input2) : !mo.tensor<[], f32> -> !mo.tensor<[D3, D4], f32>
  // expected-note @below {{parameter "D5" is defined here, which references the first expression}}
  mo.invoke_shape_func["arg_nonzero_shape"]<() -> D5, D6>(%temp2)
    [dataDeptTensors = [0 : i32]] : (!mo.tensor<[D3, D4], f32>)

  mo.output<D5, D6, D3, D4> %temp1, %temp2 : !mo.tensor<[D5, D6], f32>, !mo.tensor<[D3, D4], f32>
}

// -----

// expected-note  @below {{see existing symbol definition here}}
mo.graph @repro(%in: !mo.tensor<[1], f32>) -> !mo.tensor<[1], f32> {
  mo.output %in : !mo.tensor<[1], f32>
}

// expected-error @below {{redefinition of symbol named 'repro'}}
mo.graph @repro(%in: !mo.tensor<[1], f32>) -> !mo.tensor<[1], f32> {
  mo.output %in : !mo.tensor<[1], f32>
}

// -----

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

mo.graph @constant_illegal_attr() -> !mo.tensor<[4], f32> {
  // expected-error@+1 {{'mo.constant' op attribute 'value' failed to satisfy constraint}}
  %0 = mo.constant <dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>> : !mo.tensor<[4], f32>
  mo.output %0 : !mo.tensor<[4], f32>
}

// -----

mo.graph @constant_dtype() -> !mo.tensor<[1, 2], si32> {
  // expected-error @below {{'mo.constant' op element type of the value ('si64') must match the element type of the result ('si32')}}
  %res = mo.constant <#M.dense_array<1, 2> : tensor<1x2xsi64>> : !mo.tensor<[1, 2], si32>
  mo.output %res : !mo.tensor<[1, 2], si32>
}

// -----

mo.graph @constant_dtype_bool() -> !mo.tensor<[], bool> {
  // expected-error @below {{'mo.constant' op for boolean tensors, the element type of the value must be an integer type, but got 'f32'}}
  %res = mo.constant <#M.dense_array<1.0> : tensor<1xf32>> : !mo.tensor<[], bool>
  mo.output %res : !mo.tensor<[], bool>
}

// -----

//===----------------------------------------------------------------------===//
// IfOp and YieldOp
//===----------------------------------------------------------------------===//

mo.graph @if_op_missing_yield(%arg0: !mo.tensor<[2, ?, 1], f32>, %arg1: !mo.tensor<[2, ?, 1], f32>, %cond : !mo.tensor<[], bool>) -> (!mo.tensor<[2, ?, 1], f32>) {
  // expected-note@+2 {{region branch point}}
  // expected-error@+1 {{'mo.if' op along control flow edge from Operation mo.yield to Operation mo.if: region branch point has 0 operands, but region successor needs 1 inputs}}
  %0 = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.tensor<[2, ?, 1], f32> {
    mo.yield %arg0 : !mo.tensor<[2, ?, 1], f32>
  } else {
  }
  mo.output %0 : !mo.tensor<[2, ?, 1], f32>
}

// -----

mo.graph @if_op_cond_mismatch(%arg0: !mo.tensor<[2, ?, 1], f32>, %arg1: !mo.tensor<[2, ?, 1], f32>, %cond : !mo.tensor<[5], bool>) -> (!mo.tensor<[2, ?, 1], f32>) {
    // expected-error@+1 {{'mo.if' op operand #0 must be rank 0 and boolean element type or boolean scalar, but got '!mo.tensor<[5], bool>'}}
  %0 = mo.if(%cond : !mo.tensor<[5], bool>) -> !mo.tensor<[2, ?, 1], f32> {
    mo.yield %arg0 : !mo.tensor<[2, ?, 1], f32>
  } else {
    mo.yield %arg1 : !mo.tensor<[2, ?, 1], f32>
  }
  mo.output %0 : !mo.tensor<[2, ?, 1], f32>
}

// -----

mo.graph @if_op_type_mismatch(%arg0: !mo.tensor<[2, ?, 1], f32>, %arg1: !mo.tensor<[3, ?, 1], f32>, %cond : !mo.tensor<[], bool>) -> (!mo.tensor<[2, ?, 1], f32>) {
  // expected-error@+1 {{'mo.if' op along control flow edge from Operation mo.yield to Operation mo.if: successor operand type #0 '!mo.tensor<[3, ?, 1], f32>' should match successor input type #0 '!mo.tensor<[2, ?, 1], f32>'}}
  %0 = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.tensor<[2, ?, 1], f32> {
    mo.yield %arg0 : !mo.tensor<[2, ?, 1], f32>
  } else {
    // expected-note@+1 {{region branch point}}
    mo.yield %arg1 : !mo.tensor<[3, ?, 1], f32>
  }
  mo.output %0 : !mo.tensor<[2, ?, 1], f32>
}

// -----

mo.graph @if_op_yield_returning_buffer(%arg0: !mo.buffer<[2, 2], f32>, %cond : !mo.tensor<[], bool>) {
  // expected-error@+1 {{'mo.if' op result #0 must be variadic of any type but !mo.buffer}}
  %0 = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.buffer<[2, 2], f32> {
    mo.yield %arg0 : !mo.buffer<[2, 2], f32>
  } else {
    mo.yield %arg0 : !mo.buffer<[2, 2], f32>
  }
  mo.output
}

// -----

//===----------------------------------------------------------------------===//
// WhileOp and YieldOp
//===----------------------------------------------------------------------===//

mo.graph @while_op_wrong_terminator(%arg0: !mo.tensor<[2], f32>, %arg1: !mo.tensor<[3], f32>, %cond : !mo.tensor<[], bool>) -> !mo.tensor<[2], f32> {

  // expected-error@+1 {{'mo.while' op expects the 'before' region to terminate with 'mo.while.condition'}}
  %0:2 = mo.while (%arg0 as %argi0: !mo.tensor<[2], f32>, %cond as %condi: !mo.tensor<[], bool>) {
    // expected-note@+1 {{terminator here}}
    mo.yield %condi, %argi0, %condi : !mo.tensor<[], bool>, !mo.tensor<[2], f32>, !mo.tensor<[], bool>
  } do {
    mo.yield %argi0, %condi : !mo.tensor<[2], f32>, !mo.tensor<[], bool>
  }
  mo.output %0#0 : !mo.tensor<[2], f32>
}

// -----

mo.graph @while_op_type_mismatch_cond(%arg0: !mo.tensor<[2], f32>, %arg1: !mo.tensor<[3], f32>, %cond : !mo.tensor<[], bool>) -> !mo.tensor<[2], f32> {

  // expected-note@+1 {{prior use here}}
  %0:2 = mo.while (%arg0 as %argi0: !mo.tensor<[2], f32>, %cond as %condi: !mo.tensor<[], bool>) {
    // expected-error@+1 {{use of value '%argi0' expects different type than prior uses: '!mo.tensor<[], bool>' vs '!mo.tensor<[2], f32>'}}
    mo.while.condition(%argi0) %argi0, %condi : !mo.tensor<[2], f32>, !mo.tensor<[2], f32>, !mo.tensor<[], bool>
  } do {
    mo.yield %argi0, %condi : !mo.tensor<[2], f32>, !mo.tensor<[], bool>
  }
  mo.output %0#0 : !mo.tensor<[2], f32>
}

// -----

mo.graph @while_op_arg_count_mismatch_cond(%arg0: !mo.tensor<[2], f32>, %arg1: !mo.tensor<[3], f32>, %cond : !mo.tensor<[], bool>) -> !mo.tensor<[2], f32> {

    // expected-error@+1 {{'mo.while' op along control flow edge from Operation mo.while.condition to Operation mo.while: region branch point has 1 operands, but region successor needs 2 inputs}}
  %0:2 = mo.while (%arg0 as %argi0: !mo.tensor<[2], f32>, %cond as %condi: !mo.tensor<[], bool>) {
    // expected-note@+1 {{region branch point}}
    mo.while.condition(%condi) %argi0 : !mo.tensor<[2], f32>
  } do {
    mo.yield %argi0, %condi : !mo.tensor<[2], f32>, !mo.tensor<[], bool>
  }
  mo.output %0#0 : !mo.tensor<[2], f32>
}

// -----

mo.graph @while_op_arg_count_mismatch_cond(%arg0: !mo.tensor<[2], f32>, %arg1: !mo.tensor<[3], f32>, %cond : !mo.tensor<[], bool>) -> !mo.tensor<[2], f32> {

  // expected-error@+1 {{'mo.while' op along control flow edge from Operation mo.while.condition to Operation mo.while: successor operand type #0 '!mo.tensor<[], bool>' should match successor input type #0 '!mo.tensor<[2], f32>'}}
  %0:2 = mo.while (%arg0 as %argi0: !mo.tensor<[2], f32>, %cond as %condi: !mo.tensor<[], bool>) {
    // expected-note@+1 {{region branch point}}
    mo.while.condition(%condi) %condi, %argi0 : !mo.tensor<[], bool>, !mo.tensor<[2], f32>
  } do {
    mo.yield %argi0, %condi : !mo.tensor<[2], f32>, !mo.tensor<[], bool>
  }
  mo.output %0#0 : !mo.tensor<[2], f32>
}

// -----

mo.graph @while_op_type_mismatch_body(%arg0: !mo.tensor<[2], f32>, %arg1: !mo.tensor<[3], f32>, %cond : !mo.tensor<[], bool>) -> !mo.tensor<[2], f32> {
  // expected-error@+1 {{'mo.while' op along control flow edge from Operation mo.yield to Region #0: successor operand type #0 '!mo.tensor<[3], f32>' should match successor input type #0 '!mo.tensor<[2], f32>'}}
  %0:3 = mo.while (%arg0 as %argi0: !mo.tensor<[2], f32>, %arg1 as %argi1: !mo.tensor<[3], f32>, %cond as %condi: !mo.tensor<[], bool>) {
    mo.while.condition(%condi) %argi0, %argi1, %condi : !mo.tensor<[2], f32>, !mo.tensor<[3], f32>, !mo.tensor<[], bool>
  } do {
    // expected-note@+1 {{region branch point}}
    mo.yield %argi1, %argi0, %condi : !mo.tensor<[3], f32>, !mo.tensor<[2], f32>, !mo.tensor<[], bool>
  }
  mo.output %0#0 : !mo.tensor<[2], f32>
}

// -----

//===----------------------------------------------------------------------===//
// StableCreateOp
//===----------------------------------------------------------------------===//

// The input must be the result minus the stable marker, so an already-stable
// input can only be written in the generic form and fails the type constraint.
mo.graph @stable_create_twice(%arg0 : !mo.tensor<[3, 2], f32, gpu:0, {stable}>) -> !mo.tensor<[3, 2], f32, gpu:0, {stable}> {
  // expected-error @below {{failed to verify that input is the result without the stable marker}}
  %0 = "mo.stable.create"(%arg0) : (!mo.tensor<[3, 2], f32, gpu:0, {stable}>) -> !mo.tensor<[3, 2], f32, gpu:0, {stable}>
  mo.output %0 : !mo.tensor<[3, 2], f32, gpu:0, {stable}>
}

// -----

mo.graph @stable_create_result_not_stable(%arg0 : !mo.tensor<[3, 2], f32, gpu:0>) -> !mo.tensor<[3, 2], f32, gpu:0> {
  // expected-error @below {{result must be marked `stable`}}
  %0 = "mo.stable.create"(%arg0) : (!mo.tensor<[3, 2], f32, gpu:0>) -> !mo.tensor<[3, 2], f32, gpu:0>
  mo.output %0 : !mo.tensor<[3, 2], f32, gpu:0>
}

// -----

//===----------------------------------------------------------------------===//
// TransferOp
//===----------------------------------------------------------------------===//

mo.graph @transfer_dtype_mismatch(%arg0 : !mo.tensor<[3, 2], f32>) -> !mo.tensor<[3, 2], si32> {
  %c0 = mo.chain.create()
  // expected-error @below {{expected result type to only differ from input type in device}}
  %0:2 = "mo.transfer"(%arg0, %c0) : (!mo.tensor<[3, 2], f32>, !mo.chain) -> (!mo.tensor<[3, 2], si32>, !mo.chain)
  mo.output %0#0 : !mo.tensor<[3, 2], si32>
}

// -----

mo.graph @transfer_shape_mismatch(%arg0 : !mo.tensor<[3, 2], f32>) -> !mo.tensor<[5], f32> {
  %c0 = mo.chain.create()
  // expected-error @below {{expected result type to only differ from input type in device}}
  %0:2 = "mo.transfer"(%arg0, %c0) : (!mo.tensor<[3, 2], f32>, !mo.chain) -> (!mo.tensor<[5], f32>, !mo.chain)
  mo.output %0#0 : !mo.tensor<[5], f32>
}
