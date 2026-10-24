// RUN: mo-opt %s --mogg-to-mgp | FileCheck %s --enable-var-scope

// CHECK-LABEL: mgp.model @add_expressions : mof
// CHECK: } execute

mo.graph @add_expressions<X, Y>(%arg0: !mo.tensor<[X, Y], f32>) {
  //     CHECK: %[[DIM0:.*]] = mgp.tensor_spec.get_dim[0] %{{.*}}
  //     CHECK: %[[DIM1:.*]] = mgp.tensor_spec.get_dim[1] %{{.*}}
  // CHECK-DAG: %[[I0:.*]] = index.casts %[[DIM0]] : index to si64
  // CHECK-DAG: %[[I1:.*]] = index.casts %[[DIM1]] : index to si64
  // CHECK-DAG: %[[D0:.*]] = pop.cast_from_builtin %[[I0]] : si64 to !kgen.scalar<si64>
  // CHECK-DAG: %[[D1:.*]] = pop.cast_from_builtin %[[I1]] : si64 to !kgen.scalar<si64>
  // CHECK-DAG: %[[ADD1:.*]] = pop.add %[[D0]], %[[D1]] : !kgen.scalar<si64>
  kgen.param.declare A0: !kgen.scalar<si64> = <add(X, Y)>
  kgen.param.declare A1: !kgen.scalar<si64> = <add(Y, X)>

  // CHECK-DAG: %[[C2:.*]] = kgen.param.constant: scalar<si64> = <2>
  // CHECK-DAG: %[[ADD2:.*]] = pop.add %[[ADD1]], %[[C2]] : !kgen.scalar<si64>
  // CHECK: mef.output
  kgen.param.declare A2: !kgen.scalar<si64> = <add(A0, 2)>
  mo.output
}

// -----

// CHECK-LABEL: mgp.model @sub_expressions : mof
// CHECK: } execute
mo.graph @sub_expressions<X, Y>(%arg0: !mo.tensor<[X, Y], f32>) {
  //     CHECK: %[[DIM0:.*]] = mgp.tensor_spec.get_dim[0] %{{.*}}
  //     CHECK: %[[DIM1:.*]] = mgp.tensor_spec.get_dim[1] %{{.*}}
  // CHECK-DAG: %[[I0:.*]] = index.casts %[[DIM0]]
  // CHECK-DAG: %[[I1:.*]] = index.casts %[[DIM1]]
  // CHECK-DAG: %[[D0:.*]] = pop.cast_from_builtin %[[I0]] : si64 to !kgen.scalar<si64>
  // CHECK-DAG: %[[D1:.*]] = pop.cast_from_builtin %[[I1]] : si64 to !kgen.scalar<si64>
  // CHECK-DAG: %[[NEG1:.*]] = kgen.param.constant: scalar<si64> = <-1>
  // CHECK-DAG: %[[MUL:.*]] = pop.mul %[[D1]], %[[NEG1]]
  // CHECK-DAG: %[[ADD1:.*]] = pop.add %[[MUL]], %[[D0]]
  kgen.param.declare S0: !kgen.scalar<si64> = <sub(X, Y)>

  // CHECK-DAG: %[[NEG3:.*]] = kgen.param.constant: scalar<si64> = <-3>
  // CHECK-DAG: %[[ADD2:.*]] = pop.add %[[D0]], %[[NEG3]]
  //     CHECK: mef.output
  kgen.param.declare S1: !kgen.scalar<si64> = <sub(X, 3)>
  mo.output
}

// -----

// CHECK-LABEL: mgp.model @mul_expressions : mof
// CHECK: } execute
mo.graph @mul_expressions<X, Y>(%arg0: !mo.tensor<[X, Y], f32>) {
  //     CHECK: %[[DIM0:.*]] = mgp.tensor_spec.get_dim[0] %{{.*}}
  //     CHECK: %[[DIM1:.*]] = mgp.tensor_spec.get_dim[1] %{{.*}}
  // CHECK-DAG: %[[I0:.*]] = index.casts %[[DIM0]]
  // CHECK-DAG: %[[I1:.*]] = index.casts %[[DIM1]]
  // CHECK-DAG: %[[D0:.*]] = pop.cast_from_builtin %[[I0]] : si64 to !kgen.scalar<si64>
  // CHECK-DAG: %[[D1:.*]] = pop.cast_from_builtin %[[I1]] : si64 to !kgen.scalar<si64>
  // CHECK-DAG: %[[MUL0:.*]] = pop.mul %[[D0]], %[[D1]]
  kgen.param.declare M0: !kgen.scalar<si64> = <mul_no_wrap(X, Y)>
  kgen.param.declare M1: !kgen.scalar<si64> = <mul_no_wrap(Y, X)>

  // CHECK-DAG: %[[C2:.*]] = kgen.param.constant: scalar<si64> = <2>
  // CHECK-DAG: %[[MUL1:.*]] = pop.mul %[[MUL0]], %[[C2]]
  //     CHECK: mef.output
  kgen.param.declare M2: !kgen.scalar<si64> = <mul_no_wrap(M0, 2)>
  mo.output
}

// -----

// CHECK-LABEL: mgp.model @mod_expressions : mof
// CHECK: } execute
mo.graph @mod_expressions<X, Y>(%arg0: !mo.tensor<[X, Y], f32>) {
  //     CHECK: %[[DIM0:.*]] = mgp.tensor_spec.get_dim[0] %{{.*}}
  //     CHECK: %[[DIM1:.*]] = mgp.tensor_spec.get_dim[1] %{{.*}}
  // CHECK-DAG: %[[I0:.*]] = index.casts %[[DIM0]]
  // CHECK-DAG: %[[I1:.*]] = index.casts %[[DIM1]]
  // CHECK-DAG: %[[D0:.*]] = pop.cast_from_builtin %[[I0]] : si64 to !kgen.scalar<si64>
  // CHECK-DAG: %[[D1:.*]] = pop.cast_from_builtin %[[I1]] : si64 to !kgen.scalar<si64>
  // CHECK-DAG: %[[REM0:.*]] = pop.rem %[[D0]], %[[D1]]
  kgen.param.declare M0: !kgen.scalar<si64> = <mod(X, Y)>

  // CHECK-DAG: %[[REM1:.*]] = pop.rem %[[D1]], %[[D0]]
  kgen.param.declare M1: !kgen.scalar<si64> = <mod(Y, X)>

  // CHECK-DAG: %[[C2:.*]] = kgen.param.constant: scalar<si64> = <2>
  // CHECK-DAG: %[[REM3:.*]] = pop.rem %[[REM0]], %[[C2]] : !kgen.scalar<si64>
  //     CHECK: mef.output
  kgen.param.declare M2: !kgen.scalar<si64> = <mod(M0, 2)>
  mo.output
}

// -----

// CHECK-LABEL: mgp.model @eq_expressions : mof
// CHECK: } execute
mo.graph @eq_expressions<X, Y>(%arg0: !mo.tensor<[X, Y], f32>) {
  //     CHECK: %[[DIM0:.*]] = mgp.tensor_spec.get_dim[0] %{{.*}}
  //     CHECK: %[[DIM1:.*]] = mgp.tensor_spec.get_dim[1] %{{.*}}
  // CHECK-DAG: %[[I0:.*]] = index.casts %[[DIM0]]
  // CHECK-DAG: %[[I1:.*]] = index.casts %[[DIM1]]
  // CHECK-DAG: %[[D0:.*]] = pop.cast_from_builtin %[[I0]] : si64 to !kgen.scalar<si64>
  // CHECK-DAG: %[[D1:.*]] = pop.cast_from_builtin %[[I1]] : si64 to !kgen.scalar<si64>
  //     CHECK: %[[CMP:.*]] = pop.cmp eq(%[[D0]], %[[D1]])

  // There should be only one pop.cmp as all comparisons are the same
  // CHECK-NOT: pop.cmp
  //     CHECK: mef.output
  kgen.param.declare E0: scalar<bool> = <identical(:scalar<si64> X, Y)>
  kgen.param.declare E1: scalar<bool> = <identical(:scalar<si64> Y, X)>
  kgen.param.declare E2: scalar<bool> = <identical(:scalar<si64> X, X)>
  mo.output
}

// -----

// CHECK-LABEL: mgp.model @lt_expressions : mof
// CHECK: } execute
mo.graph @lt_expressions<X, Y>(%arg0: !mo.tensor<[X, Y], f32>) {
  //     CHECK: %[[DIM0:.*]] = mgp.tensor_spec.get_dim[0] %{{.*}}
  //     CHECK: %[[DIM1:.*]] = mgp.tensor_spec.get_dim[1] %{{.*}}
  // CHECK-DAG: %[[I0:.*]] = index.casts %[[DIM0]] : index to si64
  // CHECK-DAG: %[[I1:.*]] = index.casts %[[DIM1]] : index to si64
  // CHECK-DAG: %[[D0:.*]] = pop.cast_from_builtin %[[I0]] : si64 to !kgen.scalar<si64>
  // CHECK-DAG: %[[D1:.*]] = pop.cast_from_builtin %[[I1]] : si64 to !kgen.scalar<si64>
  // CHECK-DAG: %[[CMP0:.*]] = pop.cmp lt(%[[D0]], %[[D1]]) : <1, si64>
  kgen.param.declare LT0: scalar<bool> = <lt(:scalar<si64> X, Y)>

  // CHECK-DAG: %[[CMP1:.*]] = pop.cmp lt(%[[D1]], %[[D0]])
  kgen.param.declare LT1: scalar<bool> = <lt(:scalar<si64> Y, X)>

  // CHECK-DAG: %[[CMP2:.*]] = pop.cmp lt(%[[D0]], %[[D0]])
  //     CHECK: mef.output
  kgen.param.declare LT2: scalar<bool> = <lt(:scalar<si64> X, X)>
  mo.output
}

// -----

// CHECK-LABEL: mgp.model @le_expressions : mof
// CHECK: } execute
mo.graph @le_expressions<X, Y>(%arg0: !mo.tensor<[X, Y], f32>) {
  //     CHECK: %[[DIM0:.*]] = mgp.tensor_spec.get_dim[0] %{{.*}}
  //     CHECK: %[[DIM1:.*]] = mgp.tensor_spec.get_dim[1] %{{.*}}
  // CHECK-DAG: %[[I0:.*]] = index.casts %[[DIM0]]
  // CHECK-DAG: %[[I1:.*]] = index.casts %[[DIM1]]
  // CHECK-DAG: %[[D0:.*]] = pop.cast_from_builtin %[[I0]] : si64 to !kgen.scalar<si64>
  // CHECK-DAG: %[[D1:.*]] = pop.cast_from_builtin %[[I1]] : si64 to !kgen.scalar<si64>

  // CHECK-DAG: %[[CMP0:.*]] = pop.cmp le(%[[D0]], %[[D1]])
  kgen.param.declare LE0: scalar<bool> = <le(:scalar<si64> X, Y)>

  // CHECK-DAG: %[[CMP1:.*]] = pop.cmp le(%[[D1]], %[[D0]])
  kgen.param.declare LE1: scalar<bool> = <le(:scalar<si64> Y, X)>

  // CHECK-DAG: %[[CMP2:.*]] = pop.cmp le(%[[D0]], %[[D0]])
  //     CHECK: mef.output
  kgen.param.declare LE2: scalar<bool>  = <le(:scalar<si64> X, X)>
  mo.output
}

// -----

// CHECK-LABEL: mgp.model @cond_expression : mof
// CHECK: } execute
mo.graph @cond_expression<X, Y>(%arg0: !mo.tensor<[X, Y], f32>) {
  //     CHECK: %[[DIM0:.*]] = mgp.tensor_spec.get_dim[0] %{{.*}}
  //     CHECK: %[[DIM1:.*]] = mgp.tensor_spec.get_dim[1] %{{.*}}
  // CHECK-DAG: %[[I0:.*]] = index.casts %[[DIM0]]
  // CHECK-DAG: %[[I1:.*]] = index.casts %[[DIM1]]
  // CHECK-DAG: %[[D0:.*]] = pop.cast_from_builtin %[[I0]] : si64 to !kgen.scalar<si64>
  // CHECK-DAG: %[[D1:.*]] = pop.cast_from_builtin %[[I1]] : si64 to !kgen.scalar<si64>
  // CHECK-DAG: %[[CMP:.*]] = pop.cmp le(%[[D0]], %[[D1]])
  kgen.param.declare B: scalar<bool> = <le(:scalar<si64> X, Y)>

  // CHECK-DAG: %[[COND0:.*]] = pop.simd.select %[[CMP]], %[[D0]], %[[D1]] : <1, si64>
  kgen.param.declare M0: !kgen.scalar<si64> = <cond(B, X, Y)>

  // CHECK-DAG: %[[COND1:.*]] = pop.simd.select %[[CMP]], %[[D1]], %[[D0]] : <1, si64>
  kgen.param.declare M1: !kgen.scalar<si64> = <cond(B, Y, X)>

  // CHECK-DAG: %[[C2:.*]] = kgen.param.constant: scalar<si64> = <2>
  // CHECK-DAG: %[[COND2:.*]] = pop.simd.select %[[CMP]], %[[COND0]], %[[C2]] : <1, si64>
  //     CHECK: mef.output
  kgen.param.declare M2: !kgen.scalar<si64> = <cond(B, M0, 2)>
  mo.output
}

// -----

// CHECK-LABEL: mgp.model @bool_expressions : mof
// CHECK: } execute
mo.graph @bool_expressions<E0, E1>(%arg0: !mo.tensor<[E0, E1], f32>) -> !mo.tensor<[], si64> {
  //     CHECK: %[[DIM0:.*]] = mgp.tensor_spec.get_dim[0] %{{.*}}
  //     CHECK: %[[DIM1:.*]] = mgp.tensor_spec.get_dim[1] %{{.*}}
  // CHECK-DAG: %[[I0:.*]] = index.casts %[[DIM0]]
  // CHECK-DAG: %[[I1:.*]] = index.casts %[[DIM1]]
  // CHECK-DAG: %[[D0:.*]] = pop.cast_from_builtin %[[I0]] : si64 to !kgen.scalar<si64>
  // CHECK-DAG: %[[D1:.*]] = pop.cast_from_builtin %[[I1]] : si64 to !kgen.scalar<si64>

  // CHECK-DAG: %[[OR:.*]] = pop.simd.or %[[D0]], %[[D1]] : <1, si64>
  kgen.param.declare O0: !kgen.scalar<si64> = <or(E0, E1)>
  kgen.param.declare O1: !kgen.scalar<si64> = <or(E1, E0)>

  // CHECK-DAG: %[[XOR:.*]] = pop.simd.xor %[[D0]], %[[D1]] : <1, si64>
  kgen.param.declare X0: !kgen.scalar<si64> = <xor(E0, E1)>
  kgen.param.declare X1: !kgen.scalar<si64> = <xor(E1, E0)>

  // CHECK-DAG: %[[AND:.*]] = pop.simd.and %[[D0]], %[[D1]] : <1, si64>
  kgen.param.declare A0: !kgen.scalar<si64> = <and(E0, E1)>
  kgen.param.declare A1: !kgen.scalar<si64> = <and(E1, E0)>

  // Here to ensure the values are not eliminated by CSE
  %reduced = mosh.param.to_value : !kgen.scalar<si64> = <and(O0, O1, X0, X1, A0, A1)>
  %out = mo.index.to_tensor(%reduced) -> !mo.tensor<[], si64>

  //     CHECK: mef.output
  mo.output %out : !mo.tensor<[], si64>
}

// -----

// CHECK-LABEL: mgp.model @complex_expressions : mof
// CHECK: } execute
mo.graph @complex_expressions<X, Y, Z>(%arg0: !mo.tensor<[X, Y, Z], f32>) {
  //     CHECK: %[[DIM0:.*]] = mgp.tensor_spec.get_dim[0] %{{.*}}
  //     CHECK: %[[DIM1:.*]] = mgp.tensor_spec.get_dim[1] %{{.*}}
  //     CHECK: %[[DIM2:.*]] = mgp.tensor_spec.get_dim[2] %{{.*}}
  // CHECK-DAG: %[[I0:.*]] = index.casts %[[DIM0]]
  // CHECK-DAG: %[[I1:.*]] = index.casts %[[DIM1]]
  // CHECK-DAG: %[[I2:.*]] = index.casts %[[DIM2]]
  // CHECK-DAG: %[[D0:.*]] = pop.cast_from_builtin %[[I0]] : si64 to !kgen.scalar<si64>
  // CHECK-DAG: %[[D1:.*]] = pop.cast_from_builtin %[[I1]] : si64 to !kgen.scalar<si64>
  // CHECK-DAG: %[[D2:.*]] = pop.cast_from_builtin %[[I2]] : si64 to !kgen.scalar<si64>

  // CHECK-DAG: %[[MUL0:.*]] = pop.mul %[[D0]], %[[D1]] : !kgen.scalar<si64>
  // CHECK-DAG: %[[MUL1:.*]] = pop.mul %[[D1]], %[[D2]] : !kgen.scalar<si64>
  // CHECK-DAG: %[[ADD:.*]] = pop.add %[[MUL0]], %[[MUL1]] : !kgen.scalar<si64>
  //     CHECK: mef.output
  kgen.param.declare C0: !kgen.scalar<si64> = <add(mul_no_wrap(X, Y), mul_no_wrap(Y, Z))>
  mo.output
}

// CHECK-LABEL: mgp.model @variadic_exprs : mof
// CHECK: } execute
mo.graph @variadic_exprs<X, Y, Z>(%arg0: !mo.tensor<[X, Y, Z], f32>) {
  //     CHECK: %[[DIM0:.*]] = mgp.tensor_spec.get_dim[0] %{{.*}}
  //     CHECK: %[[DIM1:.*]] = mgp.tensor_spec.get_dim[1] %{{.*}}
  //     CHECK: %[[DIM2:.*]] = mgp.tensor_spec.get_dim[2] %{{.*}}
  // CHECK-DAG: %[[I0:.*]] = index.casts %[[DIM0]]
  // CHECK-DAG: %[[I1:.*]] = index.casts %[[DIM1]]
  // CHECK-DAG: %[[I2:.*]] = index.casts %[[DIM2]]
  // CHECK-DAG: %[[D0:.*]] = pop.cast_from_builtin %[[I0]] : si64 to !kgen.scalar<si64>
  // CHECK-DAG: %[[D1:.*]] = pop.cast_from_builtin %[[I1]] : si64 to !kgen.scalar<si64>
  // CHECK-DAG: %[[D2:.*]] = pop.cast_from_builtin %[[I2]] : si64 to !kgen.scalar<si64>

  // CHECK-DAG: %[[ADD0:.*]] = pop.add %[[D0]], %[[D1]] : !kgen.scalar<si64>
  // CHECK-DAG: %[[ADD1:.*]] = pop.add %[[ADD0]], %[[D2]] : !kgen.scalar<si64>
  //     CHECK: mef.output
  kgen.param.declare C0: !kgen.scalar<si64> = <add(X, Y, Z)>
  mo.output
}

// -----

// CHECK-LABEL: mgp.model @shape_attr : mof
// CHECK: } execute
mo.graph @shape_attr<X, Y, Z>(%arg0: !mo.tensor<[X, Y, Z], f32>) -> !mo.tensor<[3], si64> {
  //     CHECK: %[[DIM0:.*]] = mgp.tensor_spec.get_dim[0] %{{.*}}
  //     CHECK: %[[DIM1:.*]] = mgp.tensor_spec.get_dim[1] %{{.*}}
  //     CHECK: %[[DIM2:.*]] = mgp.tensor_spec.get_dim[2] %{{.*}}
  // CHECK-DAG: %[[I0:.*]] = index.casts %[[DIM0]] : index to si64
  // CHECK-DAG: %[[I1:.*]] = index.casts %[[DIM1]] : index to si64
  // CHECK-DAG: %[[I2:.*]] = index.casts %[[DIM2]] : index to si64
  // CHECK-DAG: %[[D0:.*]] = index.casts %[[I0]] : si64 to index
  // CHECK-DAG: %[[D1:.*]] = index.casts %[[I1]] : si64 to index
  // CHECK-DAG: %[[D2:.*]] = index.casts %[[I2]] : si64 to index
  // CHECK-DAG: %[[SH:.*]] = mgp.tensor_spec.create %[[D0]], %[[D1]], %[[D2]] : !mgp.tensor_spec<?x?x?xsi64>

  kgen.param.declare SH : !mosh.ape = <[X, Y, Z]>

  //     CHECK: mgp.tensor_spec.get_dim[0] %[[SH]]
  //     CHECK: mgp.tensor_spec.get_dim[1] %[[SH]]
  //     CHECK: mgp.tensor_spec.get_dim[2] %[[SH]]
  //     CHECK: mef.output
  %Sh = mosh.param.to_value : !mosh.ape = <SH>
  %res = mo.shape.to_tensor(%Sh) -> !mo.tensor<[3], si64>
  mo.output %res : !mo.tensor<[3], si64>
}

// -----

// CHECK-LABEL: mgp.model @param_expr_in_output_shape : mof
// CHECK: } execute
mo.graph @param_expr_in_output_shape<D0>(
  %arg0: !mo.tensor<[D0, 4], f32>
) -> !mo.tensor<[mul_no_wrap(D0, 2), 2], f32> {
  //     CHECK: %[[DIM0:.*]] = mgp.tensor_spec.get_dim[0] %{{.*}}
  // CHECK-DAG: %[[I0:.*]] = index.casts %[[DIM0]] : index to si64
  // CHECK-DAG: %[[D0:.*]] = pop.cast_from_builtin %[[I0]] : si64 to !kgen.scalar<si64>

  // CHECK-DAG: %[[C2:.*]] = kgen.param.constant: scalar<si64> = <2>
  // CHECK-DAG: %[[MUL:.*]] = pop.mul %[[D0]], %[[C2]] : !kgen.scalar<si64>
  // CHECK-DAG: %[[MULSI:.*]] = pop.cast_to_builtin %[[MUL]] : !kgen.scalar<si64> to si64
  // CHECK-DAG: %[[INDEX:.*]] = index.casts %[[MULSI]] : si64 to index
  // CHECK-DAG: %[[SPEC:.*]] = mgp.tensor_spec.create %[[INDEX]] : !mgp.tensor_spec<?x2xf32>
  // CHECK-DAG: %[[TENSOR:.*]] = mgp.tensor.create %{{.*}}, %{{.*}}, %[[SPEC]] : !mgp.tensor<?x2xf32>
  //     CHECK: mef.output {{.*}}%[[TENSOR]]
  %res = mo.static.reshape(%arg0) : !mo.tensor<[D0, 4], f32> -> !mo.tensor<[mul_no_wrap(D0, 2), 2], f32>
  mo.output %res : !mo.tensor<[mul_no_wrap(D0, 2), 2], f32>
}

// -----

// CHECK-LABEL: mgp.model @shift_expressions : mof
// CHECK: } execute
mo.graph @shift_expressions<X, Y, Z>(%arg0: !mo.tensor<[X, Y, Z], f32>) {
  //     CHECK: %[[DIM0:.*]] = mgp.tensor_spec.get_dim[0] %{{.*}}
  //     CHECK: %[[DIM1:.*]] = mgp.tensor_spec.get_dim[1] %{{.*}}
  //     CHECK: %[[DIM2:.*]] = mgp.tensor_spec.get_dim[2] %{{.*}}
  // CHECK-DAG: %[[I0:.*]] = index.casts %[[DIM0]]
  // CHECK-DAG: %[[I1:.*]] = index.casts %[[DIM1]]
  // CHECK-DAG: %[[I2:.*]] = index.casts %[[DIM2]]
  // CHECK-DAG: %[[D0:.*]] = pop.cast_from_builtin %[[I0]] : si64 to !kgen.scalar<si64>
  // CHECK-DAG: %[[D1:.*]] = pop.cast_from_builtin %[[I1]] : si64 to !kgen.scalar<si64>
  // CHECK-DAG: %[[D2:.*]] = pop.cast_from_builtin %[[I2]] : si64 to !kgen.scalar<si64>

  // CHECK-DAG: %[[SLH:.*]] = pop.shl %[[D0]], %[[D1]]
  kgen.param.declare SHL: !kgen.scalar<si64> = <shl(X, Y)>

  // CHECK-DAG: %[[SHR:.*]] = pop.shr %[[D1]], %[[D2]]
  kgen.param.declare SHR: !kgen.scalar<si64> = <shr(Y, Z)>

  //     CHECK: mef.output
  mo.output
}

// -----

// CHECK-LABEL: mgp.model @from_param : mof
// CHECK: } execute
mo.graph @from_param<X, Y, Z>() -> !mo.tensor<[3], si64> {
  // CHECK-DAG: %[[X:.*]] = mosh.param.to_value = <X>
  // CHECK-DAG: %[[XB:.*]] = pop.cast_to_builtin %[[X]] : !kgen.scalar<si64> to si64
  // CHECK-DAG: %[[IX:.*]] = index.casts %[[XB]] : si64 to index
  // CHECK-DAG: %[[Y:.*]] = mosh.param.to_value = <Y>
  // CHECK-DAG: %[[YB:.*]] = pop.cast_to_builtin %[[Y]] : !kgen.scalar<si64> to si64
  // CHECK-DAG: %[[IY:.*]] = index.casts %[[YB]] : si64 to index
  // CHECK-DAG: %[[Z:.*]] = mosh.param.to_value = <Z>
  // CHECK-DAG: %[[ZB:.*]] = pop.cast_to_builtin %[[Z]] : !kgen.scalar<si64> to si64
  // CHECK-DAG: %[[IZ:.*]] = index.casts %[[ZB]] : si64 to index
  // CHECK-DAG: %[[SPEC:.*]] = mgp.tensor_spec.create %[[IX]], %[[IY]], %[[IZ]] : !mgp.tensor_spec<?x?x?xsi64>
  // CHECK:     %[[DIM0:.*]] = mgp.tensor_spec.get_dim[0] %[[SPEC]] : <?x?x?xsi64>
  // CHECK:     %[[DIM1:.*]] = mgp.tensor_spec.get_dim[1] %[[SPEC]] : <?x?x?xsi64>
  // CHECK:     %[[DIM2:.*]] = mgp.tensor_spec.get_dim[2] %[[SPEC]] : <?x?x?xsi64>
  // CHECK:     %[[BUF:.*]] = mgp.buffer.set_with_index %{{.*}}, %[[DIM0]], %[[DIM1]], %[[DIM2]] : !mgp.buffer
  // CHECK:     %[[TENSOR:.*]] = mgp.tensor.create %[[BUF]], %{{.*}}, %{{.*}} : !mgp.tensor<3xsi64>
  // CHECK:     mef.output %{{.*}}, %[[TENSOR]] : !mef.chain, !mgp.tensor<3xsi64>

  %val = mosh.param.to_value : !mosh.ape = <[X, Y, Z]>

  mosh.param.from_value FROM_VALUE : !mosh.ape = %val

  %Sh = mosh.param.to_value : !mosh.ape = <FROM_VALUE>
  %res = mo.shape.to_tensor(%Sh) -> !mo.tensor<[3], si64>
  mo.output %res : !mo.tensor<[3], si64>
}
