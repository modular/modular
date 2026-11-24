// RUN: mo-opt --canonicalize %s | FileCheck --enable-var-scope %s

// CHECK-LABEL: kgen.func @assert
// CHECK-SAME: (%[[ARG0:.*]]: index, %[[ARG1:.*]]: index)
kgen.func @assert(%arg0: index, %arg1: index) -> !mo.chain {
  // CHECK: %[[CMP:.*]] = index.cmp eq(%[[ARG0]], %[[ARG1]])
  %cmp = index.cmp eq(%arg0, %arg1)
  // CHECK: %[[VAL0:.*]] = pop.cast_from_builtin %[[CMP]] : i1 to !kgen.scalar<bool>
  %0 = pop.cast_from_builtin %cmp : i1 to !kgen.scalar<bool>
  // CHECK: mo.assert %[[VAL0]], "args are not equal"
  %ch0 = mo.assert %0, "args are not equal"
  hlcf.return %ch0 : !mo.chain
}

// CHECK-LABEL: kgen.func @assert_sequence
// CHECK-SAME: ([[ARG0:%.*]]: index, [[ARG1:%.*]]: index, [[ARG2:%.*]]: index, [[ARG3:%.*]]: index)
kgen.func @assert_sequence(%arg0: index, %arg1: index, %arg2: index, %arg3: index) -> !mo.chain {

  // CHECK-DAG: [[CMP0:%.*]] = index.cmp eq([[ARG0]], [[ARG1]])
  // CHECK-DAG: [[COND0:%.*]] = pop.cast_from_builtin [[CMP0]] : i1 to !kgen.scalar<bool>
  // CHECK-DAG: [[CMP1:%.*]] = index.cmp eq([[ARG2]], [[ARG3]])
  // CHECK-DAG: [[COND1:%.*]] = pop.cast_from_builtin [[CMP1]] : i1 to !kgen.scalar<bool>

  %cmp0 = index.cmp eq(%arg0, %arg1)
  %0 = pop.cast_from_builtin %cmp0 : i1 to !kgen.scalar<bool>
  %cmp1 = index.cmp eq(%arg2, %arg3)
  %1 = pop.cast_from_builtin %cmp1 : i1 to !kgen.scalar<bool>

  // CHECK-DAG: [[CH0:%.*]] = mo.assert [[COND0]]
  // CHECK-DAG: mo.assert[[[CH0]]] [[COND1]]

  %ch0 = mo.assert %0, "first pair of args are not equal"
  %ch1 = mo.assert[%ch0] %1, "second pair of args are not equal"

  hlcf.return %ch1 : !mo.chain
}

// CHECK-LABEL: mo.graph @assert_sequence_param_level
mo.graph @assert_sequence_param_level<D0, D1, D2, D3>(%arg0: !mo.tensor<[D0, D1], f32>, %arg1: !mo.tensor<[D2, D3], f32>, %inchain: !mo.chain)
 -> (!mo.tensor<[D0, D3], f32>)
{
  // CHECK-DAG: kgen.param.declare [[CH0:.*]]: !mo.chain
  // CHECK-DAG: kgen.param.declare{{.*}}!kgen.string}, [[CH0]]
  kgen.param.declare CH0: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "D1 != D2" : !kgen.string}, identical(:scalar<si64> D1, D2))>
  kgen.param.declare CH1: !mo.chain = <apply(:(!mo.chain, !kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "D2 != D3" : !kgen.string}, CH0, identical(:scalar<si64> D2, D3))>

  %3 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D2, D3], f32>) -> !mo.tensor<[D0, D3], f32>

  mo.output %3 : !mo.tensor<[D0, D3], f32>
}
