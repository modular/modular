// RUN: mo-opt --mogg-to-mgp %s | FileCheck --enable-var-scope %s

// In the following test we don't expect any guard op to be materialized even
// though the second assertion consumes a chain from the first and they both
// have a condition that depends on the param D2 as there's no actual
// dependency between the calculations performed between the two asserts

// CHECK-LABEL: mgp.model @assert_sequence
// CHECK:       [[COND0:%.*]] = pop.cmp
// CHECK:       [[CASTED0:%.*]] = pop.cast_to_builtin [[COND0]]
// CHECK:       [[CH0:%.*]] = mgp.assert [[CASTED0]], "D1 != D2"
// CHECK:       [[COND1:%.*]] = pop.cmp
// CHECK:       [[CASTED1:%.*]] = pop.cast_to_builtin [[COND1]]
// CHECK-NOT:   mgp.guard
// CHECK:       [[CH1:%.*]] = mgp.assert [[CASTED1]], "D2 != D3"
mo.graph @assert_sequence<D0, D1, D2, D3>(%arg0: !mo.tensor<[D0, D1], f32>, %arg1: !mo.tensor<[D2, D3], f32>)
 -> (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D2, D3], f32>)
{
  kgen.param.declare CH0: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "D1 != D2" : !kgen.string}, identical(:scalar<si64> D1, D2))>
  kgen.param.declare CH1: !mo.chain = <apply(:(!mo.chain, !kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "D2 != D3" : !kgen.string}, CH0, identical(:scalar<si64> D2, D3))>

  mo.output %arg0, %arg1 : !mo.tensor<[D0, D1], f32>, !mo.tensor<[D2, D3], f32>
}

// In this case we do expect a guard op to be materialized as the param 'DENOM'
// is used in the calculation for the condition for both assertions and the
// second assert consumes a chain from the first

// CHECK-LABEL: @needs_guard
// CHECK: [[NEG1:%.*]] = kgen.param.constant: scalar<si64> = <-1>
// CHECK: [[DENOM:%.*]] = pop.add [[BS:%.*]], [[NEG1]]
// CHECK: [[CH0:%.*]] = mgp.assert {{.*}} "error 1"
// CHECK: mgp.guard [[CH0]], [[DENOM]]
// CHECK: [[CH1:%.*]] = mgp.assert{{.*}} "error 2"
mo.graph @needs_guard<D0, D1, D2>(%arg0 : !mo.tensor<[D1, D1, D2], f32>) {

  kgen.param.declare DENOM: scalar<si64> = <add(D0, -1)>
  kgen.param.declare COND1: scalar<bool> = <ne(:scalar<si64> DENOM, 0)>
  kgen.param.declare CH0: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "error 1" : !kgen.string}, COND1)>

  kgen.param.declare RATIO: scalar<si64> = <div(D2, DENOM)>
  kgen.param.declare COND2: scalar<bool> = <ge(:scalar<si64> RATIO, 2)>

  kgen.param.declare CH1: !mo.chain = <apply(:(!mo.chain, !kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "error 2" : !kgen.string}, CH0, COND2)>

  mo.output
}

// CHECK-LABEL: @matching_conds
// CHECK-NOT: mo.guard
// CHECK: mgp.assert
// CHECK: mgp.assert
mo.graph @matching_conds<S2, C1_S2, RS, BC1, BC2, BC3, BC4>()
{
  kgen.param.declare MHA_BCAST_CHECK1: scalar<bool> = <and(identical(:scalar<si64> BC1, 1), identical(:scalar<si64> BC2, 6), identical(:scalar<si64> BC3, S2), identical(:scalar<si64> BC4, C1_S2))>
  kgen.param.declare MHA_BCAST_CHECK2: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "error 1" : !kgen.string}, MHA_BCAST_CHECK1)>

  kgen.param.declare MHA_BCAST_CHECK3: scalar<bool> = <and(identical(:scalar<si64> BC1, 1), identical(:scalar<si64> BC2, 6), identical(:scalar<si64> BC3, S2), identical(:scalar<si64> BC4, C1_S2))>
  kgen.param.declare MHA_BCAST_CHECK4: !mo.chain = <apply(:(!mo.chain, !kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "error 2" : !kgen.string}, MHA_BCAST_CHECK2, MHA_BCAST_CHECK3)>

  mo.output
}


// Make sure we're handling cases with more than two asserts correctly We don't
// expect any guard to be materialized between the 1st and 2nd assert as no
// parameters are shared but we do expect one between the 2nd and 3rd asserts
// on the 'DENOM' param

// CHECK-LABEL @three_asserts
// CHECK: [[NEG1:%.*]] = kgen.param.constant: scalar<si64> = <-1>
// CHECK: [[DENOM:%.*]] = pop.add [[BS:%.*]], [[NEG1]]
// CHECK: [[CH0:%.*]] = mgp.assert {{.*}} "error 1"
// CHECK-NOT: mgp.guard
// CHECK: [[CH1:%.*]] = mgp.assert {{.*}} "error 2"
// CHECK: mgp.guard [[CH1]], [[DENOM]]
// CHECK: mgp.assert {{.*}} "error 3"
mo.graph @three_asserts<D0, D1, D2>()
{

  kgen.param.declare DENOM: scalar<si64> = <add(D0, -1)>
  kgen.param.declare COND1: scalar<bool> = <ne(:scalar<si64> DENOM, 0)>
  kgen.param.declare CH0: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "error 1" : !kgen.string}, COND1)>

  kgen.param.declare COND2: scalar<bool> = <ge(:scalar<si64> add(D1, 3), 2)>
  kgen.param.declare CH1: !mo.chain = <apply(:(!mo.chain, !kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "error 2" : !kgen.string}, CH0, COND2)>

  kgen.param.declare COND3: scalar<bool> = <ge(:scalar<si64> div(D2, DENOM), 4)>
  kgen.param.declare CH2: !mo.chain = <apply(:(!mo.chain, !kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "error 3" : !kgen.string}, CH1, COND3)>

  mo.output
}
