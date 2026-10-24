// RUN: kgen-opt -lower-hlcf-to-scf -allow-unregistered-dialect %s | FileCheck %s

// `lower-hlcf-to-scf` lowers HLCF structured control flow to SCF and leaves
// everything else, including `kgen.func` and `hlcf.return`, untouched. A
// `!kgen.scalar<bool>` condition is bridged to `i1` with an
// `unrealized_conversion_cast` for `lower-kgen-to-arith` to resolve.

// CHECK-LABEL: kgen.func @hlcf_for
// CHECK:         scf.for %{{.*}} = %[[LB:.*]] to %[[UB:.*]] step %[[STEP:.*]] {
// CHECK-NEXT:      "use"
// CHECK:         hlcf.return
kgen.func @hlcf_for(%lb: index, %ub: index, %step: index) {
  hlcf.for [%lb to %ub step %step slt add] (%iv = %lb : index) {
    "use"(%iv) : (index) -> ()
    hlcf.for.yield [induction_var (%iv : index)] [retvals ()] [iterargs ()]
  }
  hlcf.return
}

// A loop-carried `!kgen.scalar` value: the init and the yield are bridged
// to the converted iter-arg type, and the still-POP body op reads the block
// argument back through a cast.
// CHECK-LABEL: kgen.func @hlcf_for_iter_arg
// CHECK:         %[[INIT:.*]] = builtin.unrealized_conversion_cast %arg3 : !kgen.scalar<f32> to f32
// CHECK:         %[[R:.*]] = scf.for %{{.*}} = %arg0 to %arg1 step %arg2 iter_args(%[[ACC:.*]] = %[[INIT]]) -> (f32) {
// CHECK:           %[[ACC_POP:.*]] = builtin.unrealized_conversion_cast %[[ACC]] : f32 to !kgen.scalar<f32>
// CHECK:           %[[SUM:.*]] = pop.add %[[ACC_POP]], %arg3 : !kgen.scalar<f32>
// CHECK:           %[[SUM_F32:.*]] = builtin.unrealized_conversion_cast %[[SUM]] : !kgen.scalar<f32> to f32
// CHECK:           scf.yield %[[SUM_F32]] : f32
// CHECK:         %[[R_POP:.*]] = builtin.unrealized_conversion_cast %[[R]] : f32 to !kgen.scalar<f32>
// CHECK:         hlcf.return %[[R_POP]]
kgen.func @hlcf_for_iter_arg(%lb: index, %ub: index, %step: index, %init: !kgen.scalar<f32>) -> !kgen.scalar<f32> {
  %r = hlcf.for [%lb to %ub step %step slt add] (%iv = %lb : index, %acc = %init : !kgen.scalar<f32>) -> !kgen.scalar<f32> {
    %sum = pop.add %acc, %init : !kgen.scalar<f32>
    hlcf.for.yield [induction_var (%iv : index)] [retvals (%sum : !kgen.scalar<f32>)] [iterargs ()]
  }
  hlcf.return %r : !kgen.scalar<f32>
}

// Mojo's downcounting `range(n)` shape maps to an ascending loop whose body
// recovers the original induction value.
// CHECK-LABEL: kgen.func @hlcf_for_downcounting
// CHECK:         %[[LB:.*]] = arith.constant 0 : index
// CHECK:         %[[UB:.*]] = arith.constant 4 : index
// CHECK:         %[[STEP:.*]] = arith.constant 1 : index
// CHECK:         scf.for %[[K:.*]] = %[[LB]] to %[[UB]] step %[[STEP]] {
// CHECK:           %[[SUM:.*]] = arith.addi
// CHECK:           %[[IV:.*]] = arith.subi %[[SUM]], %[[K]]
// CHECK:           "use"(%[[IV]])
kgen.func @hlcf_for_downcounting() {
  %lb = index.constant 4
  %ub = index.constant 0
  %step = index.constant 1
  hlcf.for [%lb to %ub step %step sgt sub] (%iv = %lb : index) {
    "use"(%iv) : (index) -> ()
    hlcf.for.yield [induction_var (%iv : index)] [retvals ()] [iterargs ()]
  }
  hlcf.return
}

// CHECK-LABEL: kgen.func @hlcf_if
// CHECK:         %[[C:.*]] = builtin.unrealized_conversion_cast %{{.*}} : !kgen.scalar<bool> to i1
// CHECK:         scf.if %[[C]] {
// CHECK:         } else {
kgen.func @hlcf_if(%cond: !kgen.scalar<bool>) {
  hlcf.if %cond {
    "then"() : () -> ()
    hlcf.yield
  } else {
    "else"() : () -> ()
    hlcf.yield
  }
  hlcf.return
}

// CHECK-LABEL: kgen.func @hlcf_switch
// CHECK:         scf.index_switch
// CHECK:         case 0 {
// CHECK:         default {
kgen.func @hlcf_switch(%idx: index) {
  hlcf.switch %idx
  default {
    hlcf.yield
  }
  case 0 {
    hlcf.yield
  }
  hlcf.return
}

// CHECK-LABEL: kgen.func @hlcf_loop
// CHECK:         scf.while
// CHECK:           scf.condition
kgen.func @hlcf_loop() {
  hlcf.loop {
    "body"() : () -> ()
    hlcf.break
  }
  hlcf.return
}

// `hlcf.for` results are the retvals only; the other loop-carried values stay
// inside the `scf.for`.
// CHECK-LABEL: kgen.func @hlcf_for_extra_iterarg
// CHECK:         %[[R:.*]]:2 = scf.for
// CHECK:         hlcf.return %{{.*}} : !kgen.scalar<f32>
kgen.func @hlcf_for_extra_iterarg(%lb: index, %ub: index, %step: index, %init: !kgen.scalar<f32>) -> !kgen.scalar<f32> {
  %r = hlcf.for [%lb to %ub step %step slt add] (%iv = %lb : index, %acc = %init : !kgen.scalar<f32>, %other = %init : !kgen.scalar<f32>) -> !kgen.scalar<f32> {
    %sum = pop.add %acc, %other : !kgen.scalar<f32>
    hlcf.for.yield [induction_var (%iv : index)] [retvals (%sum : !kgen.scalar<f32>)] [iterargs (%other : !kgen.scalar<f32>)]
  }
  hlcf.return %r : !kgen.scalar<f32>
}

// A function ending in `hlcf.unreachable` is left for the function
// conversion, exactly as `hlcf.return` is.
// CHECK-LABEL: kgen.func @unreachable_terminator
// CHECK:         scf.for
// CHECK:         hlcf.unreachable
kgen.func @unreachable_terminator(%lb: index, %ub: index, %step: index) {
  hlcf.for [%lb to %ub step %step slt add] (%iv = %lb : index) {
    "use"(%iv) : (index) -> ()
    hlcf.for.yield [induction_var (%iv : index)] [retvals ()] [iterargs ()]
  }
  hlcf.unreachable
}

// `ForControlVariableType` also admits `!kgen.scalar<si*>` bounds, which
// convert to `iN`, so the induction recovery must not be index-only.
// CHECK-LABEL: kgen.func @hlcf_for_downcounting_si32
// CHECK:         arith.addi
// CHECK:         arith.subi
kgen.func @hlcf_for_downcounting_si32(%lb: !kgen.scalar<si32>, %ub: !kgen.scalar<si32>, %step: !kgen.scalar<si32>) {
  hlcf.for [%lb to %ub step %step : !kgen.scalar<si32> sgt sub] (%iv = %lb : !kgen.scalar<si32>) {
    "use"(%iv) : (!kgen.scalar<si32>) -> ()
    hlcf.for.yield [induction_var (%iv : !kgen.scalar<si32>)] [retvals ()] [iterargs ()]
  }
  hlcf.return
}

// An elif chain becomes nested `scf.if`s. Each condition region is inlined
// into the else arm above it, so a condition is still only evaluated once the
// earlier arms have been ruled out.
// CHECK-LABEL: kgen.func @elif_chain
// CHECK:         %[[C0:.*]] = builtin.unrealized_conversion_cast %arg0 : !kgen.scalar<bool> to i1
// CHECK:         %[[R:.*]] = scf.if %[[C0]] -> (index) {
// CHECK:           scf.yield %arg2 : index
// CHECK:         } else {
// CHECK:           %[[T:.*]] = pop.cmp lt
// CHECK:           %[[C1:.*]] = builtin.unrealized_conversion_cast %[[T]] : !kgen.scalar<bool> to i1
// CHECK:           %[[INNER:.*]] = scf.if %[[C1]] -> (index) {
// CHECK:             scf.yield %arg3 : index
// CHECK:           } else {
// CHECK:             scf.yield %arg4 : index
// CHECK:           }
// CHECK:           scf.yield %[[INNER]] : index
// CHECK:         hlcf.return %[[R]]
kgen.func @elif_chain(%c0: !kgen.scalar<bool>, %x: !kgen.scalar<si32>, %a: index, %b: index, %d: index) -> index {
  %r = hlcf.if %c0 -> index {
    hlcf.yield %a : index
  } else {
    %t = pop.cmp lt(%x, %x) : !kgen.scalar<si32>
    hlcf.if.elifcond.yield %t
  } then {
    hlcf.yield %b : index
  } else {
    hlcf.yield %d : index
  }
  hlcf.return %r : index
}

// A cond-yield passes its extra operands to both of its targets, so the
// paired `then` and the arm after it take the same values.
// CHECK-LABEL: kgen.func @elif_carried_values
// CHECK:         %[[V:.*]] = index.constant 7
// CHECK:         scf.if %{{.*}} -> (index) {
// CHECK:           scf.yield %[[V]] : index
// CHECK:         } else {
// CHECK:           scf.yield %[[V]] : index
kgen.func @elif_carried_values(%c0: !kgen.scalar<bool>, %c1: !kgen.scalar<bool>, %a: index) -> index {
  %r = hlcf.if %c0 -> index {
    hlcf.yield %a : index
  } else {
    %v = index.constant 7
    hlcf.if.elifcond.yield %c1, %v : index
  } then (%vt: index) {
    hlcf.yield %vt : index
  } else (%ve: index) {
    hlcf.yield %ve : index
  }
  hlcf.return %r : index
}
