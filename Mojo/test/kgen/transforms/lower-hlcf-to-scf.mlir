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

// A `while j < n:` that `raise-for-loops` left alone splits at its exit check:
// the ops up to the check become the before region and the rest the after
// region, which reads the loop state `scf.condition` forwards.
// CHECK-LABEL: kgen.func @while_loop
// CHECK:         scf.while (%[[J:.*]] = %arg0) : (index) -> index {
// CHECK:           %[[CMP:.*]] = index.cmp slt(%[[J]], %arg1)
// CHECK:           %[[C:.*]] = pop.cast_from_builtin %[[CMP]] : i1 to !kgen.scalar<bool>
// CHECK:           %[[C_I1:.*]] = builtin.unrealized_conversion_cast %[[C]] : !kgen.scalar<bool> to i1
// CHECK:           scf.condition(%[[C_I1]]) %[[J]] : index
// CHECK:         } do {
// CHECK:         ^bb0(%[[JA:.*]]: index):
// CHECK-NEXT:      "use"(%[[JA]])
// CHECK-NEXT:      %[[NEXT:.*]] = index.add %[[JA]], %arg2
// CHECK-NEXT:      scf.yield %[[NEXT]] : index
// CHECK:         hlcf.return
kgen.func @while_loop(%init: index, %n: index, %step: index) {
  hlcf.loop (%j = %init : index) {
    %cmp = index.cmp slt(%j, %n)
    %c = pop.cast_from_builtin %cmp : i1 to !kgen.scalar<bool>
    hlcf.if %c {
      hlcf.yield
    } else {
      hlcf.break
    }
    "use"(%j) : (index) -> ()
    %next = index.add %j, %step
    hlcf.continue %next : index
  }
  hlcf.return
}

// A pure value of the before region that the after region also reads is
// recomputed there rather than forwarded, so `scf.condition` carries only the
// loop state.
// CHECK-LABEL: kgen.func @while_loop_remat
// CHECK:         %[[TWO:.*]] = index.constant 2
// CHECK:         scf.while (%[[J:.*]] = %arg0) : (index) -> index {
// CHECK:           %[[J2:.*]] = index.mul %[[J]], %[[TWO]]
// CHECK:           index.cmp slt(%[[J2]], %arg1)
// CHECK:           scf.condition(%{{.*}}) %[[J]] : index
// CHECK:         } do {
// CHECK:         ^bb0(%[[JA:.*]]: index):
// CHECK-NEXT:      %[[J2A:.*]] = index.mul %[[JA]], %[[TWO]]
// CHECK-NEXT:      "use"(%[[J2A]])
kgen.func @while_loop_remat(%init: index, %n: index) {
  %two = index.constant 2
  %one = index.constant 1
  hlcf.loop (%j = %init : index) {
    %j2 = index.mul %j, %two
    %cmp = index.cmp slt(%j2, %n)
    %c = pop.cast_from_builtin %cmp : i1 to !kgen.scalar<bool>
    hlcf.if %c {
      hlcf.yield
    } else {
      hlcf.break
    }
    "use"(%j2) : (index) -> ()
    %next = index.add %j, %one
    hlcf.continue %next : index
  }
  hlcf.return
}

// A value of the before region that cannot be recomputed is forwarded.
// CHECK-LABEL: kgen.func @while_loop_impure_head
// CHECK:         scf.while
// CHECK:           %[[V:.*]] = "read"()
// CHECK:           scf.condition(%{{.*}}) {{.*}}%[[V]]{{.*}} : index, index
// CHECK:         } do {
// CHECK:         ^bb0(%{{.*}}: index, %{{.*}}: index):
kgen.func @while_loop_impure_head(%init: index, %c: !kgen.scalar<bool>) {
  %one = index.constant 1
  hlcf.loop (%j = %init : index) {
    %v = "read"() : () -> index
    hlcf.if %c {
      hlcf.yield
    } else {
      hlcf.break
    }
    "use"(%v) : (index) -> ()
    %next = index.add %j, %one
    hlcf.continue %next : index
  }
  hlcf.return
}

// A value live after the loop leaves through the exit check's `break`, so it
// is forwarded too and becomes the `scf.while` result.
// CHECK-LABEL: kgen.func @while_loop_live_out
// CHECK:         %[[R:.*]] = scf.while (%[[J:.*]] = %arg0) : (index) -> index {
// CHECK:           scf.condition(%{{.*}}) %[[J]] : index
// CHECK:         hlcf.return %[[R]] : index
kgen.func @while_loop_live_out(%init: index, %n: index) -> index {
  %one = index.constant 1
  %r = hlcf.loop (%j = %init : index) -> index {
    %cmp = index.cmp slt(%j, %n)
    %c = pop.cast_from_builtin %cmp : i1 to !kgen.scalar<bool>
    hlcf.if %c {
      hlcf.yield
    } else {
      hlcf.break %j : index
    }
    %next = index.add %j, %one
    hlcf.continue %next : index
  }
  hlcf.return %r : index
}

// A `while` loop nested in another lowers on its own. Its uses of the outer
// loop's state read the value the outer `scf.condition` forwards.
// CHECK-LABEL: kgen.func @nested_while_loops
// CHECK:         scf.while (%[[I:.*]] = %arg0) : (index) -> index {
// CHECK:           scf.condition(%{{.*}}) %[[I]] : index
// CHECK:         } do {
// CHECK:         ^bb0(%[[IA:.*]]: index):
// CHECK:           scf.while (%[[J:.*]] = %arg0) : (index) -> index {
// CHECK:             index.cmp slt(%[[J]], %[[IA]])
// CHECK:             scf.condition(%{{.*}}) %[[J]] : index
// CHECK:           } do {
// CHECK:           ^bb0(%[[JA:.*]]: index):
// CHECK:             "use"(%[[IA]], %[[JA]])
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           scf.yield
// CHECK-NOT:     hlcf.loop
// CHECK:         hlcf.return
kgen.func @nested_while_loops(%init: index, %n: index) {
  %one = index.constant 1
  hlcf.loop "outer" (%i = %init : index) {
    %ci = index.cmp slt(%i, %n)
    %c = pop.cast_from_builtin %ci : i1 to !kgen.scalar<bool>
    hlcf.if %c {
      hlcf.yield
    } else {
      hlcf.break
    }
    hlcf.loop "inner" (%j = %init : index) {
      %cj = index.cmp slt(%j, %i)
      %d = pop.cast_from_builtin %cj : i1 to !kgen.scalar<bool>
      hlcf.if %d {
        hlcf.yield
      } else {
        hlcf.break
      }
      "use"(%i, %j) : (index, index) -> ()
      %jn = index.add %j, %one
      hlcf.continue %jn : index
    }
    %in = index.add %i, %one
    hlcf.continue %in : index
  }
  hlcf.return
}

// A loop before the exit check lands in the before region, so it still runs
// on every iteration ahead of the check. It has a region, so its result is
// forwarded through `scf.condition` rather than recomputed.
// CHECK-LABEL: kgen.func @while_loop_inner_loop_before_check
// CHECK:         scf.while (%[[I:.*]] = %arg0) : (index) -> (index, index) {
// CHECK:           %[[R:.*]] = scf.while (%[[J:.*]] = %arg0) : (index) -> index {
// CHECK:             index.cmp slt(%[[J]], %[[I]])
// CHECK:             scf.condition(%{{.*}}) %[[J]] : index
// CHECK:           } do {
// CHECK:           }
// CHECK:           index.cmp slt(%[[I]], %arg1)
// CHECK:           scf.condition(%{{.*}}) %[[I]], %[[R]] : index, index
// CHECK:         } do {
// CHECK:         ^bb0(%{{.*}}: index, %[[RA:.*]]: index):
// CHECK-NOT:       scf.while
// CHECK:           "use"(%[[RA]])
// CHECK:           scf.yield
// CHECK-NOT:     hlcf.loop
// CHECK:         hlcf.return
kgen.func @while_loop_inner_loop_before_check(%init: index, %n: index) {
  %one = index.constant 1
  hlcf.loop "outer" (%i = %init : index) {
    %r = hlcf.loop "inner" (%j = %init : index) -> index {
      %cj = index.cmp slt(%j, %i)
      %d = pop.cast_from_builtin %cj : i1 to !kgen.scalar<bool>
      hlcf.if %d {
        hlcf.yield
      } else {
        hlcf.break %j : index
      }
      %jn = index.add %j, %one
      hlcf.continue %jn : index
    }
    %ci = index.cmp slt(%i, %n)
    %c = pop.cast_from_builtin %ci : i1 to !kgen.scalar<bool>
    hlcf.if %c {
      hlcf.yield
    } else {
      hlcf.break
    }
    "use"(%r) : (index) -> ()
    %in = index.add %i, %one
    hlcf.continue %in : index
  }
  hlcf.return
}

// A loop-carried value whose type converts is bridged into both regions, and
// back to its HLCF type for the ops that read it.
// CHECK-LABEL: kgen.func @while_loop_converted_type
// CHECK:         %[[INIT:.*]] = builtin.unrealized_conversion_cast %arg0 : !kgen.scalar<si32> to i32
// CHECK:         %[[R:.*]] = scf.while (%[[X:.*]] = %[[INIT]]) : (i32) -> i32 {
// CHECK:           scf.condition(%{{.*}}) %[[X]] : i32
// CHECK:         } do {
// CHECK:         ^bb0(%[[XA:.*]]: i32):
// CHECK-NEXT:      %[[XS:.*]] = builtin.unrealized_conversion_cast %[[XA]] : i32 to !kgen.scalar<si32>
// CHECK-NEXT:      "use"(%[[XS]])
// CHECK-NEXT:      %[[Y:.*]] = "next"(%[[XS]])
// CHECK-NEXT:      %[[YI:.*]] = builtin.unrealized_conversion_cast %[[Y]] : !kgen.scalar<si32> to i32
// CHECK-NEXT:      scf.yield %[[YI]] : i32
// CHECK:         %[[RS:.*]] = builtin.unrealized_conversion_cast %[[R]] : i32 to !kgen.scalar<si32>
// CHECK:         hlcf.return %[[RS]] : !kgen.scalar<si32>
kgen.func @while_loop_converted_type(%init: !kgen.scalar<si32>, %c: !kgen.scalar<bool>) -> !kgen.scalar<si32> {
  %r = hlcf.loop (%x = %init : !kgen.scalar<si32>) -> !kgen.scalar<si32> {
    hlcf.if %c {
      hlcf.yield
    } else {
      hlcf.break %x : !kgen.scalar<si32>
    }
    "use"(%x) : (!kgen.scalar<si32>) -> ()
    %y = "next"(%x) : (!kgen.scalar<si32>) -> !kgen.scalar<si32>
    hlcf.continue %y : !kgen.scalar<si32>
  }
  hlcf.return %r : !kgen.scalar<si32>
}
