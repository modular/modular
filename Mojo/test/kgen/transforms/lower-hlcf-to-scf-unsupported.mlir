// RUN: kgen-opt -lower-hlcf-to-scf -allow-unregistered-dialect -split-input-file -verify-diagnostics %s

// `scf` has no branch that leaves the region it sits in, and `scf.for` is
// half-open, so a shape needing either is reported rather than lowered to
// something that silently computes the wrong thing.

// An inclusive upper bound would lose the `iv == ub` iteration.
kgen.func @inclusive_bound(%lb: index, %ub: index, %step: index) {
  // expected-error @below {{failed to legalize operation 'hlcf.for'}}
  hlcf.for [%lb to %ub step %step sle add] (%iv = %lb : index) {
    "use"(%iv) : (index) -> ()
    hlcf.for.yield [induction_var (%iv : index)] [retvals ()] [iterargs ()]
  }
  hlcf.return
}

// -----

// The induction variable must be the loop's own `step` advance; `scf.for`
// drives it from `step`, so any other value changes the trip count.
kgen.func @induction_var_not_step_advanced(%lb: index, %ub: index, %step: index) {
  // expected-error @below {{failed to legalize operation 'hlcf.for'}}
  hlcf.for [%lb to %ub step %step slt add] (%iv = %lb : index) {
    %two = index.constant 2
    %skip = index.add %iv, %two
    "use"(%iv) : (index) -> ()
    hlcf.for.yield [induction_var (%skip : index)] [retvals ()] [iterargs ()]
  }
  hlcf.return
}

// -----

// A `return` inside a region would become the terminator of the `scf.if`.
kgen.func @nested_return(%c: !kgen.scalar<bool>) {
  // expected-error @below {{failed to legalize operation 'hlcf.if'}}
  hlcf.if %c {
    hlcf.return
  } else {
    hlcf.yield
  }
  hlcf.return
}

// -----

// A `break` inside an `hlcf.if` would have to leave both that region and the
// loop body, so the pass reports the shape up front.
kgen.func @exit_below_loop_body(%c: !kgen.scalar<bool>) {
  // expected-error @below {{only when its `continue`/`break` terminates the loop body}}
  hlcf.loop {
    hlcf.if %c {
      // expected-note @below {{`break` nested in `hlcf.if` here}}
      hlcf.break
    } else {
      hlcf.yield
    }
    hlcf.break
  }
  hlcf.return
}
