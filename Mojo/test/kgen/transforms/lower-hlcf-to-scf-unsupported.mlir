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

// A select that stops advancing anywhere but the bound changes the trip count.
kgen.func @induction_var_stops_before_bound() {
  %lb = kgen.param.constant = <0>
  %ub = kgen.param.constant = <4>
  %step = kgen.param.constant = <1>
  %two_s = kgen.param.constant: scalar<index> = <2>
  %one_s = kgen.param.constant: scalar<index> = <1>
  // expected-error @below {{failed to legalize operation 'hlcf.for'}}
  hlcf.for [%lb to %ub step %step slt add] (%iv = %lb : index) {
    %cur = pop.cast_from_builtin %iv : index to !kgen.scalar<index>
    %inc = pop.add %cur, %one_s : !kgen.scalar<index>
    %done = pop.cmp eq(%cur, %two_s) : <1, index>
    %next = pop.select %done, %cur, %inc : !kgen.scalar<index>
    "use"(%iv) : (index) -> ()
    %next_idx = pop.cast_to_builtin %next : !kgen.scalar<index> to index
    hlcf.for.yield [induction_var (%next_idx : index)] [retvals ()] [iterargs ()]
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

// -----

// A `while` loop whose body breaks again has a second exit, which the
// `scf.while` split cannot express.
kgen.func @while_loop_second_exit(%c: !kgen.scalar<bool>, %d: !kgen.scalar<bool>) {
  // expected-error @below {{only when its `continue`/`break` terminates the loop body}}
  hlcf.loop {
    hlcf.if %c {
      hlcf.yield
    } else {
      hlcf.break
    }
    hlcf.if %d {
      // expected-note @below {{`break` nested in `hlcf.if` here}}
      hlcf.break
    } else {
      hlcf.yield
    }
    hlcf.continue
  }
  hlcf.return
}

// -----

// A labeled `break` in a nested loop exits the outer loop, so it is a second
// exit of the outer `while` loop too.
kgen.func @while_loop_break_from_inner_loop(%c: !kgen.scalar<bool>, %d: !kgen.scalar<bool>) {
  // expected-error @below {{only when its `continue`/`break` terminates the loop body}}
  hlcf.loop "outer" {
    hlcf.if %c {
      hlcf.yield
    } else {
      hlcf.break
    }
    hlcf.loop "inner" {
      hlcf.if %d {
        // expected-note @below {{`break` nested in `hlcf.if` here}}
        hlcf.break "outer"
      } else {
        hlcf.yield
      }
      hlcf.break
    }
    hlcf.continue
  }
  hlcf.return
}

// -----

// It is a second exit wherever the nested loop sits, before the exit check
// too.
kgen.func @break_from_inner_loop_before_check(%c: !kgen.scalar<bool>, %d: !kgen.scalar<bool>) {
  // expected-error @below {{only when its `continue`/`break` terminates the loop body}}
  hlcf.loop "outer" {
    hlcf.loop "inner" {
      hlcf.if %d {
        // expected-note @below {{`break` nested in `hlcf.if` here}}
        hlcf.break "outer"
      } else {
        hlcf.yield
      }
      hlcf.break
    }
    hlcf.if %c {
      hlcf.yield
    } else {
      hlcf.break
    }
    hlcf.continue
  }
  hlcf.return
}

// -----

// The same holds for a labeled `break` that ends the nested loop's body.
kgen.func @break_ends_inner_loop_body() {
  // expected-error @below {{only when its `continue`/`break` terminates the loop body}}
  hlcf.loop "outer" {
    hlcf.loop "inner" {
      // expected-note @below {{`break` nested in `hlcf.loop` here}}
      hlcf.break "outer"
    }
    hlcf.continue
  }
  hlcf.return
}
