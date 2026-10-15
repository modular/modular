// RUN: kgen-opt %s -simplify-cf -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: @remove_trivial_loop_0
kgen.func @remove_trivial_loop_0() -> () {
  "foo.op"() : () -> ()
  // CHECK-NOT: hlcf.loop
  hlcf.loop {
    "bar.op"() : () -> ()
    hlcf.break
  }
  // CHECK-NEXT: foo.op
  // CHECK-NEXT: bar.op
  // CHECK-NEXT: return
  hlcf.return
}

// CHECK-LABEL: @remove_trivial_loop_1
kgen.func @remove_trivial_loop_1(%arg0: index) -> index {
  // CHECK-NOT: hlcf.loop
  %r = hlcf.loop () -> index {
    hlcf.break %arg0: index
  }
  // CHECK-NEXT: return %arg0
  hlcf.return %r: index
}

// CHECK-LABEL: @remove_trivial_loop_2
// This loop shouldn't be removed as it has continue.
kgen.func @remove_trivial_loop_2(%cond: !kgen.scalar<bool>, %arg0: index) -> index {
  // CHECK-NEXT: hlcf.loop
  %r = hlcf.loop () -> index {
    hlcf.if %cond {
      hlcf.continue
    } else {
      hlcf.yield
    }
    hlcf.break %arg0: index
  }
  hlcf.return %r: index
}

// CHECK-LABEL: @remove_trivial_loop_3
// This loop shouldn't be removed as it has two breaks.
kgen.func @remove_trivial_loop_3(%cond: !kgen.scalar<bool>, %arg0: index, %arg1: index) -> index {
  // CHECK-NEXT: hlcf.loop
  %r = hlcf.loop () -> index {
    hlcf.if %cond {
      hlcf.break %arg1: index
    } else {
      hlcf.yield
    }
    hlcf.break %arg0: index
  }
  hlcf.return %r: index
}

// CHECK-LABEL: @remove_trivial_loop_4
// This loop can be removed as the return doesn't make the transformation incorrect.
kgen.func @remove_trivial_loop_4(%cond: !kgen.scalar<bool>, %arg0: index, %arg1: index) -> index {
  // CHECK-NOT: hlcf.loop
  %r = hlcf.loop () -> index {
    // CHECK-NEXT: hlcf.if
    hlcf.if %cond {
      // CHECK-NEXT: return %arg2
      hlcf.return %arg1: index
    } else {
      hlcf.yield
    }
    // CHECK-NOT: break
    hlcf.break %arg0: index
  }
  // CHECK: return %arg1
  hlcf.return %r: index
}

// CHECK-LABEL: @remove_trivial_loop_5
// Here we can remove the outer loop despite the presence of break and continue
// in the inner loop (which can't be removed).
kgen.func @remove_trivial_loop_5(%cond: !kgen.scalar<bool>, %arg0: index, %arg1: index) -> index {
  // CHECK-COUNT-1: hlcf.loop
  %r = hlcf.loop () -> index {
    %t = hlcf.loop () -> index {
      // CHECK-NEXT: hlcf.if
      hlcf.if %cond {
        // CHECK-NEXT: continue
        hlcf.continue
      } else {
        hlcf.yield
      }
      // CHECK: break %arg1
      hlcf.break %arg0: index
    }
    // CHECK-NOT: break
    hlcf.break %t: index
  }
  // CHECK: return %0
  hlcf.return %r: index
}

// CHECK-LABEL: @remove_trivial_loop_6
// This loop can be removed even though the break is to the outer loop.
kgen.func @remove_trivial_loop_6(%cond: !kgen.scalar<bool>, %arg0: index, %arg1: index) -> index {
  // TODO: We should be able to delete both loops here, but we only manage to
  // delete the inner one now.

  // CHECK-NEXT: hlcf.loop "outer"
  hlcf.loop "outer" {
    // CHECK-NOT: hlcf.loop
    hlcf.loop {
      // CHECK-NEXT: bar.op
      "bar.op"() : () -> ()
      // CHECK-NEXT: hlcf.break "outer"
      hlcf.break "outer"
    }
    // CHECK-NOT: foo.op
    "foo.op"() : () -> ()
    hlcf.break
  }
  hlcf.return %arg0: index
}

// CHECK-LABEL: @remove_trivial_loop_7
kgen.func @remove_trivial_loop_7() {
  // CHECK-NEXT: hlcf.loop "outer"
  hlcf.loop "outer" {
    // CHECK-NOT: hlcf.loop
    hlcf.loop {
      // CHECK-NEXT: bar.op
      "bar.op"() : () -> ()
      // CHECK-NEXT: hlcf.continue
      hlcf.continue "outer"
    }
    // CHECK-NOT: foo.op
    "foo.op"() : () -> ()
    hlcf.break
  }
  hlcf.return
}

// CHECK-LABEL: @remove_trivial_loop_8
// The inner loop can be removed, the outer cannot.
kgen.func @remove_trivial_loop_8(%cond: !kgen.scalar<bool>) {
  // CHECK-NEXT: hlcf.loop
  hlcf.loop {
    // CHECK-NEXT: hlcf.if
    hlcf.if %cond {
      hlcf.continue
    } else {
      // CHECK-NOT: hlcf.loop
      hlcf.loop {
        hlcf.break
      }
      hlcf.yield
    }
    hlcf.break
  }
  hlcf.return
}

// CHECK-LABEL: @remove_trivial_loop_9
// Both loops can be removed.
kgen.func @remove_trivial_loop_9(%cond: !kgen.scalar<bool>) {
  // CHECK-NEXT: return
  hlcf.loop {
    hlcf.loop {
      hlcf.break
    }
    hlcf.break
  }
  hlcf.return
}

// CHECK-LABEL: @remove_trivial_loop_10
kgen.func @remove_trivial_loop_10(%cond: !kgen.scalar<bool>) {
  // FIXME: Only one loop can be removed.
  // CHECK-NEXT: hlcf.loop
  hlcf.loop {
    hlcf.loop {
      // CHECK-NEXT: return
      hlcf.return
    }
    // CHECK-NOT: foo.op
    "foo.op"() : () -> ()
    hlcf.break
  }
  hlcf.return
}

// CHECK-LABEL: @remove_trivial_loop_11
// Only the outer loop can be removed.
kgen.func @remove_trivial_loop_11(%cond: !kgen.scalar<bool>) {
  hlcf.loop () {
    // CHECK-NEXT: hlcf.loop
    hlcf.loop () {
      // CHECK-NEXT: hlcf.if
      hlcf.if %cond {
        hlcf.continue
      } else {
        hlcf.return
      }
      hlcf.break
    }
    hlcf.break
  }
  hlcf.return
}

// CHECK-LABEL: @remove_trivial_loop_12
// Only the outer loop can be removed.
kgen.func @remove_trivial_loop_12(%cond: !kgen.scalar<bool>) {
  hlcf.loop () {
    // CHECK-NEXT: hlcf.loop
    hlcf.loop () {
      // CHECK-NEXT: hlcf.if
      hlcf.if %cond {
        hlcf.yield
      } else {
        hlcf.break
      }
      hlcf.return
    }
    hlcf.break
  }
  hlcf.return
}

// CHECK-LABEL: @two_loop_erase
// COM: Ensure this doesn't result in use-after-free.
kgen.func @two_loop_erase() {
  // CHECK-NEXT: return
  hlcf.loop {
    hlcf.return
  }
  hlcf.loop {
    hlcf.return
  }
  hlcf.return
}

// CHECK-LABEL: @erase_trivial_try
kgen.func @erase_trivial_try() {
  lit.try {
    // CHECK-NEXT: foo.op
    "foo.op"() : () -> ()
    lit.try.yield
  } except (%e: index) {
    hlcf.unreachable
  } else {
    // CHECK-NEXT: bar.op
    "bar.op"() : () -> ()
    lit.try.yield
  }
  // CHECK-NEXT: return
  hlcf.return
}

// CHECK-LABEL: @raise_in_try
kgen.func @raise_in_try(%arg0: index) {
  // CHECK-NEXT: lit.try
  lit.try "try0" {
    lit.try.raise "try0" %arg0 : index
  } except (%e: index) {
    lit.try.yield
  } else {
    hlcf.unreachable
  }
  hlcf.return
}


// CHECK-LABEL: @nested_raise
kgen.func @nested_raise(%arg0: index) {
  // CHECK-NEXT: lit.try
  lit.try "try0" {
    hlcf.loop {
      lit.try.raise "try0" %arg0 : index
    }
    lit.try.yield
  } except (%e: index) {
    lit.try.yield
  } else {
    lit.try.yield
  }
  hlcf.return
}

// CHECK-LABEL: @raise_in_else
// COM: Make sure the right contextual try is selected.
kgen.func @raise_in_else(%arg0: index) {
  // CHECK-NEXT: lit.try "try0" {
  lit.try "try0" {
    // CHECK-NEXT: foo.op
    "foo.op"() : () -> ()
    lit.try "try1" {
      // CHECK-NEXT: bar.op
      "bar.op"() : () -> ()
      lit.try.yield
    } except (%arg1: index) {
      hlcf.unreachable
    } else {
      // CHECK-NEXT: baz.op
      "baz.op"() : () -> ()
      // CHECK-NEXT: lit.try.raise "try0" %arg0
      lit.try.raise "try0" %arg0 : index
    }
    lit.try.yield
  // CHECK-NEXT: except
  } except (%arg2: index) {
    // CHECK-NEXT: lit.try.yield
    lit.try.yield
  } else {
    lit.try.yield
  }
  hlcf.return
}

// CHECK-LABEL: @return_in_try
kgen.func @return_in_try() {
  lit.try {
    // CHECK-NEXT: return
    hlcf.return
  } except (%e: index) {
    hlcf.unreachable
  } else {
    lit.try.yield
  }
  // CHECK-NOT: foo.op
  "foo.op"() : () -> ()
  hlcf.return
}

// CHECK-LABEL: @return_in_else
kgen.func @return_in_else() {
  lit.try {
    lit.try.yield
  } except (%e: index) {
    hlcf.unreachable
  } else {
    // CHECK-NEXT: return
    hlcf.return
  }
  // CHECK-NOT: foo.op
  "foo.op"() : () -> ()
  hlcf.return
}

// CHECK-LABEL: @try_arg_passing
kgen.func @try_arg_passing(%arg0: index, %arg1: si32) -> (index, si32) {
  lit.try {
    lit.try.yield %arg0, %arg1 : index, si32
  } except (%e: index) {
    hlcf.unreachable
  } else {
  ^bb0(%arg2: index, %arg3: si32):
    // CHECK-NEXT: return %arg0, %arg1
    hlcf.return %arg2, %arg3 : index, si32
  }
  hlcf.unreachable
}

// CHECK-LABEL: @try_result_passing
kgen.func @try_result_passing(%arg0: index, %arg1: si32) -> (index, si32) {
  %0:2 = lit.try -> index, si32 {
    lit.try.yield %arg0, %arg1 : index, si32
  } except (%e: index) {
    hlcf.unreachable
  } else {
  ^bb0(%arg2: index, %arg3: si32):
    lit.try.yield %arg2, %arg3 : index, si32
  }
  // CHECK-NEXT: return %arg0, %arg1
  hlcf.return %0#0, %0#1 : index, si32
}

// CHECK-LABEL: @loop_break
kgen.func @loop_break(%arg0: index) -> index {
  // CHECK-NEXT: return %arg0 : index
  %0 = hlcf.loop (%arg1 = %arg0 : index) -> index {
    hlcf.break %arg1 : index
  }
  hlcf.return %0 : index
}

// A `then` that returns implies the condition for everything after the if,
// including uses nested in later regions; uses before and inside the if stay.
// CHECK-LABEL: @implied_false_after_return
kgen.func @implied_false_after_return(%cond: !kgen.scalar<bool>, %other: !kgen.scalar<bool>) {
  // CHECK:      "foo.before"(%arg0)
  // CHECK-NEXT: %[[F:.*]] = kgen.param.constant: scalar<bool> = <false>
  // CHECK-NEXT: hlcf.if %arg0 {
  // CHECK-NEXT:   "foo.inside"(%arg0)
  // CHECK:      hlcf.if %[[F]] {
  // CHECK:      hlcf.if %arg1 {
  // CHECK-NEXT:   "foo.nested"(%[[F]])
  "foo.before"(%cond) : (!kgen.scalar<bool>) -> ()
  hlcf.if %cond {
    "foo.inside"(%cond) : (!kgen.scalar<bool>) -> ()
    hlcf.return
  } else {
    hlcf.yield
  }
  hlcf.if %cond {
    "foo.after"() : () -> ()
    hlcf.yield
  } else {
    hlcf.yield
  }
  hlcf.if %other {
    "foo.nested"(%cond) : (!kgen.scalar<bool>) -> ()
    hlcf.yield
  } else {
    hlcf.yield
  }
  hlcf.return
}

// A condition that is already constant is left alone.
// CHECK-LABEL: @constant_condition
kgen.func @constant_condition() {
  // CHECK:      %[[F:.*]] = kgen.param.constant: scalar<bool> = <false>
  // CHECK-NOT:  kgen.param.constant
  // CHECK:      "foo.after"(%[[F]])
  %f = kgen.param.constant: scalar<bool> = <false>
  hlcf.if %f {
    hlcf.return
  } else {
    hlcf.yield
  }
  "foo.after"(%f) : (!kgen.scalar<bool>) -> ()
  hlcf.return
}

// An `else` that cannot continue implies the condition is true.
// CHECK-LABEL: @implied_true_after_unreachable
kgen.func @implied_true_after_unreachable(%cond: !kgen.scalar<bool>) {
  // CHECK:      %[[T:.*]] = kgen.param.constant: scalar<bool> = <true>
  // CHECK-NEXT: hlcf.if %arg0 {
  // CHECK:      "foo.after"(%[[T]])
  hlcf.if %cond {
    hlcf.yield
  } else {
    hlcf.unreachable
  }
  "foo.after"(%cond) : (!kgen.scalar<bool>) -> ()
  hlcf.return
}

// With an elif, control past the if may come from either region, so an
// exiting `else` region implies nothing; a `then` region that falls through
// implies nothing either.
// CHECK-LABEL: @no_implied_condition
kgen.func @no_implied_condition(%cond1: !kgen.scalar<bool>, %cond2: !kgen.scalar<bool>) {
  // CHECK-NOT: kgen.param.constant
  // CHECK: "foo.after"(%arg0, %arg1)
  hlcf.if %cond1 {
    hlcf.yield
  } else {
    hlcf.if.elifcond.yield %cond2
  } then {
    hlcf.yield
  } else {
    hlcf.unreachable
  }
  hlcf.if %cond2 {
    "foo.then"() : () -> ()
    hlcf.yield
  } else {
    "foo.else"() : () -> ()
    hlcf.yield
  }
  "foo.after"(%cond1, %cond2) : (!kgen.scalar<bool>, !kgen.scalar<bool>) -> ()
  hlcf.return
}

// A nested if implies its condition only within its own block: a later use in
// the enclosing block can be reached without the nested region having run.
// CHECK-LABEL: @nested_if_implies_only_its_block
kgen.func @nested_if_implies_only_its_block(%cond: !kgen.scalar<bool>, %outer: !kgen.scalar<bool>) {
  // CHECK:      hlcf.if %arg1 {
  // CHECK-NEXT:   %[[F:.*]] = kgen.param.constant: scalar<bool> = <false>
  // CHECK-NEXT:   hlcf.if %arg0 {
  // CHECK:        "foo.inner_after"(%[[F]])
  // CHECK:      hlcf.if %arg0 {
  // CHECK-NEXT:   "foo.outer_after"
  // CHECK:      "foo.end"(%arg0)
  hlcf.if %outer {
    hlcf.if %cond {
      hlcf.return
    } else {
      hlcf.yield
    }
    "foo.inner_after"(%cond) : (!kgen.scalar<bool>) -> ()
    hlcf.yield
  } else {
    hlcf.yield
  }
  hlcf.if %cond {
    "foo.outer_after"() : () -> ()
    hlcf.yield
  } else {
    hlcf.yield
  }
  "foo.end"(%cond) : (!kgen.scalar<bool>) -> ()
  hlcf.return
}

// A top-level `if` whose `then` region exits implies the condition in every
// later op, however deeply the use is nested.
// CHECK-LABEL: @top_level_if_implies_nested_uses
kgen.func @top_level_if_implies_nested_uses(%cond: !kgen.scalar<bool>, %outer: !kgen.scalar<bool>) {
  // CHECK:      %[[F:.*]] = kgen.param.constant: scalar<bool> = <false>
  // CHECK-NEXT: hlcf.if %arg0 {
  // CHECK:      hlcf.if %arg1 {
  // CHECK-NEXT:   hlcf.if %[[F]] {
  // CHECK:        "foo.inner_after"(%[[F]])
  // CHECK:      "foo.end"(%[[F]])
  hlcf.if %cond {
    hlcf.return
  } else {
    hlcf.yield
  }
  hlcf.if %outer {
    hlcf.if %cond {
      "foo.nested"() : () -> ()
      hlcf.yield
    } else {
      hlcf.yield
    }
    "foo.inner_after"(%cond) : (!kgen.scalar<bool>) -> ()
    hlcf.yield
  } else {
    hlcf.yield
  }
  "foo.end"(%cond) : (!kgen.scalar<bool>) -> ()
  hlcf.return
}

// A `then` region that continues does not fall through either, so the
// condition is false for the rest of the loop body.
// CHECK-LABEL: @implied_false_after_continue
kgen.func @implied_false_after_continue(%cond: !kgen.scalar<bool>) {
  // CHECK:      %[[F:.*]] = kgen.param.constant: scalar<bool> = <false>
  // CHECK:      hlcf.if %arg0 {
  // CHECK-NEXT:   hlcf.continue
  // CHECK:      hlcf.if %[[F]] {
  // CHECK:      "foo.after"(%[[F]])
  hlcf.loop {
    hlcf.if %cond {
      hlcf.continue
    } else {
      hlcf.yield
    }
    hlcf.if %cond {
      "foo.in_if"() : () -> ()
      hlcf.yield
    } else {
      hlcf.yield
    }
    "foo.after"(%cond) : (!kgen.scalar<bool>) -> ()
    hlcf.break
  }
  hlcf.return
}

// CHECK-LABEL: @implied_false_after_break
kgen.func @implied_false_after_break(%cond: !kgen.scalar<bool>) {
  // CHECK:      %[[F:.*]] = kgen.param.constant: scalar<bool> = <false>
  // CHECK:      hlcf.if %arg0 {
  // CHECK-NEXT:   hlcf.break
  // CHECK:      "foo.after"(%[[F]])
  hlcf.loop {
    hlcf.if %cond {
      hlcf.break
    } else {
      hlcf.yield
    }
    "foo.after"(%cond) : (!kgen.scalar<bool>) -> ()
    hlcf.break
  }
  hlcf.return
}

// CHECK-LABEL: @implied_true_after_continue_in_else
kgen.func @implied_true_after_continue_in_else(%cond: !kgen.scalar<bool>) {
  // CHECK:      %[[T:.*]] = kgen.param.constant: scalar<bool> = <true>
  // CHECK:      "foo.after"(%[[T]])
  hlcf.loop {
    hlcf.if %cond {
      hlcf.yield
    } else {
      hlcf.continue
    }
    "foo.after"(%cond) : (!kgen.scalar<bool>) -> ()
    hlcf.break
  }
  hlcf.return
}

// Only the terminators that are known to leave the block imply the condition.
// A match arm's terminators are not among them, so nothing is folded.
// CHECK-LABEL: @no_implied_condition_match_terminators
kgen.func @no_implied_condition_match_terminators(%cond: !kgen.scalar<bool>) {
  // CHECK-NOT:  kgen.param.constant
  // CHECK:      "foo.after"(%arg0)
  hlcf.match {
    hlcf.if %cond {
      hlcf.match.complete
    } else {
      hlcf.match.next
    }
    "foo.after"(%cond) : (!kgen.scalar<bool>) -> ()
    hlcf.unreachable
  }
  case {
    "foo.other"() : () -> ()
    hlcf.match.complete
  }
  else {
    hlcf.yield
  }
  hlcf.return
}
