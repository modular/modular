// RUN: kgen-opt %s -allow-unregistered-dialect -lower-semantic-cf -verify-diagnostics

lit.fn @dead_code() {
  lit.return
  // expected-warning @below {{unreachable code after return statement}}
  "do.something"() : () -> ()
  lit.end_fn
}

lit.fn @no_return_result() -> i32 {
  // expected-error @below {{return expected at end of function with results}}
  lit.end_fn
}

lit.fn @bad_break() {
  // expected-error @below {{'break' is not inside a loop}}
  lit.break
  lit.end_fn
}

lit.fn @bad_continue() {
  // expected-error @below {{'continue' is not inside a loop}}
  lit.continue
  lit.end_fn
}

// break in an 'else' is an error unless in a nested loop.
lit.fn @bad_break_2(%arg0: !kgen.scalar<bool>) {
  // CHECK: hlcf.loop "_loop_0"
  lit.loop {
    hlcf.elif %arg0 {
      hlcf.yield
    } else {
      lit.loop.break.else
    }
    lit.loop.continue
  } else {
    lit.break // expected-error {{'break' is not inside a loop}}
    lit.loop.yield
  }

  lit.return
  lit.end_fn
}

lit.fn @unresolved_fn() {
  lit.end_fn unresolved // disables error.
}

lit.fn @resolved_fn() {
  lit.end_fn // expected-error {{return expected at end of function with results}}
}

// First case always completes, so the second case and else are dead. The
// match itself still falls through via match.complete.
lit.fn @unreachable_match_case_and_else() {
  hlcf.match {
    hlcf.match.complete
  }
  case {
    // expected-warning @below {{unreachable code after match case that always completes}}
    "do.something"() : () -> ()
    hlcf.match.next
  }
  else {
    // expected-warning @below {{unreachable code after match that always completes}}
    "do.something.else"() : () -> ()
    hlcf.yield
  }
  lit.return
  lit.end_fn
}

// No case completes: the first case advances to else, which returns, so code
// after the match is dead.
lit.fn @unreachable_after_match() {
  hlcf.match {
    hlcf.match.next
  }
  else {
    lit.return
    hlcf.yield
  }
  // expected-warning @below {{unreachable code after match statement that does not fall through}}
  lit.return
  lit.end_fn
}
