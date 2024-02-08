// RUN: kgen-opt %s -allow-unregistered-dialect -lower-semantic-cf -verify-diagnostics

lit.func @dead_code() {
  lit.return
  // expected-warning @below {{unreachable code after return statement}}
  "do.something"() : () -> ()
  lit.end_func
}

lit.func @no_return_result() -> i32 {
  // expected-error @below {{return expected at end of function with results}}
  lit.end_func
}

lit.func @no_return_result2(%x: !kgen.pointer<i32> byref_result) -> !kgen.none {
// expected-error @below {{return expected at end of function with results}}
  lit.end_func
}

lit.func @bad_break(%x: !kgen.pointer<i32> byref_result) -> !kgen.none {
  // expected-error @below {{'break' is not inside a loop}}
  lit.break
  lit.end_func
}

lit.func @bad_continue(%x: !kgen.pointer<i32> byref_result) -> !kgen.none {
  // expected-error @below {{'continue' is not inside a loop}}
  lit.continue
  lit.end_func
}

// break in an 'else' is an error unless in a nested loop.
lit.func @bad_break_2(%arg0: i1) {
  // CHECK: hlcf.loop "_loop_0"
  lit.loop cond {
    lit.loop.condition %arg0: i1
  } body {
    lit.loop.continue
  } else {
    lit.break // expected-error {{'break' is not inside a loop}}
    lit.loop.yield
  }

  lit.return
  lit.end_func
}
