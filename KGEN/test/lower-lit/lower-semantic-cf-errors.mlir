// RUN: kgen-opt %s -allow-unregistered-dialect -split-input-file -lower-semantic-cf -verify-diagnostics

lit.func @dead_code() {
  lit.return
  // expected-warning @below {{unreachable code after return statement}}
  "do.something"() : () -> ()
  lit.end_func
}

// -----

lit.func @no_return_result() -> i32 {
  // expected-error @below {{return expected at end of function with results}}
  lit.end_func
}

// -----

lit.func @no_return_result(%x: !pop.pointer<i32> byref_result) -> !lit.none {
  // expected-error @below {{return expected at end of function with results}}
  lit.end_func
}

// -----

lit.func @no_return_result<() -> index>() -> !lit.none {
  // expected-error @below {{missing parameter return for function with result parameters}}
  lit.end_func
}

// -----

// expected-error @below {{function throws but no 'Error' type was found}}
lit.func @throws() throws -> !lit.none {
  lit.end_func
}

// -----

lit.func @result_params<() -> r0>() {
  lit.return
  // expected-error @below {{missing parameter return for function with result parameters}}
  lit.end_func
}

// -----

lit.func @result_params<() -> r0>() {
  // expected-error @below {{result parameters are not defined along all branches}}
  kgen.param.if <1> {
    lit.param_return<1>
    kgen.param.yield
  } else {
    kgen.param.yield
  }
  lit.return
  lit.end_func
}
