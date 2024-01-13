// RUN: kgen-opt %s -allow-unregistered-dialect -split-input-file -lower-semantic-cf -verify-diagnostics

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
