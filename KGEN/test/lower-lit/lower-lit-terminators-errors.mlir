// RUN: kgen-opt %s -allow-unregistered-dialect -split-input-file -lower-lit-terminators -verify-diagnostics

lit.func @dead_code() {
  lit.return
  // expected-warning @below {{unreachable code after return statement}}
  "do.something"() : () -> ()
  lit.end_func
}

// -----

lit.func @disagreed_result_parameters<() -> index>(%c: i1) {
  hlcf.if %c {
    // expected-note @below {{see conflicting result meta-parameters here}}
    lit.return<5>
    hlcf.yield
  } else {
    hlcf.yield
  }
  // expected-error @below {{function return defines different result meta-parameters than previous return statement}}
  lit.return<6>
  lit.end_func
}

// -----

lit.func @no_return_result() -> i32 {
  // expected-error @below {{return expected at end of function with results}}
  lit.end_func
}

// -----

lit.func @no_return_result<() -> index>() -> !lit.none {
  // expected-error @below {{return expected at end of function with results}}
  lit.end_func
}

// -----

// expected-error @below {{function throws but no 'Error' type was found}}
lit.func @throws() throws -> !lit.none {
  lit.end_func
}
