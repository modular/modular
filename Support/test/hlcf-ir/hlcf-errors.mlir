// RUN: support-dialect-opt %s -allow-unregistered-dialect -verify-diagnostics -split-input-file

func.func @loop_args() {
  // expected-error @below {{'hlcf.loop' op operand types do not match body region argument types}}
  "hlcf.loop"() ({
  ^bb0(%arg0: i32):
    hlcf.return
  }) : () -> ()
  return
}

// -----

func.func @return_not_in_func(%arg0: i1) {
  "foo.region"() ({
    // expected-note @below {{see control-flow root here}}
    hlcf.if %arg0 {
      // expected-error @below {{'hlcf.return' op is not nested within a function}}
      hlcf.return
    } else {
      hlcf.yield
    }
    "foo.terminator"() : () -> ()
  }) : () -> ()
}

// -----

// expected-note @below {{see function here}}
func.func @return_mismatch_result_count(%arg0: i32) {
  hlcf.loop {
    // expected-error @below {{'hlcf.return' op specifies 1 results but surrounding function expects 0}}
    hlcf.return %arg0 : i32
  }
}

// -----

// expected-note @below {{see function here}}
func.func @return_mismatch_result_count(%arg0: i32) -> i64 {
  hlcf.loop {
    // expected-error @below {{'hlcf.return' op operand #0 type 'i32' does not match expected result type 'i64'}}
    hlcf.return %arg0 : i32
  }
}

// -----

func.func @yield_mismatch(%arg0: i1, %arg1 : i32) {
  // expected-note @below {{to end of parent operation here}}
  %0 = hlcf.if %arg0 -> i64 {
    // expected-error @below {{'hlcf.yield' op branch input #0 has type 'i32' but target expected 'i64' along control-flow edge from here}}
    hlcf.yield %arg1 : i32
  } else {
    hlcf.return
  }
}

// -----

// expected-note @below {{see control-flow root here}}
func.func @break_no_loop(%arg0: i1) {
  hlcf.if %arg0 {
    hlcf.return
  } else {
    // expected-error @below {{'hlcf.break' op is not nested within a suitable parent operation}}
    hlcf.break
  }
}

// -----

func.func @break_wrong_types(%arg0: i32) {
  // expected-note @below {{to end of parent operation here}}
  %0 = hlcf.loop () -> i64 {
    // expected-error @below {{'hlcf.break' op branch input #0 has type 'i32' but target expected 'i64' along control-flow edge from here}}
    hlcf.break %arg0 : i32
  }
}

// -----

func.func @continue_wrong_types(%arg0: i32, %arg1 : i64) {
  hlcf.loop (%0 = %arg0 : i32) -> () {
    // expected-error @below {{'hlcf.continue' op branch input #0 has type 'i64' but target expected 'i32' along control-flow edge from here}}
    // expected-note @below {{to beginning of region #0 here}}
    hlcf.continue %arg1 : i64
  }
}

// -----

func.func @labelled_break_mismatch(%arg0: i32) {
  // expected-note @below {{to end of parent operation here}}
  hlcf.loop "foo" () -> index {
    hlcf.loop () -> i32 {
      // expected-error @below {{'hlcf.break' op branch input #0 has type 'i32' but target expected 'index' along control-flow edge from here}}
      hlcf.break "foo" %arg0 : i32
    }
    hlcf.continue
  }
  return
}
