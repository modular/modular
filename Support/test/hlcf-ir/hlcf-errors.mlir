// RUN: index-opt %s -allow-unregistered-dialect -verify-diagnostics -split-input-file

// expected-note @below {{see invalid parent here}}
func.func @hlcf_parent_op() {
  // expected-error @below {{'hlcf.return' op expected parent operation to be a control-flow operation but got 'func.func'}}
  hlcf.return
}

// -----

func.func @hlcf_terminator_op() {
  // expected-error @below {{'hlcf.loop' op expected terminator without successors to be a control-flow terminator but got 'foo.terminator'}}
  hlcf.loop {
    // expected-note @below {{see invalid terminator here}}
    "foo.terminator"() : () -> ()
  }
  return
}

// -----

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
    // expected-error @below {{'hlcf.return' op specifies 1 return values but surrounding function expects 0}}
    hlcf.return %arg0 : i32
  }
}

// -----

// expected-note @below {{see function here}}
func.func @return_mismatch_result_count(%arg0: i32) -> i64 {
  hlcf.loop {
    // expected-error @below {{'hlcf.return' op operand #0 type 'i32' does not match expected return value type 'i64'}}
    hlcf.return %arg0 : i32
  }
}

// -----

func.func @yield_mismatch(%arg0: i1, %arg1 : i32) {
  // expected-note @below {{see if here}}
  %0 = hlcf.if %arg0 -> i64 {
    // expected-error @below {{'hlcf.yield' op operand #0 type 'i32' does not match expected result type 'i64'}}
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
    // expected-error @below {{'hlcf.break' op is not nested within an 'hlcf.loop' operation}}
    hlcf.break
  }
}

// -----

func.func @break_wrong_types(%arg0: i32) {
  // expected-note @below {{see loop here}}
  %0 = hlcf.loop () -> i64 {
    // expected-error @below {{'hlcf.break' op operand #0 type 'i32' does not match expected result type 'i64'}}
    hlcf.break %arg0 : i32
  }
}

// -----

func.func @continue_wrong_types(%arg0: i32, %arg1 : i64) {
  // expected-note @below {{see loop here}}
  hlcf.loop (%0 = %arg0 : i32) -> () {
    // expected-error @below {{'hlcf.continue' op operand #0 type 'i64' does not match expected argument type 'i32'}}
    hlcf.continue %arg1 : i64
  }
}
