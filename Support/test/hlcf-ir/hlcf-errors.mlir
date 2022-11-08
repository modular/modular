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
