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

lit.func @no_return_result2(%x: !kgen.pointer<i32> byref_result) -> !lit.none {
// expected-error @below {{return expected at end of function with results}}
  lit.end_func
}

// expected-error @below {{missing parameter return for function with result parameters}}
lit.func @no_return_result3<() -> index>() -> !lit.none {
  %0 = kgen.param.constant: !lit.none = <#lit.none>
  lit.return %0 :  !lit.none
  lit.end_func
}

// expected-error @below {{missing parameter return for function with result parameters}}
lit.func @result_params4<() -> r0>(%a: !lit.none) -> !lit.none {
  lit.return %a : !lit.none
  lit.end_func
}

lit.func @result_params5<() -> r0>() -> !lit.none {
  // expected-note @+1 {{previous parameter return is here}}
  lit.param_return<1>

  // expected-error @+1 {{result parameters already defined in this scope}}
  lit.param_return<2>

  %0 = kgen.param.constant: !lit.none = <#lit.none>
  lit.return %0 :  !lit.none
  lit.end_func
}

lit.func @if_result_params1<cond: i1 -> r0>(%a: !lit.none) -> !lit.none {
  // expected-error @below {{result parameters in '@parameter if' may not use fall-through else}}
  kgen.param.if <cond> {
    lit.param_return<1>  // expected-note {{one parameter return is here}}
    kgen.param.yield
  } else {
    kgen.param.yield
  }
  lit.return %a : !lit.none
  lit.end_func
}

lit.func @if_result_params2<cond: i1 -> r0>(%a: !lit.none) -> !lit.none {
  // expected-error @below {{result parameters must be specified in 'if' and 'else' branches of '@parameter if'}}
  kgen.param.if <cond> {
    lit.param_return<1>  // expected-note {{one parameter return is here}}
    kgen.param.yield
  } else {
    %0 = kgen.param.constant: !lit.none = <#lit.none>
    kgen.param.yield
  }
  lit.return %a : !lit.none
  lit.end_func
}


lit.func @if_result_params3<cond: i1 -> r0>(%a: !lit.none) -> !lit.none {
  // this is ok because the condition is known true.
  kgen.param.if <1> {
    lit.param_return<1>
    kgen.param.yield
  } else {
    kgen.param.yield
  }
  lit.return %a : !lit.none
  lit.end_func
}
