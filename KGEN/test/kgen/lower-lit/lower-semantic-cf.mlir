// RUN: kgen-opt %s -lower-semantic-cf -verify-parameters -verify-diagnostics -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: lit.struct.decl @SomeStruct
lit.struct.decl @SomeStruct {
  // CHECK-LABEL: lit.func @dead_returns
  lit.func @dead_returns(%c: i1, %a: i32, %b: i32) -> i32 {
    // CHECK: hlcf.if %c
    hlcf.if %c {
      // CHECK-NEXT: kgen.return %b : i32
      lit.return %b: i32
      lit.return %a: i32 // expected-warning {{unreachable code after return statement}}
      hlcf.yield
    // CHECK-NEXT: else
    } else {
      hlcf.yield
    }
    // CHECK: kgen.return %a : i32
    lit.return %a : i32
    lit.return %b : i32 // expected-warning {{unreachable code after return statement}}
    lit.end_func
  // CHECK-NEXT: }
  }
}

// CHECK-LABEL: lit.file_module @FileModule
lit.file_module @FileModule {
  // CHECK-LABEL: lit.struct.decl @SomeStruct
  lit.struct.decl @SomeStruct {
    // CHECK-LABEL: lit.func @try_and_raise
    lit.func @try_and_raise(%a: i32) throws {
      // CHECK-NEXT: lit.try
      lit.try {
        // CHECK-NEXT: lit.try.raise
        lit.raise
        lit.try.yield
      // CHECK-NEXT: except
      } except {
        // CHECK-NEXT: kgen.return
        lit.return
        lit.try.yield
      // CHECK-NEXT: else
      } else {
        // CHECK-NEXT: kgen.unreachable

        // expected-warning @+1 {{'else' logic in 'try' is unreachable}}
        lit.return
        lit.try.yield
      // CHECK-NEXT: }
      } finally {
        lit.try.yield
      }

      // CHECK-NEXT: kgen.unreachable
      // expected-warning @+1 {{unreachable code after try statement that doesn't fall through}}
      lit.return
      lit.end_func
    }
  }

  // CHECK-LABEL: lit.func @break_and_continue
  lit.func @break_and_continue(%c: i1) {
    // CHECK-NEXT: hlcf.loop
    // CHECK-NEXT: hlcf.if %c {
    // CHECK-NEXT:   hlcf.yield
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   hlcf.break
    // CHECK-NEXT: }
    lit.loop cond {
      lit.loop.condition %c: i1
    } body {
      // CHECK-NEXT: hlcf.if %c {
      hlcf.if %c {
        // CHECK-NEXT: hlcf.break
        lit.break
        lit.continue // expected-warning {{unreachable code after break statement}}
        hlcf.yield
      // CHECK-NEXT: else
      } else {
        // CHECK-NEXT: hlcf.continue
        lit.continue
        lit.break  // expected-warning {{unreachable code after continue statement}}
        hlcf.yield
      // CHECK-NEXT: }
      }
      // CHECK-NEXT: kgen.unreachable
      // CHECK-NEXT: }
      lit.return  // expected-warning {{unreachable code after if statement with then/else that do not fall through}}
      lit.loop.continue
    } else {
      lit.loop.yield
    }

    // CHECK-NEXT: kgen.return
    lit.return
    lit.end_func
  }
}

// CHECK-LABEL: lit.func @no_return
lit.func @no_return() -> !kgen.none {
  // CHECK: kgen.return
  %0 = kgen.param.constant: none = <#kgen.none>
  lit.return %0 :  !kgen.none
  lit.end_func
}

lit.func @if_true_return() -> index {
  %0 = index.constant 0
  %true = index.bool.constant true
  hlcf.if %true {
    lit.return %0 : index
    hlcf.yield
  } else {
    // expected-warning @+1 {{unreachable code after 'if True'}}
    lit.return %0 : index
    hlcf.yield
  }
  lit.end_func
}

lit.func @while_true() -> index {
  %true = index.bool.constant true
  lit.loop cond {
    lit.loop.condition %true: i1
  } body {
    hlcf.if %true {
      lit.continue
      hlcf.yield
    } else {
      hlcf.yield
    }
    lit.break // expected-warning {{unreachable code after if statement with then/else that do not fall through}}
    lit.loop.continue
  } else {
    lit.loop.yield
  }
  lit.end_func
}

// CHECK-LABEL: lit.func @if_false_raise
lit.func @if_false_raise() throws -> i1 {
  %false = index.bool.constant false
  hlcf.if %false {
    hlcf.yield
  // CHECK: else
  } else {
    // CHECK-NEXT: [[TRUE:%.*]] = kgen.param.constant: i1 = <1>
    // CHECK-NEXT: lit.error_return [[TRUE]]
    lit.raise
    hlcf.yield
  }
  lit.end_func
}

// CHECK-LABEL: lit.func @raise_raise
lit.func @raise_raise() throws {
  // CHECK: lit.try
  lit.try {
    // CHECK: lit.try.raise
    lit.raise
    lit.try.yield
  // CHECK-NEXT: except
  } except {
    // CHECK-NEXT: kgen.return
    lit.return
    lit.try.yield
  // CHECK-NEXT: else
  } else {
    lit.try.yield
  // CHECK-NOT: finally
  } finally {
    lit.try.yield
  }

  lit.end_func
}

// CHECK-LABEL: lit.func @throwing_func
lit.func @throwing_func[mut elt, mut lt](
    %0[*""]: !lit.ref<@Error, mut elt> byref_error,
    %1[*""]: !lit.ref<none, mut *[0,1]> byref_result
) throws -> i1 {
  // CHECK-NEXT: [[TRUE:%.*]] = kgen.param.constant: i1 = <1>
  // CHECK-NEXT: lit.error_return [[TRUE]]
  lit.raise
  lit.end_func
}

lit.struct.decl @Error {}

// CHECK-LABEL: lit.func @throwing_calls
lit.func @throwing_calls(
    %f: !lit.signature<[2](!lit.ref<@Error, mut *[0,0]> byref_error, !lit.ref<none, mut *[0,1]> byref_result) throws -> i1>
) throws -> i1 {
  %err = lit.var.decl "err" synth : !lit.ref<@Error, mut elt>
  %result = lit.var.decl "result" synth : !lit.ref<none, mut lt>

  // CHECK:      [[IS_ERR:%.*]] = lit.call @throwing_func
  // CHECK-NEXT: hlcf.if [[IS_ERR]]
  // CHECK-NEXT:   mark_consumed %result
  // CHECK-NEXT:   [[TRUE:%.*]] = kgen.param.constant: i1 = <1>
  // CHECK-NEXT:   lit.error_return [[TRUE]]
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   mark_consumed %err
  // CHECK-NEXT:   yield
  // CHECK-NEXT: }
  lit.call @throwing_func[mut elt, mut lt](%err, %result) : !lit.signature<[2](!lit.ref<@Error, mut *[0,0]> byref_error, !lit.ref<none, mut *[0,1]> byref_result) throws -> i1>

  %error = lit.var.decl "error" synth : !lit.ref<@Error, mut tlt>
  // CHECK: lit.try {
  lit.try %error : !lit.ref<@Error, mut tlt> {
    // CHECK-NEXT: [[IS_ERR:%.*]] = lit.call_indirect %f
    // CHECK-NEXT: hlcf.if [[IS_ERR]]
    // CHECK-NEXT:   mark_consumed %result
    // CHECK-NEXT:   lit.try.raise
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   mark_consumed %error
    // CHECK-NEXT:   yield
    // CHECK-NEXT: }
    lit.call_indirect %f[mut tlt, mut lt](%error, %result) :  !lit.signature<[2](!lit.ref<@Error, mut *[0,0]> byref_error, !lit.ref<none, mut *[0,1]> byref_result) throws -> i1>
    lit.try.yield
  } except {
    lit.try.yield
  } else {
    lit.try.yield
  } finally {
    lit.try.yield
  }
  kgen.unreachable
}

// CHECK-LABEL: lit.func @unreachable_try
lit.func @unreachable_try() {
  lit.try {
    lit.try.yield
  } except {
    // expected-warning @+1 {{'except' logic is unreachable, try doesn't raise an exception}}
    index.constant 0
    lit.try.yield
  } else {
    lit.return
    lit.try.yield
  } finally {
    lit.try.yield
  }
  // CHECK: kgen.unreachable
  // expected-warning @+1 {{unreachable code after try statement that doesn't fall through}}
  index.constant 0
  lit.end_func
}

// CHECK-LABEL: lit.func @suppressed_try
lit.func @suppressed_try() {
  lit.try {
    lit.try.yield
  } except {
    // CHECK: except
    // CHECK-NEXT: kgen.unreachable
    index.constant 0
    lit.try.yield
  } else {
    index.constant 0
    lit.return
    lit.try.yield
  } finally {
    lit.try.yield
  } {"suppressWarnings" = true}
  // CHECK: kgen.unreachable
  // expected-warning @+1 {{unreachable code after try statement that doesn't fall through}}
  index.constant 0
  lit.end_func
}

// CHECK-LABEL: lit.func @coroutine() async -> index
lit.func @coroutine() async -> index {
  %idx0 = index.constant 0
  // CHECK: return %idx0
  lit.return %idx0 : index
  lit.end_func
}

// CHECK-LABEL: lit.func @call_coroutine
// CHECK-SAME: coro: () async -> !kgen.none
// CHECK-SAME: ) async -> !kgen.none
lit.func @call_coroutine<coro: () async -> !kgen.none>() async -> !kgen.none {
  // CHECK-NEXT: lit.async.call[() async -> !kgen.none: coro]()
  lit.async.call[() async -> !kgen.none: coro]()
  %0 = kgen.param.constant: none = <#kgen.none>
  lit.return %0 :  !kgen.none
  lit.end_func
}

// CHECK-LABEL: lit.func @return_after_return
lit.func @return_after_return() -> !kgen.none {
  %0 = kgen.param.constant: none = <#kgen.none>
  // CHECK: kgen.return %none : !kgen.none
  lit.return %0 : !kgen.none
  %1 = kgen.param.constant: i1 = <1>  // expected-warning {{unreachable code after return statement}}
  hlcf.if %1 {
    %2 = kgen.param.constant: none = <#kgen.none>
    lit.return %2 : !kgen.none
    hlcf.yield
  } else {
    hlcf.yield
  }
  lit.end_func
}

// CHECK-LABEL: lit.func @if_else_return
lit.func @if_else_return(%cond: i1) -> index {
  %0 = index.constant 0
  hlcf.if %cond {
    lit.return %0 : index
    hlcf.yield
  } else {
    lit.return %0 : index
    hlcf.yield
  }
  // CHECK: kgen.unreachable
  lit.end_func
}

// CHECK-LABEL: lit.func @coroutine2
lit.func @coroutine2() async -> index {
  %0 = index.constant 0
  %true = index.bool.constant true

  lit.loop cond {
    lit.loop.condition %true: i1
  } body {
    lit.return %0 : index
    lit.break  // expected-warning {{unreachable code after return statement}}
    lit.loop.continue
  } else {
    lit.loop.yield
  }

  // CHECK: kgen.unreachable
  lit.end_func
}

// CHECK-LABEL: lit.func @pointlessTry
lit.func @pointlessTry() -> !kgen.none {
  lit.try { // expected-warning {{try body doesn't raise an exception}}
    lit.try.yield
  } except {
    lit.try.yield
  } else {
    lit.try.yield
  } finally {
    lit.try.yield
  }
  %0 = kgen.param.constant: none = <#kgen.none>
  lit.return %0 :  !kgen.none
  lit.end_func
}

// CHECK-LABEL: lit.func @reraise_in_try
lit.func @reraise_in_try() {
  // CHECK-NEXT: lit.try
  lit.try {
    // CHECK-NEXT: lit.try
    lit.try {
      // CHECK-NEXT: lit.try.raise
      lit.raise
      lit.try.yield
    // CHECK-NEXT: except
    } except {
      // CHECK-NEXT: lit.try.raise
      lit.raise
      lit.try.yield
    // CHECK-NEXT: else
    } else {
      // CHECK-NEXT: unreachable
      lit.try.yield
    // CHECK-NOT: finally
    } finally {
      lit.try.yield
    }
    // CHECK: unreachable
    lit.try.yield
  // CHECK-NEXT: except
  } except {
    // CHECK-NEXT: yield
    lit.try.yield
  // CHECK-NEXT: else
  } else {
    // CHECK-NEXT: unreachable
    lit.try.yield
  } finally {
    lit.try.yield
  }
  kgen.return
}

// CHECK-LABEL: lit.func @finally_breaks
lit.func @finally_breaks() -> index {
  // CHECK-LABEL: lit.try
  lit.try {
    // CHECK-NEXT: lit.try.yield
    lit.try.yield
  // CHECK-NEXT: except
  } except (%e: index) {
    // CHECK-NEXT: unreachable
    lit.try.yield
  // CHECK-NEXT: else
  } else {
    // CHECK: kgen.return %idx0
    lit.try.yield
  // CHECK-NOT: finally
  } finally {
    %idx0 = index.constant 0
    lit.return %idx0 : index
    lit.try.yield
  }
  // CHECK: kgen.unreachable
  lit.end_func
}

// CHECK-LABEL: lit.func @try_finally
lit.func @try_finally(%arg0: i1, %arg1: i32, %arg2: i64) -> (i32, i64) {
  %true = index.bool.constant true

  // CHECK: hlcf.loop "_loop_0" {
  // CHECK-NEXT: hlcf.if %true {
  // CHECK-NEXT:         hlcf.yield
  // CHECK-NEXT:       } else {
  // CHECK-NEXT:         kgen.unreachable
  // CHECK-NEXT:       }
  lit.loop cond {
    lit.loop.condition %true: i1
  } body {
    // CHECK-NEXT: lit.try
    lit.try {
      // CHECK-NEXT: hlcf.if %arg0
      hlcf.if %arg0 {
        // CHECK: clean.up
        // CHECK-NEXT: break
        hlcf.break
      // CHECK-NEXT: else
      } else {
        // CHECK-NEXT: yield
        hlcf.yield
      }
      // CHECK: clean.up
      // CHECK-NEXT: return %arg1, %arg2
      kgen.return %arg1, %arg2 : i32, i64
    // CHECK-NEXT: except
    } except (%err: index) {
      // CHECK-NEXT: unreachable
      lit.try.yield
    // CHECK-NEXT: else
    } else {
      // CHECK-NEXT: unreachable
      lit.try.yield
    // CHECK-NOT: finally
    } finally {
      "clean.up"() : () -> ()
      lit.try.yield
    }
    // CHECK: unreachable
    lit.break // expected-warning {{unreachable code after try statement that doesn't fall through}}
    lit.loop.continue
  } else {
    lit.loop.yield
  }
  // CHECK: return %arg1, %arg2
  kgen.return %arg1, %arg2 : i32, i64
}

// CHECK-LABEL: lit.func @try_finally_return
lit.func @try_finally_return(%arg0: index, %arg1: index, %arg2: i1) -> index {
  %true = index.bool.constant true

  // CHECK: hlcf.loop "_loop_0" {
  // CHECK-NEXT: hlcf.if %true {
  // CHECK-NEXT:         hlcf.yield
  // CHECK-NEXT:       } else {
  // CHECK-NEXT:         kgen.unreachable
  // CHECK-NEXT:       }

  lit.loop cond {
    lit.loop.condition %true: i1
  } body {
    // CHECK-NEXT: lit.try
    lit.try {
      // CHECK-NEXT: hlcf.if %arg2
      hlcf.if %arg2 {
        // CHECK-NEXT: return %arg1
        hlcf.break
      // CHECK-NEXT: else
      } else {
        // CHECK-NEXT: return %arg1
        hlcf.continue
      }
      // CHECK: unreachable
      kgen.return %arg0 : index
    } except (%err: index) {
      lit.try.yield
    } else {
      lit.try.yield
    } finally {
      kgen.return %arg1 : index
    }
    lit.break // expected-warning {{unreachable code after try statement that doesn't fall through}}
    lit.loop.continue
  } else {
    lit.loop.yield
  }

  kgen.return %arg1 : index
}

// CHECK-LABEL: lit.func @nested_try_finally
lit.func @nested_try_finally() {
  // CHECK-NEXT: lit.try
  lit.try {
    // CHECK-NEXT: lit.try
    lit.try {
      // CHECK-NEXT: clean.up0
      // CHECK-NEXT: clean.up1
      // CHECK-NEXT: return
      kgen.return
    } except (%err: index) {
      lit.try.yield
    // CHECK: else
    } else {
      lit.try.yield
    } finally {
      "clean.up0"() : () -> ()
      lit.try.yield
    }
    lit.try.yield
  } except (%err: index) {
    lit.try.yield
  // CHECK: else
  } else {
    // CHECK-NEXT: unreachable
    lit.try.yield
  } finally {
    "clean.up1"() : () -> ()
    lit.try.yield
  }
  // CHECK: unreachable
  kgen.return
}

// CHECK-LABEL: lit.func @try_in_loop
lit.func @try_in_loop(%arg0: i1) {
  lit.loop cond {
    lit.loop.condition %arg0: i1
  } body {
    lit.try {
      lit.try.yield
    // CHECK: except
    } except (%e: index) {
      // CHECK-NEXT: kgen.unreachable
      lit.try.yield
    } else {
      lit.try.yield
    } finally {
      lit.try.yield
    } {"suppressWarnings" = true}
    // CHECK: hlcf.continue
    lit.loop.continue
  } else {
    lit.loop.yield
  }
  // CHECK: after.loop
  "after.loop"() : () -> ()
  kgen.return
}

// CHECK-LABEL: lit.func @recurse
// CHECK-SAME (%x: !pop.scalar<index>) -> !pop.scalar<index> {
// CHECK-NEXT: %0 = kgen.call @recurse(%x) : !lit.signature<("x": !pop.scalar<index>) -> !pop.scalar<index>>
// CHECK-NEXT: kgen.return %0 : !pop.scalar<index>
// CHECK-NEXT:}
lit.func @recurse(%x: !pop.scalar<index>) -> !pop.scalar<index> {
  %0 = kgen.call @recurse(%x) : !lit.signature<("x": !pop.scalar<index>) -> !pop.scalar<index>>
  lit.return %0 : !pop.scalar<index>
  lit.end_func
}

// CHECK-LABEL: lit.func @coroutine_await
lit.func @coroutine_await(%arg0: i1) {
  // CHECK-NEXT: co.suspend
  co.suspend (%hdl0) {
    hlcf.if %arg0 {
      // CHECK: kgen.return
      lit.return
      hlcf.yield
    } else {
      hlcf.yield
    }
    // CHECK: co.suspend.end
    co.suspend.end
  }
  lit.return
  lit.end_func
}

// CHECK-LABEL: lit.func @loop_with_else
lit.func @loop_with_else(%arg0: i1) {
  // CHECK: hlcf.loop "_loop_0"
  lit.loop cond {
    lit.loop.condition %arg0: i1
  } body {
    lit.loop cond {
      lit.loop.condition %arg0: i1
    } body {
      // CHECK: hlcf.if %arg0 {
      // CHECK-NEXT:   hlcf.yield
      // CHECK-NEXT: } else {
      // CHECK-NEXT:   hlcf.break
      // CHECK-NEXT: }
      // CHECK-NEXT: hlcf.loop "_loop_1" {
      // CHECK-NEXT:   hlcf.if %arg0 {
      // CHECK-NEXT:     hlcf.yield
      // CHECK-NEXT:   } else {
      // CHECK-NEXT:     hlcf.continue "_loop_0"
      // CHECK-NEXT:   }
      // CHECK-NEXT:   hlcf.continue
      // CHECK-NEXT: }
      // CHECK-NEXT: kgen.unreachable
      lit.loop.continue
    } else {
      lit.continue
      lit.break     // expected-warning {{unreachable code after continue statement}}
      lit.loop.yield
    }
    lit.loop.continue
  } else {
    lit.loop.yield
  }

  lit.return
  lit.end_func
}

// CHECK-LABEL: lit.trait.decl @Trait
lit.trait.decl @Trait {
  // CHECK-NOT: @trait_fn
  lit.func @trait_fn() {
    lit.trait_func
  }
}

// CHECK-LABEL: lit.func @loop_with_cond_raise
// Crash handling exception
// https://github.com/modularml/modular/issues/27937
// Checking the loop body clobbered the "can raise" flag for the try block.
lit.func @loop_with_cond_raise(%cond: i1) {
  lit.try {
    hlcf.if %cond {
      lit.raise
      hlcf.yield
    } else {
      hlcf.yield
    }

    lit.loop cond {
      lit.loop.condition %cond: i1
    } body {
      hlcf.if %cond {
        hlcf.yield
      } else {
        hlcf.break
      }
      lit.loop.continue
    } else {
      lit.loop.yield
    }
    lit.try.yield
  // CHECK: } except {
  } except {
    // CHECK-NEXT: kgen.return
    lit.return
    kgen.unreachable
  } else {
    lit.try.yield
  } finally {
    lit.try.yield
  }
  lit.return
  lit.end_func
}

// [QoI] Generate error for obviously self recursive functions
// https://github.com/modularml/mojo/issues/222
lit.func @self_recursive() -> !kgen.none {
  // expected-warning @+1 {{self recursive call will cause an infinite loop}}
  %0 = lit.call @self_recursive() : !lit.signature<() -> !kgen.none>
  %none = kgen.param.constant: none = <#kgen.none>
  lit.return %none : !kgen.none
  lit.end_func
}
lit.func @self_recursive_arg(%a: index, %cond: i1) -> !kgen.none {
  // expected-warning @+1 {{self recursive call will cause an infinite loop}}
  %0 = lit.call @self_recursive_arg(%a, %cond) : !lit.signature<("a": index, "cond": i1) -> !kgen.none>
  hlcf.if %cond {
    %4 = kgen.param.constant: index = <1>
    %5 = index.sub %a, %4
    // No warning.
    %6 = lit.call @self_recursive_arg(%5, %cond) : !lit.signature<("a": index, "cond": i1) -> !kgen.none>
    hlcf.yield
  } else {
    hlcf.yield
  }
  %none = kgen.param.constant: none = <#kgen.none>
  lit.return %none : !kgen.none
  lit.end_func
}

lit.func @self_recursive_param<a: index, cond: i1>() -> !kgen.none attributes {sourceName = "self_recursive_param", specialFnKind = 0 : i8} {
  // expected-warning @+1 {{self recursive call will cause an infinite loop}}
  %0 = lit.call @self_recursive_param<a, :i1 cond>() : !lit.signature<() -> !kgen.none>
  kgen.param.if <cond> {
    // No warning.
    %1 = lit.call @self_recursive_param<a, :i1 cond>() : !lit.signature<() -> !kgen.none>
    kgen.param.yield
  } else {
    kgen.param.yield
  }
  %none = kgen.param.constant: none = <#kgen.none>
  lit.return %none : !kgen.none
  lit.end_func
}

// #28551: Should report infinite recursion on this testcase
lit.func @self_recursive_arg_diff(%a: index) -> !kgen.none {
  %one = kgen.param.constant: index = <1>
  %b = index.sub %a, %one
  // expected-warning @+1 {{self recursive call will cause an infinite loop}}
  lit.call @self_recursive_arg_diff(%b) : !lit.signature<("a": index) -> !kgen.none>

  %none = kgen.param.constant: none = <#kgen.none>
  lit.return %none : !kgen.none
  lit.end_func
}

// CHECK-LABEL: lit.func @elif
// CHECK-NEXT: %idx0 = index.constant 0
// CHECK-NEXT: %idx1 = index.constant 1
// CHECK-NEXT: %idx2 = index.constant 2
// CHECK-NEXT: %0 = hlcf.elif -> index {
// CHECK-NEXT: [[V1:%*.]] = index.cmp eq(%arg0, %idx0)
// CHECK-NEXT: hlcf.elif.yield [[V1]] : i1
// CHECK-NEXT: } then {
// CHECK-NEXT: hlcf.yield %arg0 : index
// CHECK-NEXT: } {
// CHECK-NEXT: [[V2:%*.]] = index.cmp eq(%arg0, %idx1)
// CHECK-NEXT: hlcf.elif.yield [[V2]] : i1
// CHECK-NEXT: } then {
// CHECK-NEXT: kgen.return %arg1 : index
// CHECK-NEXT: } else {
// CHECK-NEXT: kgen.return %arg1 : index
// CHECK-NEXT: }
lit.func @elif(%arg0: index, %arg1: index, %arg2: index) -> index {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %idx2 = index.constant 2
  %0 = hlcf.elif -> index {
    %c = index.cmp eq(%arg0, %idx0)
    hlcf.elif.yield %c : i1
  } then {
    hlcf.yield %arg0 : index
  } {
    %c = index.cmp eq(%arg0, %idx1)
    hlcf.elif.yield %c : i1
  } then {
    lit.return %arg1 : index
    hlcf.yield %arg1 : index
  } else {
    lit.return %arg1 : index
    hlcf.yield %arg2 : index
  }
  kgen.return %0 : index
}


// COM: https://github.com/modularml/modular/issues/33570
// COM: When cloning the finally block, we must uniquely mangle parameters to
// COM: avoid duplicate parameter name errors.
// CHECK-LABEL: lit.func @mangle_params_finally_1
lit.func @mangle_params_finally_1<x>(%c: i1 borrow) -> !kgen.none {
  lit.try {
    // CHECK: hlcf.if %c
    hlcf.if %c {
      %none_0 = kgen.param.constant: none = <#kgen.none>
      // CHECK: lit.alias.decl *"y`"
      // CHECK-NEXT: kgen.return
      lit.return %none_0 : !kgen.none
      hlcf.yield
    // CHECK-NEXT: } else {
    } else {
      // CHECK-NEXT: hlcf.yield
      hlcf.yield
    }
    lit.try.yield
  // CHECK: } except
  } except {
    // CHECK-NEXT: kgen.unreachable
    lit.try.yield
  // CHECK-NEXT: } else {
  } else {
    // CHECK-NEXT: lit.alias.decl *"y`f0"
    // CHECK-NEXT: lit.try.yield
    lit.try.yield
  } finally {
    lit.alias.decl *"y`" = <x>
    lit.try.yield
  }
  %none = kgen.param.constant: none = <#kgen.none>
  lit.return %none : !kgen.none
  lit.end_func
}


// CHECK-LABEL: lit.func @mangle_params_finally_2
lit.func @mangle_params_finally_2<x>(%c: i1 borrow) -> !kgen.none {
  lit.try {
    // CHECK: hlcf.if %c
    hlcf.if %c {
      %none_1 = kgen.param.constant: none = <#kgen.none>
      // CHECK: lit.alias.decl *"y`"
      // CHECK-NEXT: kgen.return
      lit.return %none_1 : !kgen.none
      hlcf.yield
    } else {
      hlcf.yield
    }

    // CHECK: hlcf.if %c
    hlcf.if %c {
      %none_1 = kgen.param.constant: none = <#kgen.none>
      // CHECK: lit.alias.decl *"y`f0"
      // CHECK-NEXT: kgen.return
      lit.return %none_1 : !kgen.none
      hlcf.yield
    } else {
      hlcf.yield
    }

    %none_0 = kgen.param.constant: none = <#kgen.none>
    // CHECK: lit.alias.decl *"y`f1"
    // CHECK: kgen.return
    lit.return %none_0 : !kgen.none
    lit.try.yield
  // CHECK: } except
  } except {
    // CHECK-NEXT: kgen.unreachable
    lit.try.yield
  // CHECK-NEXT: } else {
  } else {
    // CHECK-NEXT: kgen.unreachable
    lit.try.yield
  } finally {
    lit.alias.decl *"y`" = <x>
    lit.try.yield
  }
  // CHECK: kgen.unreachable

  // expected-warning @+1 {{unreachable code after try statement that doesn't fall through}}
  %none = kgen.param.constant: none = <#kgen.none>
  lit.return %none : !kgen.none
  lit.end_func
}


// CHECK-LABEL: lit.func @mangle_params_finally_3
lit.func @mangle_params_finally_3<x>(%c: i1 borrow) -> !kgen.none {
  lit.try {
    // CHECK: lit.func nested()
    lit.func nested() -> !kgen.none {
      // CHECK-NEXT: %[[NONE:.*]] = kgen.param.constant: none
      %none_0 = kgen.param.constant: none = <#kgen.none>
      // CHECK-NEXT: kgen.return %[[NONE:.*]]
      lit.return %none_0 : !kgen.none
      lit.end_func
    }
    // CHECK: hlcf.if
    hlcf.if %c {
      %none_0 = kgen.param.constant: none = <#kgen.none>
      // CHECK: lit.alias.decl *"y`"
      // CHECK: kgen.return
      lit.return %none_0 : !kgen.none
      hlcf.yield
    // CHECK: } else {
    } else {
      // CHECK-NEXT: hlcf.yield
      hlcf.yield
    }
    lit.try.yield
  } except {
    lit.try.yield
  // CHECK: } else {
  } else {
    // CHECK: lit.alias.decl *"y`f0"
    // CHECK: lit.try.yield
    lit.try.yield
  } finally {
    lit.alias.decl *"y`" = <x>
    lit.try.yield
  }
  %none = kgen.param.constant: none = <#kgen.none>
  lit.return %none : !kgen.none
  lit.end_func
}

// CHECK-LABEL: lit.func @containsEarlyReturn
lit.func @containsEarlyReturn(%arg: i1) -> !kgen.none {
  // CHECK: hlcf.elif {
  // CHECK:     hlcf.elif.yield %arg : i1
  // CHECK:    } then {
  // CHECK:     %none = kgen.param.constant: none = <#kgen.none>
  // CHECK:     kgen.return %none : !kgen.none
  // CHECK:   } else {
  // CHECK:     %none = kgen.param.constant: none = <#kgen.none>
  // CHECK:     kgen.return %none : !kgen.none
  // CHECK:   }
  // CHECK:   kgen.unreachable
  hlcf.elif {
    hlcf.elif.yield %arg : i1
  } then {
    %none_0 = kgen.param.constant: none = <#kgen.none>
    lit.return %none_0 : !kgen.none
    hlcf.yield
  } else {
    %none_0 = kgen.param.constant: none = <#kgen.none>
    lit.return %none_0 : !kgen.none
    hlcf.yield
  }
  lit.end_func
}

// CHECK-LABEL: lit.func @fallthrough
lit.func @fallthrough<cond0: i1, cond1: i1>(%lhs: index, %rhs: index, %cond2 : i1) -> index {
// CHECK: kgen.param.if <cond0> {
// CHECK-NEXT:   kgen.return %lhs : index
// CHECK-NEXT: } else {
// CHECK-NEXT: kgen.param.if <cond1> {
// CHECK-NEXT:   kgen.return %rhs : index
// CHECK-NEXT:  } else {
// CHECK-NEXT:  hlcf.elif {
// CHECK-NEXT:    hlcf.elif.yield %cond2 : i1
// CHECK-NEXT:  } then {
// CHECK-NEXT:    hlcf.yield
// CHECK-NEXT:  } else {
// CHECK-NEXT:    hlcf.yield
// CHECK-NEXT:  }
// CHECK-NEXT:  %index0 = kgen.param.constant = <0>
// CHECK-NEXT:  kgen.return %index0 : index
// CHECK-NEXT:  }
// CHECK-NEXT:  kgen.unreachable
// CHECK-NEXT: }
// CHECK-NEXT: kgen.unreachable
 kgen.param.if <cond0> {
   lit.return %lhs : index
   kgen.param.yield
 } else {
   kgen.param.if <cond1> {
     lit.return %rhs : index
     kgen.param.yield
   } else {
     hlcf.elif {
       hlcf.elif.yield %cond2 : i1
     } then {
       hlcf.yield
     } else {
       hlcf.yield
     }
     %0 = kgen.param.constant: index = <0>
     lit.return %0 : index
     kgen.param.yield
   }
   kgen.param.yield
 }
 lit.end_func
}


// CHECK-LABEL: lit.func @consecutiveElifs
lit.func @consecutiveElifs(%arg0: index, %arg1: index) -> index {
  %idx0 = index.constant 0
  %idx1 = index.constant 1

  // CHECK:  hlcf.elif -> index {
  // CHECK-NEXT: index.cmp eq(%arg0, %idx0)
  %0 = hlcf.elif -> index {
    %c = index.cmp eq(%arg0, %idx0)
    hlcf.elif.yield %c : i1
  } then {
    hlcf.yield %arg0 : index
  } else {
    hlcf.yield %arg1 : index
  }
  // CHECK:  hlcf.elif {
  // CHECK-NEXT:   index.cmp eq(%arg0, %idx1)
  // CHECK-NEXT:   hlcf.elif.yield
  // CHECK-NEXT: } then {
  // CHECK-NEXT:   kgen.return %arg0 : index
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   kgen.return %arg1 : index
  // CHECK-NEXT: }
  hlcf.elif {
    %c = index.cmp eq(%arg0, %idx1)
    hlcf.elif.yield %c : i1
  } then {
    lit.return %arg0 : index
    hlcf.yield
  } else {
    lit.return %arg1 : index
    hlcf.yield
  }
  // CHECK-NEXT: kgen.unreachable
  // CHECK-NEXT: }
  lit.end_func
}
