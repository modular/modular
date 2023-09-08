// RUN: kgen-opt %s -lower-semantic-cf -verify-parameters -verify-diagnostics -allow-unregistered-dialect | FileCheck %s

lit.struct.decl @Error {}

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
    // CHECK-SAME:  -> !pop.variant<@Error, i32>
    lit.func @try_and_raise(%a: i32, %b: !kgen.declref<@Error>) throws -> !pop.variant<@Error, i32> {
      // CHECK-NEXT: lit.try
      lit.try {
        // CHECK-NEXT: lit.try.raise %b
        lit.raise %b : !kgen.declref<@Error>
        lit.try.yield
      // CHECK-NEXT: except (%arg0:
      } except (%err: !kgen.declref<@Error>) {
        // CHECK-NEXT: %[[R:.*]] = pop.variant.create %arg0
        // CHECK-NEXT: kgen.return %[[R]]
        %tmp2 = pop.variant.create %err : !kgen.declref<@Error> -> !pop.variant<@Error, i32>
        lit.return %tmp2 : !pop.variant<@Error, i32>
        lit.try.yield
      // CHECK-NEXT: else
      } else {
        // CHECK-NEXT: kgen.unreachable

        // expected-warning @+1 {{'else' logic in 'try' is unreachable}}
        %tmp3 = pop.variant.create %a : i32 -> !pop.variant<@Error, i32>
        lit.return %tmp3 : !pop.variant<@Error, i32>
        lit.try.yield
      // CHECK-NEXT: }
      } finally {
        lit.try.yield
      }

      // CHECK-NEXT: kgen.unreachable
      // expected-warning @+1 {{unreachable code after try statement that doesn't fall through}}
      %tmp1 = pop.variant.create %a : i32 -> !pop.variant<@Error, i32>
      lit.return %tmp1 : !pop.variant<@Error, i32>
      lit.end_func
    }
  }

  // CHECK-LABEL: lit.func @break_and_continue
  lit.func @break_and_continue(%c: i1) {
    // CHECK-NEXT: hlcf.loop
    hlcf.loop {
      // CHECK-NEXT: hlcf.if
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
      lit.return  // expected-warning {{unreachable code after if statement with then/else that do not fall through}}
      hlcf.continue
    // CHECK-NEXT: }
    }
    // CHECK-NEXT: kgen.return
    lit.return
    lit.end_func
  }
}

// CHECK-LABEL: lit.func @result_parameters
lit.func @result_parameters<() -> r1: i32, r2: i64>(%c: i1) {
  // CHECK: hlcf.if
  hlcf.if %c {
    // CHECK-NEXT: kgen.return
    lit.param_return<:i32 1, :i64 2>
    lit.return
    hlcf.yield
  } else {
    hlcf.yield
  }
  // CHECK: kgen.param.result_bind<:i32 1, :i64 2>
  lit.return
  lit.end_func
}


// CHECK-LABEL: lit.func @no_return
lit.func @no_return() -> !lit.none {
  // CHECK: kgen.return
  %0 = kgen.param.constant: !lit.none = <#lit.none>
  lit.return %0 :  !lit.none
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
  hlcf.loop {
    %true = index.bool.constant true
    hlcf.if %true {
      lit.continue
      hlcf.yield
    } else {
      hlcf.yield
    }
    lit.break // expected-warning {{unreachable code after if statement with then/else that do not fall through}}
    hlcf.continue
  }
  lit.end_func
}


// CHECK-LABEL: lit.func @if_false_raise
lit.func @if_false_raise() throws -> !pop.variant<@Error, index> {
  %false = index.bool.constant false
  hlcf.if %false {
    hlcf.yield
  } else {
    %err = lit.struct.create () : () -> !kgen.declref<@Error>
    %tmp = pop.variant.create %err : !kgen.declref<@Error>
        -> !pop.variant<@Error, index>
    lit.return %tmp : !pop.variant<@Error, index>
    hlcf.yield
  }
  lit.end_func
}

// CHECK-LABEL: lit.func @raise_raise
lit.func @raise_raise() throws -> !pop.variant<@Error, index> {
  // CHECK: lit.try
  lit.try {
    %err = lit.struct.create () : () -> !kgen.declref<@Error>
    // CHECK: lit.try.raise
    lit.raise %err : <@Error>
    lit.try.yield
  // CHECK-NEXT: except
  } except (%err: !kgen.declref<@Error>) {
    %tmp = pop.variant.create %err : !kgen.declref<@Error> -> !pop.variant<@Error, index>
    // CHECK: kgen.return
    lit.return %tmp : !pop.variant<@Error, index>
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

// CHECK-LABEL: @no_return_throws
// CHECK-SAME ) -> !pop.variant<@Error, !lit.none>
lit.func @no_return_throws() throws -> !pop.variant<@Error, !lit.none> {
  %0 = kgen.param.constant: !lit.none = <#lit.none>
  %tmp2 = pop.variant.create %0 : !lit.none -> !pop.variant<@Error, !lit.none>
  lit.return %tmp2 : !pop.variant<@Error, !lit.none>
  // CHECK-NEXT: %0 = kgen.param.constant: !lit.none = <#lit.none>
  // CHECK-NEXT: %1 = pop.variant.create %0
  // CHECK-NEXT: kgen.return %1
  lit.end_func
}

// CHECK-LABEL: lit.func @throws
// CHECK-SAME:  -> !pop.variant<@Error, index>
lit.func @throws(%e: !kgen.declref<@Error>) throws -> !pop.variant<@Error, index> {
  // CHECK-NEXT: pop.variant.create %e
  %tmp = pop.variant.create %e : !kgen.declref<@Error> -> !pop.variant<@Error, index>
  lit.return %tmp : !pop.variant<@Error, index>
  lit.end_func
}

// CHECK-LABEL: lit.func @ref
// CHECK-SAME: !kgen.signature<() throws -> !pop.variant<@Error, !lit.none>>
lit.func @ref(%e: !kgen.declref<@Error>,
              %f: !kgen.signature<() throws -> !pop.variant<@Error, !lit.none>>) throws -> !pop.variant<@Error, !lit.none> {
  lit.try {
    // CHECK: = kgen.call @throws
    kgen.call @throws(%e) : (!kgen.declref<@Error>) throws -> !pop.variant<@Error, index>
    lit.try.yield
  } except (%err: !kgen.declref<@Error>) {
    // CHECK: except (
    // CHECK-NEXT: kgen.unreachable
    // expected-warning @+1 {{'except' logic is unreachable, try doesn't raise an exception}}
    %tmp = pop.variant.create %err : !kgen.declref<@Error> -> !pop.variant<@Error, !lit.none>
    lit.return %tmp : !pop.variant<@Error, !lit.none>
    lit.try.yield
  } else {
    // CHECK: %[[V:.*]] = kgen.param.constant: !lit.none
    // CHECK-NEXT: %[[R:.*]] = pop.variant.create %[[V]]
    // CHECK-NEXT: kgen.return %[[R]]
    %none = kgen.param.constant: !lit.none = <#lit.none>
    %tmp2 = pop.variant.create %none : !lit.none -> !pop.variant<@Error, !lit.none>
    lit.return %tmp2 : !pop.variant<@Error, !lit.none>
    lit.try.yield
  } finally {
    lit.try.yield
  }
  // CHECK: kgen.unreachable
  // expected-warning @+1 {{unreachable code after try statement that doesn't fall through}}
  kgen.param.constant: <>(!kgen.declref<@Error>) throws -> !pop.variant<@Error, index> = <@throws>
  lit.end_func
}

// CHECK-LABEL: lit.func @suppressed_try
lit.func @suppressed_try() throws -> !lit.none {
  lit.try {
    lit.try.yield
  } except (%err: !kgen.declref<@Error>) {
    // CHECK: except (
    // CHECK-NEXT: kgen.unreachable
    %none = kgen.param.constant: !lit.none = <#lit.none>
    lit.try.yield
  } else {
    %none = kgen.param.constant: !lit.none = <#lit.none>
    lit.return %none : !lit.none
    lit.try.yield
  } finally {
    lit.try.yield
  } {"suppressWarnings" = true}
  // CHECK: kgen.unreachable
  // expected-warning @+1 {{unreachable code after try statement that doesn't fall through}}
  kgen.param.constant: <>(!kgen.declref<@Error>) throws -> !pop.variant<@Error, index> = <@throws>
  lit.end_func
}

// CHECK-LABEL: @parametric_throws
// CHECK-SAME:  -> !pop.variant<@Error, !lit.none>
lit.func @parametric_throws<fn: <>() throws -> !pop.variant<@Error, !lit.none>>() throws -> !pop.variant<@Error, !lit.none> {
  // CHECK-NEXT: %[[MAYBE_ERR:.*]] = kgen.call_param[<>() throws -> !pop.variant<@Error, !lit.none>: fn]()
  kgen.call_param[<>() throws -> !pop.variant<@Error, !lit.none>: fn]()

  %0 = kgen.param.constant: !lit.none = <#lit.none>
  %tmp2 = pop.variant.create %0 : !lit.none -> !pop.variant<@Error, !lit.none>
  lit.return %tmp2 : !pop.variant<@Error, !lit.none>
  lit.end_func
}

// CHECK-LABEL: lit.file_module @Module
lit.file_module @Module {
  // CHECK-LABEL: lit.struct.decl @Struct
  lit.struct.decl @Struct {
    // CHECK-NEXT: field x : !kgen.signature<() throws -> !pop.variant<@Error, !lit.none>>
    lit.struct.field x : !kgen.signature<() throws -> !pop.variant<@Error, !lit.none>>

    // CHECK-LABEL: lit.func @throws
    lit.func @throws(%self: !kgen.declref<@Module::@Struct>) throws -> !pop.variant<@Error, !lit.none> {
      // CHECK-NEXT: !kgen.signature<() throws -> !pop.variant<@Error, !lit.none>>
      %x = lit.struct.extract %self[x] : !kgen.signature<() throws -> !pop.variant<@Error, !lit.none>>
        from !kgen.declref<@Module::@Struct>
      %0 = kgen.param.constant: !lit.none = <#lit.none>
      %tmp2 = pop.variant.create %0 : !lit.none -> !pop.variant<@Error, !lit.none>
      lit.return %tmp2 : !pop.variant<@Error, !lit.none>
      lit.end_func
    }
  }
}

// CHECK-LABEL: lit.func @coroutine() async -> index
lit.func @coroutine() async -> index {
  %idx0 = index.constant 0
  // CHECK: return %idx0
  lit.return %idx0 : index
  lit.end_func
}

// CHECK-LABEL: lit.func @call_coroutine
// CHECK-SAME: coro: <>() async -> !lit.none
// CHECK-SAME: ) async -> !lit.none
lit.func @call_coroutine<coro: <>() async -> !lit.none>() async -> !lit.none {
  // CHECK-NEXT: lit.async.call[<>() async -> !lit.none: coro]()
  lit.async.call[<>() async -> !lit.none: coro]()
  %0 = kgen.param.constant: !lit.none = <#lit.none>
  lit.return %0 :  !lit.none
  lit.end_func
}

// CHECK-LABEL: lit.func @throwing_coro
// CHECK-SAME: -> !pop.variant<@Error, index>
lit.func @throwing_coro<cond: i1, a>(%err: !kgen.declref<@Error>) async|throws -> !pop.variant<@Error, index> {
  %c = kgen.param.constant: i1 = <cond>
  hlcf.if %c {
    // CHECK: %[[A:.*]] = kgen.param.constant = <a>
    %a = kgen.param.constant = <a>
    // CHECK-NEXT: %[[RESULT:.*]] = pop.variant.create %[[A]]
    // CHECK-NEXT: kgen.return %[[RESULT]]
    %tmp = pop.variant.create %a : index -> !pop.variant<@Error, index>
    lit.return %tmp : !pop.variant<@Error, index>
    hlcf.yield
  } else {
    hlcf.yield
  }
  // CHECK: %[[ERR:.*]] = pop.variant.create %err
  // CHECK-NEXT: kgen.return %[[ERR]]

  %tmp2 = pop.variant.create %err : !kgen.declref<@Error> -> !pop.variant<@Error, index>
  lit.return %tmp2 : !pop.variant<@Error, index>
  lit.end_func
}

// CHECK-LABEL: lit.func @call_throwing_coro({{.*}}) throws|async ->
lit.func @call_throwing_coro(%err: !kgen.declref<@Error>) async|throws -> !pop.variant<@Error, !lit.none> {
  // CHECK-NEXT: callee: <>(!kgen.declref<@Error>) throws|async -> !pop.variant<@Error, index>
  // CHECK-SAME: = <@throwing_coro<:i1 1, 0>>
  kgen.param.declare callee: <>(!kgen.declref<@Error>) async|throws -> !pop.variant<@Error, index>
    = <@throwing_coro<:i1 1, 0>>
  // CHECK: lit.async.call[<>(!kgen.declref<@Error>) throws|async -> !pop.variant<@Error, index>: callee](%err)
  %hdl = lit.async.call[<>(!kgen.declref<@Error>) async|throws -> !pop.variant<@Error, index> : callee](%err)

  %0 = kgen.param.constant: !lit.none = <#lit.none>
  %tmp2 = pop.variant.create %0 : !lit.none -> !pop.variant<@Error, !lit.none>
  lit.return %tmp2 : !pop.variant<@Error, !lit.none>
  lit.end_func
}


//===----------------------------------------------------------------------===//
// Nested Functions
//===----------------------------------------------------------------------===//

// CHECK-LABEL: lit.struct.decl @StructWithNestedFn<a_param>
lit.struct.decl @StructWithNestedFn<a_param> {
  // CHECK-NEXT: lit.func @topLevelFunction<b_param>() -> index
  lit.func @topLevelFunction<b_param>() -> index {
    %a = lit.varlet.decl "a", var = true, synth=false : <index>
    %idx0 = index.constant 0
    pop.store %idx0, %a : !kgen.pointer<index>

    // CHECK: kgen.param.declare.region nestedFunction = () -> index
    lit.func nestedFunction() -> index {
      // CHECK-NEXT: pop.load %a
      %0 = pop.load %a : !kgen.pointer<index>
      lit.return %0 : index
      lit.end_func
    }
    // CHECK: kgen.param.declare b: () -> index = <nestedFunction>
    kgen.param.declare b: () -> index = <nestedFunction>

    // CHECK: kgen.param.declare.region paramNestedFunc = <c_param -> d_param>()
    lit.func paramNestedFunc<c_param -> d_param>() {
      // CHECK-NEXT: kgen.param.result_bind<c_param>
      lit.param_return<c_param>
      lit.return
      lit.end_func
    }
    // CHECK: kgen.param.declare c: <[] -> index>() -> () = <bind_signature(:<index -> index>() -> () paramNestedFunc, 2)>
    kgen.param.declare c: <[] -> index>() -> () = <bind_signature(:<index -> index>() -> () paramNestedFunc, 2)>

    %idx0_0 = index.constant 0
    lit.return %idx0_0 : index
    lit.end_func
  }
}

// CHECK-LABEL: lit.func @topFunc
lit.func @topFunc() -> !lit.none {
  // CHECK: kgen.param.declare.region midFunc
  lit.func midFunc() -> !lit.none {
    // CHECK: kgen.param.declare.region botFunc
    lit.func botFunc() -> !lit.none {
      %0 = kgen.param.constant: !lit.none = <#lit.none>
      lit.return %0 :  !lit.none
      lit.end_func
    }
    // CHECK: declare bot: () -> !lit.none = <botFunc>
    kgen.param.declare bot: () -> !lit.none = <botFunc>
    %0 = kgen.param.constant: !lit.none = <#lit.none>
    lit.return %0 :  !lit.none
    lit.end_func
  }
  // CHECK: declare mid: () -> !lit.none = <midFunc>
  kgen.param.declare mid: () -> !lit.none = <midFunc>
  %0 = kgen.param.constant: !lit.none = <#lit.none>
  lit.return %0 :  !lit.none
  lit.end_func
}

// CHECK-LABEL: lit.func @return_after_return
lit.func @return_after_return() -> !lit.none {
  %0 = kgen.param.constant: !lit.none = <#lit.none>
  // CHECK: kgen.return %0 : !lit.none
  lit.return %0 : !lit.none
  %1 = kgen.param.constant: i1 = <1>  // expected-warning {{unreachable code after return statement}}
  hlcf.if %1 {
    %2 = kgen.param.constant: !lit.none = <#lit.none>
    lit.return %2 : !lit.none
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

// CHECK-LABEL: lit.func @coroutine
lit.func @coroutine2() async -> index {
  %0 = index.constant 0
  hlcf.loop {
    lit.return %0 : index
    lit.break  // expected-warning {{unreachable code after return statement}}
    hlcf.continue
  }
  // CHECK: kgen.unreachable
  lit.end_func
}

// CHECK-LABEL: lit.func @bubble_result_params
lit.func @bubble_result_params<cond: i1, cond2: i1 -> r0, r1: dtype>() {
  // CHECK: kgen.param.if <0>
  // CHECK-NEXT: kgen.unreachable
  // CHECK-NEXT: } else {
  // CHECK-NEXT: kgen.param.yield
  kgen.param.if <0> {
    kgen.param.yield
  } else {
    kgen.param.yield
  }

  // CHECK: kgen.param.if <cond ->
  kgen.param.if <cond> {
    // CHECK-NEXT: result_bind<1, :dtype si8>
    lit.param_return<1, :dtype si8>
    // CHECK-NEXT: kgen.return
    kgen.return
  // CHECK: else
  } else {
    // CHECK: kgen.param.if <cond2 -> *"(branch_result_0)", *"(branch_result_1)": dtype>
    kgen.param.if <cond2> {
      // CHECK-NEXT: result_bind<2, :dtype si16>
      lit.param_return<2, :dtype si16>
      kgen.param.yield
    // CHECK: else
    } else {
      // CHECK-NEXT: result_bind<3, :dtype si32>
      lit.param_return<3, :dtype si32>
      kgen.param.yield
    }
    // CHECK: result_bind<*"(branch_result_0)", :dtype *"(branch_result_1)">
    kgen.param.yield
  }
  // CHECK: result_bind<*"(branch_result_2)", :dtype *"(branch_result_3)">
  lit.return
  lit.end_func
}

// CHECK-LABEL: lit.func @result_params_fallthrough
lit.func @result_params_fallthrough<() -> r0>() -> !lit.none {
  // CHECK: %0 = kgen.param.constant: !lit.none
  // CHECK: kgen.param.result_bind<1>
  // CHECK: kgen.return %0
  lit.param_return<1>
  %0 = kgen.param.constant: !lit.none = <#lit.none>
  lit.return %0 :  !lit.none
  lit.end_func
}


// CHECK-LABEL: lit.func @pointlessTry
lit.func @pointlessTry() -> !lit.none {
  lit.try { // expected-warning {{try body doesn't raise an exception}}
    lit.try.yield
  } except (%err: !kgen.declref<@Error>) {
    lit.try.yield
  } else {
    lit.try.yield
  } finally {
    lit.try.yield
  }
  %0 = kgen.param.constant: !lit.none = <#lit.none>
  lit.return %0 :  !lit.none
  lit.end_func
}

// CHECK-LABEL: lit.func @reraise_in_try
lit.func @reraise_in_try(%err: !kgen.declref<@Error>) {
  // CHECK-NEXT: lit.try
  lit.try {
    // CHECK-NEXT: lit.try
    lit.try {
      // CHECK-NEXT: lit.try.raise %err
      lit.raise %err : <@Error>
      lit.try.yield
    // CHECK-NEXT: except
    } except (%reraise: !kgen.declref<@Error>) {
      // CHECK-NEXT: lit.try.raise %arg0
      lit.raise %reraise : <@Error>
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
  } except (%arg0: !kgen.declref<@Error>) {
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
  // CHECK-NEXT: hlcf.loop
  hlcf.loop {
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
    hlcf.break
  }
  // CHECK: return %arg1, %arg2
  kgen.return %arg1, %arg2 : i32, i64
}

// CHECK-LABEL: lit.func @try_finally_return
lit.func @try_finally_return(%arg0: index, %arg1: index, %arg2: i1) -> index {
  // CHECK-NEXT: hlcf.loop
  hlcf.loop {
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
    hlcf.break
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

// CHECK-LABEL: lit.func @throwing_func
lit.func @throwing_func<T: type>() throws -> !pop.variant<@Error, T> {
  %0 = lit.struct.create() : () -> !kgen.declref<@Error>
  // CHECK: %1 = pop.variant.create %0 : !kgen.declref<@Error> -> !pop.variant<@Error, T>
  // CHECK: lit.error_return %1 : <@Error, T>
  lit.raise %0 : <@Error>
  lit.end_func
}

// CHECK-LABEL: lit.func @try_in_loop
lit.func @try_in_loop(%arg0: i1) {
  hlcf.loop {
    hlcf.if %arg0 {
      hlcf.yield
    } else {
      hlcf.break
    }
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
    hlcf.continue
  }
  // CHECK: after.loop
  "after.loop"() : () -> ()
  kgen.return
}

// CHECK-LABEL: lit.func @recurse
// CHECK-SAME (%x: !pop.scalar<index>) -> !pop.scalar<index> {
// CHECK-NEXT: %0 = kgen.call @recurse(%x) : (!pop.scalar<index>) -> !pop.scalar<index>
// CHECK-NEXT: kgen.return %0 : !pop.scalar<index>
// CHECK-NEXT:}
lit.func @recurse(%x: !pop.scalar<index>) -> !pop.scalar<index> {
  %0 = kgen.call @recurse(%x) : (!pop.scalar<index>) -> !pop.scalar<index>
  lit.return %0 : !pop.scalar<index>
  lit.end_func
}

// CHECK-LABEL: lit.func @coroutine_await
lit.func @coroutine_await(%arg0: i1) {
  // CHECK-NEXT: pop.coroutine.await
  pop.coroutine.await {
    hlcf.if %arg0 {
      // CHECK: kgen.return
      lit.return
      hlcf.yield
    } else {
      hlcf.yield
    }
    // CHECK: pop.coroutine.await.end
    pop.coroutine.await.end
  }
  lit.return
  lit.end_func
}
