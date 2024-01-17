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
    // CHECK-SAME:  -> !kgen.variant<@Error, i32>
    lit.func @try_and_raise(%a: i32, %b: !kgen.declref<@Error>) throws -> !kgen.variant<@Error, i32> {
      // CHECK-NEXT: lit.try
      lit.try {
        // CHECK-NEXT: lit.try.raise %b
        lit.raise %b : !kgen.declref<@Error>
        lit.try.yield
      // CHECK-NEXT: except (%arg0:
      } except (%err: !kgen.declref<@Error>) {
        // CHECK-NEXT: %[[R:.*]] = kgen.variant.create %arg0
        // CHECK-NEXT: kgen.return %[[R]]
        %tmp2 = kgen.variant.create %err, 0 : <@Error, i32>
        lit.return %tmp2 : !kgen.variant<@Error, i32>
        lit.try.yield
      // CHECK-NEXT: else
      } else {
        // CHECK-NEXT: kgen.unreachable

        // expected-warning @+1 {{'else' logic in 'try' is unreachable}}
        %tmp3 = kgen.variant.create %a, 1 : <@Error, i32>
        lit.return %tmp3 : !kgen.variant<@Error, i32>
        lit.try.yield
      // CHECK-NEXT: }
      } finally {
        lit.try.yield
      }

      // CHECK-NEXT: kgen.unreachable
      // expected-warning @+1 {{unreachable code after try statement that doesn't fall through}}
      %tmp1 = kgen.variant.create %a, 1 : <@Error, i32>
      lit.return %tmp1 : !kgen.variant<@Error, i32>
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
lit.func @if_false_raise() throws -> !kgen.variant<@Error, index> {
  %false = index.bool.constant false
  hlcf.if %false {
    hlcf.yield
  } else {
    %err = lit.struct.create () : () -> !kgen.declref<@Error>
    %tmp = kgen.variant.create %err, 0 : <@Error, index>
    lit.return %tmp : !kgen.variant<@Error, index>
    hlcf.yield
  }
  lit.end_func
}

// CHECK-LABEL: lit.func @raise_raise
lit.func @raise_raise() throws -> !kgen.variant<@Error, index> {
  // CHECK: lit.try
  lit.try {
    %err = lit.struct.create () : () -> !kgen.declref<@Error>
    // CHECK: lit.try.raise
    lit.raise %err : <@Error>
    lit.try.yield
  // CHECK-NEXT: except
  } except (%err: !kgen.declref<@Error>) {
    %tmp = kgen.variant.create %err, 0 : <@Error, index>
    // CHECK: kgen.return
    lit.return %tmp : !kgen.variant<@Error, index>
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
// CHECK-SAME ) -> !kgen.variant<@Error, none>
lit.func @no_return_throws() throws -> !kgen.variant<@Error, none> {
  %0 = kgen.param.constant: none = <#kgen.none>
  %tmp2 = kgen.variant.create %0, 1 : <@Error, none>
  lit.return %tmp2 : !kgen.variant<@Error, none>
  // CHECK-NEXT: %none = kgen.param.constant: none = <#kgen.none>
  // CHECK-NEXT: %0 = kgen.variant.create %none
  // CHECK-NEXT: kgen.return %0
  lit.end_func
}

// CHECK-LABEL: lit.func @throws
// CHECK-SAME:  -> !kgen.variant<@Error, index>
lit.func @throws(%e: !kgen.declref<@Error>) throws -> !kgen.variant<@Error, index> {
  // CHECK-NEXT: kgen.variant.create %e
  %tmp = kgen.variant.create %e, 0 : <@Error, index>
  lit.return %tmp : !kgen.variant<@Error, index>
  lit.end_func
}

// CHECK-LABEL: lit.func @ref
// CHECK-SAME: !kgen.signature<() throws -> !kgen.variant<@Error, none>>
lit.func @ref(%e: !kgen.declref<@Error>,
              %f: !kgen.signature<() throws -> !kgen.variant<@Error, none>>) throws -> !kgen.variant<@Error, none> {
  lit.try {
    // CHECK: = kgen.call @throws
    kgen.call @throws(%e) : !lit.signature<("e": !kgen.declref<@Error>) throws -> !kgen.variant<@Error, index>>
    lit.try.yield
  } except (%err: !kgen.declref<@Error>) {
    // CHECK: except (
    // CHECK-NEXT: kgen.unreachable
    // expected-warning @+1 {{'except' logic is unreachable, try doesn't raise an exception}}
    %tmp = kgen.variant.create %err, 0 : <@Error, none>
    lit.return %tmp : !kgen.variant<@Error, none>
    lit.try.yield
  } else {
    // CHECK: %[[V:.*]] = kgen.param.constant: none
    // CHECK-NEXT: %[[R:.*]] = kgen.variant.create %[[V]]
    // CHECK-NEXT: kgen.return %[[R]]
    %none = kgen.param.constant: none = <#kgen.none>
    %tmp2 = kgen.variant.create %none, 1 : <@Error, none>
    lit.return %tmp2 : !kgen.variant<@Error, none>
    lit.try.yield
  } finally {
    lit.try.yield
  }
  // CHECK: kgen.unreachable
  // expected-warning @+1 {{unreachable code after try statement that doesn't fall through}}
  kgen.param.constant: (!kgen.declref<@Error>) throws -> !kgen.variant<@Error, index> = <@throws>
  lit.end_func
}

// CHECK-LABEL: lit.func @suppressed_try
lit.func @suppressed_try() throws -> !kgen.none {
  lit.try {
    lit.try.yield
  } except (%err: !kgen.declref<@Error>) {
    // CHECK: except (
    // CHECK-NEXT: kgen.unreachable
    %none = kgen.param.constant: none = <#kgen.none>
    lit.try.yield
  } else {
    %none = kgen.param.constant: none = <#kgen.none>
    lit.return %none : !kgen.none
    lit.try.yield
  } finally {
    lit.try.yield
  } {"suppressWarnings" = true}
  // CHECK: kgen.unreachable
  // expected-warning @+1 {{unreachable code after try statement that doesn't fall through}}
  kgen.param.constant: (!kgen.declref<@Error>) throws -> !kgen.variant<@Error, index> = <@throws>
  lit.end_func
}

// CHECK-LABEL: @parametric_throws
// CHECK-SAME:  -> !kgen.variant<@Error, none>
lit.func @parametric_throws<fn: () throws -> !kgen.variant<@Error, none>>() throws -> !kgen.variant<@Error, none> {
  // CHECK-NEXT: %[[MAYBE_ERR:.*]] = kgen.call_param[() throws -> !kgen.variant<@Error, none>: fn]()
  kgen.call_param[() throws -> !kgen.variant<@Error, none>: fn]()

  %0 = kgen.param.constant: none = <#kgen.none>
  %tmp2 = kgen.variant.create %0, 1 : <@Error, none>
  lit.return %tmp2 : !kgen.variant<@Error, none>
  lit.end_func
}

// CHECK-LABEL: lit.file_module @Module
lit.file_module @Module {
  // CHECK-LABEL: lit.struct.decl @Struct
  lit.struct.decl @Struct {
    // CHECK-NEXT: field x : !kgen.signature<() throws -> !kgen.variant<@Error, none>>
    lit.struct.field x : !kgen.signature<() throws -> !kgen.variant<@Error, none>>

    // CHECK-LABEL: lit.func @throws
    lit.func @throws(%self: !kgen.declref<@Module::@Struct>) throws -> !kgen.variant<@Error, none> {
      // CHECK-NEXT: !kgen.signature<() throws -> !kgen.variant<@Error, none>>
      %x = lit.struct.extract %self[x] : !kgen.signature<() throws -> !kgen.variant<@Error, none>>
        from !kgen.declref<@Module::@Struct>
      %0 = kgen.param.constant: none = <#kgen.none>
      %tmp2 = kgen.variant.create %0, 1 : <@Error, none>
      lit.return %tmp2 : !kgen.variant<@Error, none>
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
// CHECK-SAME: coro: () async -> !kgen.none
// CHECK-SAME: ) async -> !kgen.none
lit.func @call_coroutine<coro: () async -> !kgen.none>() async -> !kgen.none {
  // CHECK-NEXT: lit.async.call[() async -> !kgen.none: coro]()
  lit.async.call[() async -> !kgen.none: coro]()
  %0 = kgen.param.constant: none = <#kgen.none>
  lit.return %0 :  !kgen.none
  lit.end_func
}

// CHECK-LABEL: lit.func @throwing_coro
// CHECK-SAME: -> !kgen.variant<@Error, index>
lit.func @throwing_coro<cond: i1, a>(%err: !kgen.declref<@Error>) async|throws -> !kgen.variant<@Error, index> {
  %c = kgen.param.constant: i1 = <cond>
  hlcf.if %c {
    // CHECK: %[[A:.*]] = kgen.param.constant = <a>
    %a = kgen.param.constant = <a>
    // CHECK-NEXT: %[[RESULT:.*]] = kgen.variant.create %[[A]]
    // CHECK-NEXT: kgen.return %[[RESULT]]
    %tmp = kgen.variant.create %a, 1 : <@Error, index>
    lit.return %tmp : !kgen.variant<@Error, index>
    hlcf.yield
  } else {
    hlcf.yield
  }
  // CHECK: %[[ERR:.*]] = kgen.variant.create %err
  // CHECK-NEXT: kgen.return %[[ERR]]

  %tmp2 = kgen.variant.create %err, 0 : <@Error, index>
  lit.return %tmp2 : !kgen.variant<@Error, index>
  lit.end_func
}

// CHECK-LABEL: lit.func @call_throwing_coro({{.*}}) throws|async ->
lit.func @call_throwing_coro(%err: !kgen.declref<@Error>) async|throws -> !kgen.variant<@Error, none> {
  // CHECK-NEXT: callee: !lit.signature<("err": !kgen.declref<@Error>) throws|async -> !kgen.variant<@Error, index>>
  // CHECK-SAME: = <@throwing_coro<:i1 1, 0>>
  kgen.param.declare callee: !lit.signature<("err": !kgen.declref<@Error>) async|throws -> !kgen.variant<@Error, index>>
    = <@throwing_coro<:i1 1, 0>>
  // CHECK: lit.async.call[!lit.signature<("err": !kgen.declref<@Error>) throws|async -> !kgen.variant<@Error, index>>: callee](%err)
  %hdl = lit.async.call[!lit.signature<("err": !kgen.declref<@Error>) async|throws -> !kgen.variant<@Error, index>> : callee](%err)

  %0 = kgen.param.constant: none = <#kgen.none>
  %tmp2 = kgen.variant.create %0, 1 : <@Error, none>
  lit.return %tmp2 : !kgen.variant<@Error, none>
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
  } except (%err: !kgen.declref<@Error>) {
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
  %true = index.bool.constant true

  // CHECK: hlcf.loop {
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

  // CHECK: hlcf.loop {
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

// CHECK-LABEL: lit.func @throwing_func
lit.func @throwing_func<T: type>() throws -> !kgen.variant<@Error, T> {
  %0 = lit.struct.create() : () -> !kgen.declref<@Error>
  // CHECK: %1 = kgen.variant.create %0, 0 : <@Error, T>
  // CHECK: lit.error_return %1 : <@Error, T>
  lit.raise %0 : <@Error>
  lit.end_func
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
      // CHECK-NEXT: hlcf.loop {
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
      %0 = lit.struct.create() : () -> !kgen.declref<@Error>
      lit.raise %0 : <@Error>
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
  // CHECK: } except (
  } except (%err: !kgen.declref<@Error>) {
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
