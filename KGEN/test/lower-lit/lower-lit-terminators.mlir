// RUN: kgen-opt %s -lower-lit-terminators -verify-parameters -split-input-file | FileCheck %s

lit.struct.decl @Error {}

// CHECK-LABEL: lit.struct.decl @SomeStruct
lit.struct.decl @SomeStruct {
  // CHECK-LABEL: lit.func @dead_returns
  lit.func @dead_returns(%c: i1, %a: i32, %b: i32) -> i32 {
    // CHECK: hlcf.if %c
    hlcf.if %c {
      // CHECK-NEXT: kgen.return %b : i32
      lit.return %b: i32
      lit.return %a: i32
      hlcf.yield
    // CHECK-NEXT: else
    } else {
      hlcf.yield
    }
    // CHECK: kgen.return %a : i32
    lit.return %a : i32
    lit.return %b : i32
    lit.end_func
  // CHECK-NEXT: }
  }
}

// CHECK-LABEL: lit.file_module @FileModule
lit.file_module @FileModule {
  // CHECK-LABEL: lit.struct.decl @SomeStruct
  lit.struct.decl @SomeStruct {
    // CHECK-LABEL: lit.func @try_and_raise
    // CHECK-SAME: ) -> !pop.variant<@Error, i32>
    lit.func @try_and_raise(%a: i32, %b: !kgen.declref<@Error>) throws -> i32 {
      // CHECK-NEXT: lit.try
      lit.try {
        // CHECK-NEXT: lit.try.raise %b
        lit.raise %b : !kgen.declref<@Error>
        lit.try.yield
      // CHECK-NEXT: except (%arg0:
      } except (%err: !kgen.declref<@Error>) {
        // CHECK-NEXT: %[[R:.*]] = pop.variant.create %arg0
        // CHECK-NEXT: kgen.return %[[R]]
        lit.raise %err : !kgen.declref<@Error>
        lit.try.yield
      // CHECK-NEXT: else
      } else {
        // CHECK-NEXT: %[[R:.*]] = pop.variant.create
        // CHECK-NEXT: kgen.return %[[R]]
        lit.return %a : i32
        lit.try.yield
      // CHECK-NEXT: }
      }
      // CHECK-NEXT: %[[R:.*]] = pop.variant.create %b
      // CHECK-NEXT: kgen.return %[[R]]
      lit.raise %b : !kgen.declref<@Error>
      lit.return %a : i32
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
        lit.continue
        hlcf.yield
      // CHECK-NEXT: else
      } else {
        // CHECK-NEXT: hlcf.continue
        lit.continue
        lit.break
        hlcf.yield
      // CHECK-NEXT: }
      }
      // CHECK-NEXT: kgen.return
      lit.return
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
  // CHECK-NEXT: %0 = kgen.param.constant: !lit.none = <#lit.none>
  // CHECK-NEXT: kgen.return %0
  lit.end_func
}


// CHECK-LABEL: @no_return_throws
// CHECK-SAME ) -> !pop.variant<@Error, !lit.none>
lit.func @no_return_throws() throws -> !lit.none {
  // CHECK-NEXT: %0 = kgen.param.constant: !lit.none = <#lit.none>
  // CHECK-NEXT: %1 = pop.variant.create %0
  // CHECK-NEXT: kgen.return %1
  lit.end_func
}

// CHECK-LABEL: lit.func @throws
// CHECK-SAME: ) -> !pop.variant<@Error, index>
lit.func @throws(%e: !kgen.declref<@Error>) throws -> index {
  // CHECK-NEXT: pop.variant.create %e
  lit.raise %e : !kgen.declref<@Error>
  lit.end_func
}

// CHECK-LABEL: lit.func @ref
// CHECK-SAME: !kgen.signature<() -> !pop.variant<@Error, !lit.none>>
lit.func @ref(%e: !kgen.declref<@Error>,
              %f: !kgen.signature<() throws -> !lit.none>) throws -> !lit.none {
  lit.try {
    // CHECK: %[[MAYBE_ERR:.*]] = kgen.call @throws
    // CHECK-NEXT: %[[IS_ERR:.*]] = pop.variant.is !kgen.declref<@Error>, %[[MAYBE_ERR]]
    // CHECK-NEXT: %[[VAL:.*]] = hlcf.if %[[IS_ERR]]
    // CHECK-NEXT:   %[[UNWRAP:.*]] = pop.variant.get %[[MAYBE_ERR]] : !pop.variant<@Error, index> as !kgen.declref<@Error>
    // CHECK-NEXT:   lit.try.raise %[[UNWRAP]]
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   %[[UNWRAP:.*]] = pop.variant.get %[[MAYBE_ERR]] : !pop.variant<@Error, index> as index
    // CHECK-NEXT:   hlcf.yield %[[UNWRAP]]
    kgen.call @throws(%e) : (!kgen.declref<@Error>) throws -> index
    lit.try.yield
  } except (%err: !kgen.declref<@Error>) {
    // CHECK: %[[R:.*]] = pop.variant.create %arg0
    // CHECK-NEXT: kgen.return %[[R]]
    lit.raise %err : !kgen.declref<@Error>
    lit.try.yield
  } else {
    // CHECK: %[[V:.*]] = kgen.param.constant: !lit.none
    // CHECK-NEXT: %[[R:.*]] = pop.variant.create %[[V]]
    // CHECK-NEXT: kgen.return %[[R]]
    %none = kgen.param.constant: !lit.none = <#lit.none>
    lit.return %none : !lit.none
    lit.try.yield
  }
  // CHECK: constant: (!kgen.declref<@Error>) -> !pop.variant<@Error, index> = <@throws>
  kgen.param.constant: <>(!kgen.declref<@Error>) throws -> index = <@throws>
  lit.end_func
}

// CHECK-LABEL: @parametric_throws
// CHECK-SAME: fn: () -> !pop.variant<@Error, !lit.none>
lit.func @parametric_throws<fn: <>() throws -> !lit.none>() throws -> !lit.none {
  // CHECK-NEXT: %[[MAYBE_ERR:.*]] = kgen.call_param[() -> !pop.variant<@Error, !lit.none>: fn]()
  // CHECK-NEXT: pop.variant.is !kgen.declref<@Error>, %[[MAYBE_ERR]]
  kgen.call_param[<>() throws -> !lit.none: fn]()
  lit.end_func
}

// CHECK-LABEL: lit.file_module @Module
lit.file_module @Module {
  // CHECK-LABEL: lit.struct.decl @Struct
  lit.struct.decl @Struct {
    // CHECK-NEXT: field x : !kgen.signature<() -> !pop.variant<@Error, !lit.none>>
    lit.struct.field x : !kgen.signature<() throws -> !lit.none>

    // CHECK-LABEL: lit.func @throws
    lit.func @throws(%self: !kgen.declref<@Module::@Struct>) throws -> !lit.none {
      // CHECK-NEXT: !kgen.signature<() -> !pop.variant<@Error, !lit.none>> from
      %x = lit.struct.extract %self[x] : !kgen.signature<() throws -> !lit.none>
        from !kgen.declref<@Module::@Struct>
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
  lit.end_func
}

// CHECK-LABEL: lit.func @throwing_coro
// CHECK-SAME: ) async -> !pop.variant<@Error, index>
lit.func @throwing_coro<cond: i1, a>(%err: !kgen.declref<@Error>) async|throws -> index {
  %c = kgen.param.constant: i1 = <cond>
  hlcf.if %c {
    // CHECK: %[[A:.*]] = kgen.param.constant = <a>
    %a = kgen.param.constant = <a>
    // CHECK-NEXT: %[[RESULT:.*]] = pop.variant.create %[[A]]
    // CHECK-NEXT: kgen.return %[[RESULT]]
    lit.return %a : index
    hlcf.yield
  } else {
    hlcf.yield
  }
  // CHECK: %[[ERR:.*]] = pop.variant.create %err
  // CHECK-NEXT: kgen.return %[[ERR]]
  lit.raise %err : !kgen.declref<@Error>
  lit.end_func
}

// CHECK-LABEL: lit.func @call_throwing_coro({{.*}}) async ->
lit.func @call_throwing_coro(%err: !kgen.declref<@Error>) async|throws -> !lit.none {
  // CHECK-NEXT: callee: <>(!kgen.declref<@Error>) async -> !pop.variant<@Error, index>
  // CHECK-SAME: = <@throwing_coro<cond: i1 = 1, a = 0>>
  kgen.param.declare callee: <>(!kgen.declref<@Error>) async|throws -> index
    = <@throwing_coro<cond: i1 = 1, a = 0>>
  // CHECK: lit.async.call[<>(!kgen.declref<@Error>) async -> !pop.variant<@Error, index>: callee](%err)
  %hdl = lit.async.call[<>(!kgen.declref<@Error>) async|throws -> index: callee](%err)
  lit.end_func
}

//===----------------------------------------------------------------------===//
// Nested Functions
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: lit.struct.decl @StructWithNestedFn<a_param>
lit.struct.decl @StructWithNestedFn<a_param> {
  // CHECK-NEXT: lit.func @topLevelFunction<b_param>() -> index
  lit.func @topLevelFunction<b_param>() -> index {
    %a = lit.varlet.decl "a", var = true : <index>
    %idx0 = index.constant 0
    pop.store %idx0, %a : !pop.pointer<index>

    // CHECK: kgen.param.declare.region nestedFunction = () -> index
    lit.func nestedFunction() -> index {
      // CHECK-NEXT: pop.load %a
      %0 = pop.load %a : !pop.pointer<index>
      lit.return %0 : index
      lit.end_func
    }
    // CHECK: kgen.param.declare b: () -> index = <nestedFunction>
    kgen.param.declare b: () -> index = <nestedFunction>

    // CHECK: kgen.param.declare.region paramNestedFunc = <b_param -> c_param>()
    lit.func paramNestedFunc<b_param -> c_param>() {
      // CHECK-NEXT: kgen.param.result_bind<b_param>
      lit.param_return<b_param>
      lit.return
      lit.end_func
    }
    // CHECK: kgen.param.declare c: <() -> c_param>() -> () = <bind_signature(:<b_param -> c_param>() -> () paramNestedFunc, 2)>
    kgen.param.declare c: <() -> c_param>() -> () = <bind_signature(:<b_param -> c_param>() -> () paramNestedFunc, 2)>

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
      lit.end_func
    }
    // CHECK: declare bot: () -> !lit.none = <botFunc>
    kgen.param.declare bot: () -> !lit.none = <botFunc>
    lit.end_func
  }
  // CHECK: declare mid: () -> !lit.none = <midFunc>
  kgen.param.declare mid: () -> !lit.none = <midFunc>
  lit.end_func
}

// CHECK-LABEL: lit.func @return_after_return
lit.func @return_after_return() -> !lit.none {
  %0 = kgen.param.constant: !lit.none = <#lit.none>
  // CHECK: kgen.return %0 : !lit.none
  lit.return %0 : !lit.none
  %1 = kgen.param.constant: i1 = <1>
  hlcf.if %1 {
    %2 = kgen.param.constant: !lit.none = <#lit.none>
    lit.return %2 : !lit.none
    hlcf.yield
  } else {
    hlcf.yield
  }
  lit.end_func
}

// -----

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
  // CHECK: %0 = kgen.static.undef : index
  // CHECK-NEXT: return %0
  lit.end_func
}

// CHECK-LABEL: lit.func @if_true_return
lit.func @if_true_return() -> index {
  %0 = index.constant 0
  %true = index.bool.constant true
  hlcf.if %true {
    lit.return %0 : index
    hlcf.yield
  } else {
    hlcf.yield
  }
  // CHECK: kgen.static.undef
  lit.end_func
}

// CHECK-LABEL: lit.func @while_true
lit.func @while_true() -> index {
  hlcf.loop {
    %true = index.bool.constant true
    hlcf.if %true {
      hlcf.continue
    } else {
      hlcf.yield
    }
    hlcf.break
  }
  // CHECK: kgen.static.undef
  lit.end_func
}

lit.struct.decl @Error {}

// CHECK-LABEL: lit.func @if_false_raise
lit.func @if_false_raise() throws -> index {
  %false = index.bool.constant false
  hlcf.if %false {
    hlcf.yield
  } else {
    %err = lit.struct.create () : () -> !kgen.declref<@Error>
    lit.raise %err : <@Error>
    hlcf.yield
  }
  // CHECK: %0 = kgen.static.undef
  // CHECK-NEXT: %1 = pop.variant.create %0 : index -> !pop.variant<@Error, index>
  // CHECK-NEXT: return %1
  lit.end_func
}

// CHECK-LABEL: lit.func @raise_raise
lit.func @raise_raise() throws -> index {
  lit.try {
    %err = lit.struct.create () : () -> !kgen.declref<@Error>
    lit.raise %err : <@Error>
    lit.try.yield
  } except (%err: !kgen.declref<@Error>) {
    lit.raise %err : <@Error>
    lit.try.yield
  } else {
    lit.try.yield
  }
  // CHECK: kgen.static.undef
  lit.end_func
}

// CHECK-LABEL: lit.func @coroutine
lit.func @coroutine() async -> index {
  %0 = index.constant 0
  hlcf.loop {
    lit.return %0 : index
    hlcf.break
  }
  // CHECK: kgen.static.undef
  lit.end_func
}

// CHECK-LABEL: lit.func @bubble_result_params
lit.func @bubble_result_params<() -> r0, r1: dtype>() {
  // CHECK: kgen.param.if
  kgen.param.if <1> {
    kgen.param.yield
  } else {
    kgen.param.yield
  }

  // CHECK: kgen.param.if <1 ->
  kgen.param.if <1> {
    // CHECK-NEXT: result_bind<1, :dtype si8>
    lit.param_return<1, :dtype si8>
    // CHECK-NEXT: kgen.return
    kgen.return
  // CHECK: else
  } else {
    // CHECK: kgen.param.if <1 -> *"(branch_result_0)", *"(branch_result_1)": dtype>
    kgen.param.if <1> {
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
