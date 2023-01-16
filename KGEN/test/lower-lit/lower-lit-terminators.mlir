// RUN: kgen-opt %s -lower-lit-terminators | FileCheck %s

kgen.struct.decl @Error {}

// CHECK-LABEL: kgen.struct.decl @SomeStruct
kgen.struct.decl @SomeStruct {
  // CHECK-LABEL: lit.func @dead_returns
  lit.func @dead_returns(%c: i1, %a: i32, %b: i32) -> i32 {
    // CHECK: hlcf.if %c
    hlcf.if %c {
      // CHECK-NEXT: hlcf.return %b : i32
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
  // CHECK-LABEL: kgen.struct.decl @SomeStruct
  kgen.struct.decl @SomeStruct {
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
        // CHECK-NEXT: hlcf.return %[[R]]
        lit.raise %err : !kgen.declref<@Error>
        lit.try.yield
      // CHECK-NEXT: else
      } else {
        // CHECK-NEXT: %[[R:.*]] = pop.variant.create
        // CHECK-NEXT: hlcf.return %[[R]]
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
      // CHECK-NEXT: hlcf.return
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
lit.func @result_parameters<() -> i32, i64>(%c: i1) {
  // CHECK: hlcf.if
  hlcf.if %c {
    // CHECK-NEXT: hlcf.return
    lit.return<:i32 1, :i64 2>
    hlcf.yield
  } else {
    hlcf.yield
  }
  // CHECK: kgen.return<:i32 1, :i64 2>
  lit.return<:i32 1, :i64 2>
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
// CHECK-SAME: !kgen.signature<[], [], () -> !pop.variant<@Error, !lit.none>>
lit.func @ref(%e: !kgen.declref<@Error>,
              %f: !kgen.signature<[], [], () throws -> !lit.none>) throws -> !lit.none {
  lit.try {
    // CHECK: %[[MAYBE_ERR:.*]] = kgen.call @throws
    // CHECK-NEXT: lit.unwrap_or_propagate %[[MAYBE_ERR]]
    kgen.call @throws(%e) : (!kgen.declref<@Error>) throws -> index
    lit.try.yield
  } except (%err: !kgen.declref<@Error>) {
    // CHECK: %[[R:.*]] = pop.variant.create %arg0
    // CHECK-NEXT: hlcf.return %[[R]]
    lit.raise %err : !kgen.declref<@Error>
    lit.try.yield
  } else {
    // CHECK: %[[V:.*]] = kgen.param.constant: !lit.none
    // CHECK-NEXT: %[[R:.*]] = pop.variant.create %[[V]]
    // CHECK-NEXT: hlcf.return %[[R]]
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
  // CHECK-NEXT: lit.unwrap_or_propagate %[[MAYBE_ERR]]
  kgen.call_param[<>() throws -> !lit.none: fn]()
  lit.end_func
}

// CHECK-LABEL: lit.file_module @Module
lit.file_module @Module {
  // CHECK-LABEL: kgen.struct.decl @Struct
  kgen.struct.decl @Struct {
    // CHECK-NEXT: field x : !kgen.signature<[], [], () -> !pop.variant<@Error, !lit.none>>
    kgen.struct.field x : !kgen.signature<[], [], () throws -> !lit.none>

    // CHECK-LABEL: lit.func @throws
    lit.func @throws(%self: !kgen.declref<@Module::@Struct>) throws -> !lit.none {
      // CHECK-NEXT: !kgen.signature<[], [], () -> !pop.variant<@Error, !lit.none>> from
      %x = kgen.struct.extract %self[x] : !kgen.signature<[], [], () throws -> !lit.none>
        from !kgen.declref<@Module::@Struct>
      lit.end_func
    }
  }
}
