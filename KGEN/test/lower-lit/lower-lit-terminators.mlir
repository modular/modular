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
    lit.func @try_and_raise(%a: i32, %b: !kgen.declref<@Error>) throws -> !pop.variant<@Error, i32> {
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
        %r = pop.variant.create %a : i32 -> !pop.variant<@Error, i32>
        // CHECK-NEXT: hlcf.return %[[R]]
        lit.return %r : !pop.variant<@Error, i32>
        lit.try.yield
      // CHECK-NEXT: }
      }
      // CHECK-NEXT: %[[R:.*]] = pop.variant.create %b
      // CHECK-NEXT: kgen.return %[[R]]
      lit.raise %b : !kgen.declref<@Error>
      %r = pop.variant.create %a : i32 -> !pop.variant<@Error, i32>
      lit.return %r : !pop.variant<@Error, i32>
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
lit.func @no_return_throws() throws -> !pop.variant<@Error, !lit.none> {
  // CHECK-NEXT: %0 = kgen.param.constant: !lit.none = <#lit.none>
  // CHECK-NEXT: %1 = pop.variant.create %0
  // CHECK-NEXT: kgen.return %1
  lit.end_func
}
