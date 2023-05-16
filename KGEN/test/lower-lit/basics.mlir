// RUN: kgen-opt -lower-lit -allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: kgen.generator @trivial_generator
// CHECK-SAME: (%[[ARG0:.*]]: si32) -> si32 {
// CHECK-NEXT:    kgen.return %[[ARG0]] : si32
// CHECK-NEXT:  }
lit.func @trivial_generator(%arg0: si32) -> si32 {
  kgen.return %arg0 : si32
}

// CHECK-LABEL: kgen.generator @varDecl
// CHECK-SAME:  (%[[ARG0:.*]]: index) -> index {
// CHECK-NEXT:    %[[VAR_A:.*]] = pop.stack_allocation 1 x index
// CHECK-NEXT:    kgen.return %[[ARG0]] : index
// CHECK-NEXT:  }

lit.func @varDecl(%arg0: index) -> index {
  %a = lit.varlet.decl "a", var = true, synth=false : <index>
  kgen.return %arg0 : index
}

// CHECK-LABEL: kgen.generator @letDecl(%arg0
// CHECK-NEXT:    kgen.return %arg0 : index
lit.func @letDecl(%arg0: index) -> index {
  %a = lit.letreg.decl "a" = %arg0 : index
  %b = lit.letreg.decl "b" = %a : index
  kgen.return %b : index
}

//===----------------------------------------------------------------------===//
// Aliases
//===----------------------------------------------------------------------===//

// CHECK-LABEL: kgen.generator @aliasFwdDecl()
// CHECK-NEXT: kgen.return
lit.func @aliasFwdDecl() {
  lit.alias.fwd.decl "xyz" : index
  kgen.return
}

//===----------------------------------------------------------------------===//
// Finally
//===----------------------------------------------------------------------===//

// CHECK-LABEL: kgen.generator @try_finally
lit.func @try_finally(%arg0: i1, %arg1: i32, %arg2: i64) -> (i32, i64) {
  // CHECK-NEXT: hlcf.loop
  hlcf.loop {
    // CHECK-NEXT: lit.try
    lit.try {
      // CHECK-NEXT: hlcf.if %arg0
      hlcf.if %arg0 {
        // CHECK: %[[R0:.*]] = kgen.undef : i32
        // CHECK: %[[R1:.*]] = kgen.undef : i64
        // CHECK: finalize %idx3, %[[R0]], %[[R1]]
        hlcf.break
      // CHECK-NEXT: else
      } else {
        // CHECK: finalize %idx2
        hlcf.continue
      }
      // CHECK: finalize %idx1, %arg1, %arg2
      kgen.return %arg1, %arg2 : i32, i64
    // CHECK-NEXT: except
    } except (%err: index) {
      // CHECK: finalize %idx0
      lit.try.yield
    // CHECK-NEXT: else
    } else {
      // CHECK: finalize %idx0
      lit.try.yield
    // CHECK-NEXT: finally {
    } finally {
    // CHECK-NEXT: ^bb0(%arg3: index, %arg4: i32, %arg5: i64):
      // CHECK-NEXT: clean.up
      "clean.up"() : () -> ()
      // CHECK-NEXT: hlcf.switch %arg3
      // CHECK-NEXT: default
      // CHECK-NEXT:   yield
      // CHECK:      case 0
      // CHECK-NEXT:   yield
      // CHECK:      case 1
      // CHECK-NEXT:   return %arg4, %arg5
      // CHECK:      case 2
      // CHECK-NEXT:   continue
      // CHECK:      case 3
      // CHECK-NEXT:   break
      // CHECK: yield
      lit.try.yield
    }
    // CHECK: break
    hlcf.break
  }
  kgen.return %arg1, %arg2 : i32, i64
}

// CHECK-LABEL: kgen.generator @try_finally_return
lit.func @try_finally_return(%arg0: index, %arg1: index, %arg2: i1) -> index {
  // CHECK-NEXT: hlcf.loop
  hlcf.loop {
    // CHECK-NEXT: lit.try
    lit.try {
      // CHECK-NEXT: hlcf.if %arg2
      hlcf.if %arg2 {
        // CHECK-NEXT: finalize
        hlcf.break
      // CHECK-NEXT: else
      } else {
        // CHECK-NEXT: finalize
        hlcf.continue
      }
      // CHECK: finalize
      kgen.return %arg0 : index
    } except (%err: index) {
      lit.try.yield
    } else {
      lit.try.yield
    // CHECK: finally {
    } finally {
      // CHECK-NEXT: return %arg1
      kgen.return %arg1 : index
    }
    hlcf.break
  }
  kgen.return %arg1 : index
}

// CHECK-LABEL: kgen.generator @nested_try_finally
lit.func @nested_try_finally() {
  // CHECK-NEXT: lit.try
  lit.try {
    // CHECK-NEXT: lit.try
    lit.try {
      // CHECK: finalize %idx1 {{.*}}finallyLabel = 0
      kgen.return
    } except (%err: index) {
      lit.try.yield
    } else {
      lit.try.yield
    // CHECK: finally {
    } finally {
    // CHECK-NEXT: ^bb0(%arg0: index):
      // CHECK-NEXT: clean.up
      "clean.up"() : () -> ()
      // CHECK: hlcf.switch %arg0
      // CHECK: case 1
      // CHECK-NEXT: %idx1 = index.constant 1
      // CHECK-NEXT: finalize %idx1 {{.*}}finallyLabel = 1
      // CHECK: lit.try.yield
      lit.try.yield
    // CHECK-NEXT: finallyLabel = 0
    }
    lit.try.yield
  } except (%err: index) {
    lit.try.yield
  } else {
    lit.try.yield
  // CHECK: finally {
  } finally {
  // CHECK-NEXT: ^bb0(%arg0: index):
    // CHECK-NEXT: clean.up
    "clean.up"() : () -> ()
    // CHECK-NEXT: hlcf.switch %arg0
    // CHECK: case 1
    // CHECK-NEXT: kgen.return
    // CHECK: lit.try.yield
    lit.try.yield
  // CHECK-NEXT: finallyLabel = 1
  }
  kgen.return
}
