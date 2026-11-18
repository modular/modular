// The --kgen-rotate-loops-max-ops flag is only registered in
// !MODULAR_PRODUCTION builds (see Mojo/include/Mojo/ToolCommon/CLOptions.h).
// UNSUPPORTED: production
// RUN: kgen-opt %s -rotate-loops -split-input-file --kgen-rotate-loops-max-ops=1 | FileCheck %s

// One op ahead of the test is within the limit, so the loop rotates.
// CHECK-LABEL: kgen.func @one_op_rotated(%arg0: !kgen.pointer<index>, %arg1: index) {
// CHECK-NEXT:   %idx0 = index.constant 0
// CHECK-NEXT:   %idx1 = index.constant 1
// CHECK-NEXT:   pop.store   %arg1, %arg0 : !kgen.pointer<index>
// CHECK-NEXT:   hlcf.loop (%arg2 = %arg1 : index) {
// CHECK-NEXT:     %0 = index.cmp eq(%arg2, %idx0)
// CHECK-NEXT:     %1 = pop.cast_from_builtin %0 : i1 to !kgen.scalar<bool>
// CHECK-NEXT:     hlcf.if %1 {
// CHECK-NEXT:       hlcf.break
// CHECK-NEXT:     } else {
// CHECK-NEXT:       hlcf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     %2 = index.sub %arg2, %idx1
// CHECK-NEXT:     pop.store   %2, %arg0 : !kgen.pointer<index>
// CHECK-NEXT:     hlcf.continue %2 : index
// CHECK-NEXT:   }
// CHECK-NEXT:   hlcf.return
// CHECK-NEXT: }
kgen.func @one_op_rotated(%p: !kgen.pointer<index>, %n: index) {
  %idx0 = index.constant 0
  %idx1 = index.constant 1

  hlcf.loop (%arg0 = %n : index) {
    pop.store %arg0, %p : !kgen.pointer<index>
    %0 = index.cmp eq(%arg0, %idx0)
    %c0 = pop.cast_from_builtin %0 : i1 to !kgen.scalar<bool>
    hlcf.if %c0 {
      hlcf.break
    } else {
      hlcf.yield
    }
    %1 = index.sub %arg0, %idx1
    hlcf.continue %1 : index
  }
  hlcf.return
}

// -----

// Two are one too many, so the loop is left alone.
// CHECK-LABEL: kgen.func @two_ops_not_rotated(%arg0: !kgen.pointer<index>, %arg1: index) {
// CHECK-NEXT:   %idx0 = index.constant 0
// CHECK-NEXT:   %idx1 = index.constant 1
// CHECK-NEXT:   hlcf.loop (%arg2 = %arg1 : index) {
// CHECK-NEXT:     pop.store   %arg2, %arg0 : !kgen.pointer<index>
// CHECK-NEXT:     pop.store   %idx1, %arg0 : !kgen.pointer<index>
// CHECK-NEXT:     %0 = index.cmp eq(%arg2, %idx0)
// CHECK-NEXT:     %1 = pop.cast_from_builtin %0 : i1 to !kgen.scalar<bool>
// CHECK-NEXT:     hlcf.if %1 {
// CHECK-NEXT:       hlcf.break
// CHECK-NEXT:     } else {
// CHECK-NEXT:       hlcf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     %2 = index.sub %arg2, %idx1
// CHECK-NEXT:     hlcf.continue %2 : index
// CHECK-NEXT:   }
// CHECK-NEXT:   hlcf.return
// CHECK-NEXT: }
kgen.func @two_ops_not_rotated(%p: !kgen.pointer<index>, %n: index) {
  %idx0 = index.constant 0
  %idx1 = index.constant 1

  hlcf.loop (%arg0 = %n : index) {
    pop.store %arg0, %p : !kgen.pointer<index>
    pop.store %idx1, %p : !kgen.pointer<index>
    %0 = index.cmp eq(%arg0, %idx0)
    %c0 = pop.cast_from_builtin %0 : i1 to !kgen.scalar<bool>
    hlcf.if %c0 {
      hlcf.break
    } else {
      hlcf.yield
    }
    %1 = index.sub %arg0, %idx1
    hlcf.continue %1 : index
  }
  hlcf.return
}
