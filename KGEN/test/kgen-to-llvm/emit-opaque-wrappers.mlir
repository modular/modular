// RUN: kgen-opt %s -split-input-file -convert-kgen-to-llvm='break-up-structs=kernel emit-opaque-wrappers' | FileCheck %s

// CHECK-LABEL: llvm.func @kernel(
// CHECK-SAME: attributes {opaque_wrapper = @kernel_opaque_wrapper}
kgen.kernel @kernel(%a: !meta.buffer<?, ?>, %i: i32) -> !meta.buffer<?, ?> {
  kgen.return %a : !meta.buffer<?, ?>
}

// CHECK: llvm.func @kernel_opaque_wrapper
// CHECK-SAME: %[[ARG0:.*]]: !llvm.ptr<struct<packed (i64, ptr, i8)>>, %[[I:.*]]: i32,
// CHECK-SAME: %[[RES:.*]]: !llvm.ptr<struct<packed (i64, ptr, i8)>>) {
// CHECK: %[[PTR0:.*]] = llvm.getelementptr %arg0[0]
// CHECK: %[[CALL0:.*]] = llvm.load %0 : !llvm.ptr<i64>
// CHECK: %[[PTR1:.*]] = llvm.getelementptr %arg0[1]
// CHECK: %[[CALL1:.*]] = llvm.load %2 : !llvm.ptr<ptr>
// CHECK: %[[PTR2:.*]] = llvm.getelementptr %arg0[2]
// CHECK: %[[CALL2:.*]] = llvm.load %4 : !llvm.ptr<i8>
// CHECK: %[[RES0:.*]] = llvm.getelementptr %arg2[0]
// CHECK: %[[RES1:.*]] = llvm.getelementptr %arg2[1]
// CHECK: %[[RES2:.*]] = llvm.getelementptr %arg2[2]
// CHECK: llvm.call @kernel(%[[CALL0]], %[[CALL1]], %[[CALL2]], %[[I]], %[[RES0]], %[[RES1]], %[[RES2]])
// CHECK: llvm.return

// -----

// CHECK-LABEL: llvm.func @kernel(
llvm.func @kernel() {
  llvm.return
}

// CHECK: llvm.func @kernel_opaque_wrapper() {
// CHECK: llvm.call @kernel()
// CHECK: llvm.return

// -----

// CHECK-LABEL: llvm.func @kernel(
llvm.func @kernel() -> i32 {
  %0 = llvm.mlir.constant(0 : i32) : i32
  llvm.return %0 : i32
}

// CHECK: llvm.func @kernel_opaque_wrapper() -> i32
// CHECK: %[[RES:.*]] = llvm.call @kernel()
// CHECK: llvm.return %[[RES]]

// -----

!nestedStruct = !llvm.struct<(i8, struct<(i16, struct<(i32)>)>)>

// CHECK-LABEL: llvm.func @kernel(
kgen.kernel @kernel(%a: !nestedStruct) -> !nestedStruct {
  kgen.return %a : !nestedStruct
}

// CHECK: llvm.func @kernel_opaque_wrapper
// CHECK-SAME: %[[IN:.*]]: !llvm.ptr<struct<packed (i8, struct<packed (i16, struct<packed (i32)>)>)>>,
// CHECK-SAME: %[[OUT:.*]]: !llvm.ptr<struct<packed (i8, struct<packed (i16, struct<packed (i32)>)>)>>
// CHECK: getelementptr %[[IN]][0]
// CHECK: getelementptr %[[IN]][1, 0]
// CHECK: getelementptr %[[IN]][1, 1, 0]
