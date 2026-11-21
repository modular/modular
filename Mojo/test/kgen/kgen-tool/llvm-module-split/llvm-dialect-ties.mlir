// RUN: llvm-module-split %s | FileCheck %s

// @foo and @bar call @baz. An exported function must stay unique, so the
// three share one slice.

// CHECK-LABEL: [LLVM Module Split: submodule 0]
// CHECK-DAG: define void @foo
// CHECK-DAG: define void @bar
// CHECK-DAG: define void @baz

llvm.func @foo() {
  llvm.call @baz() : () -> ()
  llvm.return
}

llvm.func @bar() {
  llvm.call @baz() : () -> ()
  llvm.return
}

llvm.func @baz() {
  llvm.return
}

// The users of a mutable global share a slice.

// CHECK-LABEL: [LLVM Module Split: submodule 1]
// CHECK-DAG: @counter = internal global i64 0
// CHECK-DAG: define void @inc_a
// CHECK-DAG: define void @inc_b

llvm.mlir.global internal @counter(0 : i64) : i64

llvm.func @inc_a() {
  %0 = llvm.mlir.addressof @counter : !llvm.ptr
  %1 = llvm.load %0 : !llvm.ptr -> i64
  %2 = llvm.add %1, %1 : i64
  llvm.store %2, %0 : i64, !llvm.ptr
  llvm.return
}

llvm.func @inc_b() {
  %0 = llvm.mlir.addressof @counter : !llvm.ptr
  %1 = llvm.load %0 : !llvm.ptr -> i64
  %2 = llvm.mul %1, %1 : i64
  llvm.store %2, %0 : i64, !llvm.ptr
  llvm.return
}

// A constant global ties nothing, so each user gets a slice and a copy.

// CHECK-LABEL: [LLVM Module Split: submodule 2]
// CHECK-DAG: @limit = internal constant i64 7
// CHECK-DAG: define i64 @read_a
// CHECK-LABEL: [LLVM Module Split: submodule 3]
// CHECK-DAG: @limit = internal constant i64 7
// CHECK-DAG: define i64 @read_b

llvm.mlir.global internal constant @limit(7 : i64) : i64

llvm.func @read_a() -> i64 {
  %0 = llvm.mlir.addressof @limit : !llvm.ptr
  %1 = llvm.load %0 : !llvm.ptr -> i64
  llvm.return %1 : i64
}

llvm.func @read_b() -> i64 {
  %0 = llvm.mlir.addressof @limit : !llvm.ptr
  %1 = llvm.load %0 : !llvm.ptr -> i64
  llvm.return %1 : i64
}

// CHECK-NOT: [LLVM Module Split: submodule 4]
