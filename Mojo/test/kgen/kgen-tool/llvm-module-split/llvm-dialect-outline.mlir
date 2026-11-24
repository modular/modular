// RUN: llvm-module-split %s | FileCheck %s --check-prefix=CLONE
// RUN: llvm-module-split -outline-shared-subtrees %s | FileCheck %s --check-prefix=OUTLINE

// Two exported functions reach @shared_root, so by default each slice clones
// its subtree.

// CLONE-LABEL: [LLVM Module Split: submodule 0]
// CLONE-DAG: define void @entry_a
// CLONE-DAG: define internal void @shared_root
// CLONE-DAG: define internal void @leaf1
// CLONE-DAG: define internal void @leaf2
// CLONE-DAG: define internal void @always
// CLONE-LABEL: [LLVM Module Split: submodule 1]
// CLONE-DAG: define void @entry_b
// CLONE-DAG: define internal void @shared_root
// CLONE-DAG: define internal void @leaf1
// CLONE-DAG: define internal void @leaf2
// CLONE-DAG: define internal void @always

// With outlining, @shared_root and its leaves get a slice of their own, and
// @shared_root becomes weak_odr. The other slices declare it. @always is an
// `alwaysinline` function, so each slice keeps a copy.

// OUTLINE-LABEL: [LLVM Module Split: submodule 0]
// OUTLINE-DAG: define weak_odr dso_local void @shared_root
// OUTLINE-DAG: define internal void @leaf1
// OUTLINE-DAG: define internal void @leaf2
// OUTLINE-LABEL: [LLVM Module Split: submodule 1]
// OUTLINE-DAG: define void @entry_a
// OUTLINE-DAG: define internal void @always
// OUTLINE-DAG: declare void @shared_root(i64)
// OUTLINE-LABEL: [LLVM Module Split: submodule 2]
// OUTLINE-DAG: define void @entry_b
// OUTLINE-DAG: define internal void @always
// OUTLINE-DAG: declare void @shared_root(i64)

llvm.func @entry_a(%arg0: i64) {
  llvm.call @shared_root(%arg0) : (i64) -> ()
  llvm.call @always() : () -> ()
  llvm.return
}

llvm.func @entry_b(%arg0: i64) {
  llvm.call @shared_root(%arg0) : (i64) -> ()
  llvm.call @always() : () -> ()
  llvm.return
}

llvm.func internal @shared_root(%arg0: i64) {
  llvm.call @leaf1(%arg0) : (i64) -> ()
  llvm.call @leaf2(%arg0) : (i64) -> ()
  llvm.return
}

llvm.func internal @always() attributes {always_inline} {
  llvm.return
}

llvm.func internal @leaf1(%arg0: i64) {
  %0 = llvm.add %arg0, %arg0 : i64
  %1 = llvm.add %0, %arg0 : i64
  %2 = llvm.add %1, %arg0 : i64
  %3 = llvm.add %2, %arg0 : i64
  %4 = llvm.add %3, %arg0 : i64
  %5 = llvm.add %4, %arg0 : i64
  %6 = llvm.add %5, %arg0 : i64
  %7 = llvm.add %6, %arg0 : i64
  llvm.return
}

llvm.func internal @leaf2(%arg0: i64) {
  %0 = llvm.mul %arg0, %arg0 : i64
  %1 = llvm.mul %0, %arg0 : i64
  %2 = llvm.mul %1, %arg0 : i64
  %3 = llvm.mul %2, %arg0 : i64
  %4 = llvm.mul %3, %arg0 : i64
  %5 = llvm.mul %4, %arg0 : i64
  %6 = llvm.mul %5, %arg0 : i64
  %7 = llvm.mul %6, %arg0 : i64
  llvm.return
}
