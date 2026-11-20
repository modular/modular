; RUN: llvm-module-split %s | FileCheck %s --check-prefix=CLONE
; RUN: llvm-module-split -outline-shared-subtrees %s | FileCheck %s --check-prefix=OUTLINE

; Two anchors reach @shared_root, so by default its whole subtree is cloned
; into both partitions and optimized in each.

; CLONE-LABEL: [LLVM Module Split: submodule 0]
; CLONE-DAG: define internal void @shared_root
; CLONE-DAG: define internal void @leaf1
; CLONE-DAG: define internal void @leaf2
; CLONE-LABEL: [LLVM Module Split: submodule 1]
; CLONE-DAG: define internal void @shared_root
; CLONE-DAG: define internal void @leaf1
; CLONE-DAG: define internal void @leaf2

; With outlining it is emitted once, promoted to weak ODR, and the partitions
; that referenced it are left with a declaration.

; OUTLINE-LABEL: [LLVM Module Split: submodule 0]
; OUTLINE: define weak_odr dso_local void @shared_root
; OUTLINE-DAG: define internal void @leaf1
; OUTLINE-DAG: define internal void @leaf2
; OUTLINE-LABEL: [LLVM Module Split: submodule 1]
; OUTLINE: declare void @shared_root()
; OUTLINE-LABEL: [LLVM Module Split: submodule 2]
; OUTLINE: declare void @shared_root()

define void @entry_a() {
  call void @shared_root()
  call void @always()
  ret void
}

define void @entry_b() {
  call void @shared_root()
  call void @always()
  ret void
}

define internal void @shared_root() {
  call void @leaf1()
  call void @leaf2()
  ret void
}

; An `alwaysinline` function is never outlined: a partition left holding only
; a declaration would have an inline the AlwaysInliner cannot satisfy. It stays
; cloned under both prefixes.

; CLONE-DAG: define internal void @always
; OUTLINE-DAG: define internal void @always
define internal void @always() alwaysinline {
  ret void
}

define internal void @leaf1() {
  %1 = add i64 1, 2
  %2 = add i64 %1, 3
  %3 = add i64 %2, 4
  %4 = add i64 %3, 5
  %5 = add i64 %4, 6
  %6 = add i64 %5, 7
  %7 = add i64 %6, 8
  %8 = add i64 %7, 9
  ret void
}

define internal void @leaf2() {
  %1 = add i64 1, 2
  %2 = add i64 %1, 3
  %3 = add i64 %2, 4
  %4 = add i64 %3, 5
  %5 = add i64 %4, 6
  %6 = add i64 %5, 7
  %7 = add i64 %6, 8
  %8 = add i64 %7, 9
  ret void
}
