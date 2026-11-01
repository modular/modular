// RUN: kgen-translate -mlir-to-llvmir -split-input-file='// SP''LIT' \
// RUN:   -output-split-marker='; JOIN' %s | FileCheck %s

// CHECK: define void @a()
// CHECK-NEXT: ret void
// CHECK-NEXT: }
// CHECK: ; JOIN
// CHECK: define void @b()
llvm.func @a() {
  llvm.return
}

// SPLIT

llvm.func @b() {
  llvm.return
}
