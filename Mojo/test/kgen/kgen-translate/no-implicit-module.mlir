// RUN: kgen-translate -mlir-to-llvmir %s | FileCheck %s
// RUN: not kgen-translate -mlir-to-llvmir -no-implicit-module %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NO-IMPLICIT

// CHECK: define void @a()
// CHECK: define void @b()
// NO-IMPLICIT: error: source must contain a single top-level operation, found: 2
llvm.func @a() {
  llvm.return
}

llvm.func @b() {
  llvm.return
}
