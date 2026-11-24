// RUN: not kgen-translate -mlir-to-llvmir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DISALLOWED
// RUN: kgen-translate -mlir-to-llvmir -allow-unregistered-dialect %s \
// RUN:   | FileCheck %s

// DISALLOWED: error: #"test"<"attr"> : 'none' attribute created with unregistered dialect
// CHECK: define void @a()
module attributes {test.unregistered = #test.attr} {
  llvm.func @a() {
    llvm.return
  }
}
