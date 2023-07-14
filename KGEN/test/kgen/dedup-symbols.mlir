// RUN: kgen -emit-llvm %s | FileCheck %s

// COM: This test checks that `sliceDependencies` doesn't copy the same function
// COM: more than once.

// CHECK: define internal void @dependency
// CHECK: define internal void @nested2
// CHECK: define internal void @nested1

kgen.func @nested1() {
  kgen.return
}

kgen.func @nested2() {
  kgen.call @nested1() : () -> ()
  kgen.return
}

kgen.func @dependency() {
  kgen.call @nested2() : () -> ()
  kgen.call @nested1() : () -> ()
  kgen.return
}

kgen.func export @kernel() {
  kgen.call @dependency() : () -> ()
  kgen.call @dependency() : () -> ()
  kgen.return
}
