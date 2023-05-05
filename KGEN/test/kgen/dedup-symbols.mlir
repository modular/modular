// RUN: kgen -execute -func="kernel:()" %s | FileCheck %s

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

kgen.func @kernel() {
  kgen.call @dependency() : () -> ()
  kgen.call @dependency() : () -> ()
  kgen.return
}

kgen.export @kernel

// CHECK: --- 'kernel' finished
