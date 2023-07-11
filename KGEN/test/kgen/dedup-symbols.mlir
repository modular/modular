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

kgen.func export @kernel() {
  kgen.call @dependency() : () -> ()
  kgen.call @dependency() : () -> ()
  kgen.return
}

// CHECK: --- 'kernel' finished
