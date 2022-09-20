// RUN: kgen-execute -execute -func="kernel:():%t.o" %s | FileCheck %s

kgen.func @nested1() {
  kgen.return
}

kgen.func @nested2() {
  kgen.call @nested1() : () -> ()
  kgen.return
}

kgen.func public @dependency() {
  kgen.call @nested2() : () -> ()
  kgen.call @nested1() : () -> ()
  kgen.return
}

kgen.func public @kernel() {
  kgen.call @dependency() : () -> ()
  kgen.call @dependency() : () -> ()
  kgen.return
}

// CHECK: --- 'kernel' finished
