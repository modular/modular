// RUN: kgen-execute -execute -func="kernel:():%t.o" %s | FileCheck %s

kgen.func @nested() {
  kgen.return
}

kgen.func @dependency() {
  kgen.call @nested() : () -> ()
  kgen.return
}

kgen.func public @kernel() {
  kgen.call @dependency() : () -> ()
  kgen.call @dependency() : () -> ()
  kgen.return
}

// CHECK: --- 'kernel' finished
