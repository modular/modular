// RUN: kgen %s -emit -d=%t -I %S/../kernels -o %t.o
// RUN: FileCheck %s < %t

kgen.include "library.mlir"

kgen.generator @run() {
  kgen.return
}

kgen.export [@run]

// CHECK: {{.*}}.o:
// CHECK-SAME: kernels/library.mlir
