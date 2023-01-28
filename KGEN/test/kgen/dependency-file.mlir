// RUN: kgen %s -emit -d=%t -I %S/../kernels -o %t.o
// RUN: FileCheck %s < %t

// Dependency file generation fails when artifacts are not saved in files.
// RUN: echo "" | not kgen - -emit -d=%t -o -

kgen.include "library.mlir"

kgen.generator @run() {
  kgen.return
}

kgen.export @run

// CHECK: {{.*}}.o:
// CHECK-SAME: kernels/library.mlir
