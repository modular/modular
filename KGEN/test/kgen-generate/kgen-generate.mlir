// RUN: kgen-generate %s -library=%S/library.mlir | FileCheck %s

// CHECK-LABEL: kgen.generator @trivial_generator(%arg0: si32) -> si32 {
kgen.generator @trivial_generator(%arg0: si32) -> si32 {
  // CHECK-NEXT: kgen.return %arg0 : si32
  kgen.return %arg0 : si32
}

// Need a kgen.kernel declaration: the top level thing to produce.


