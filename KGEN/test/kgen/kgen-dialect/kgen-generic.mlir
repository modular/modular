// RUN: kgen-opt %s -mlir-print-op-generic | kgen-opt -mlir-print-op-generic | FileCheck %s

// CHECK: "kgen.generator"
// CHECK: constraints = #kgen<constraints[]>
// CHECK-SAME: signature = !kgen.signature<() -> ()>
kgen.generator @kernel() {
  kgen.return
}
