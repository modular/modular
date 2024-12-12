// RUN: kgen-opt %s -mlir-print-op-generic | kgen-opt -mlir-print-op-generic | FileCheck %s

// CHECK: "kgen.generator"
// CHECK: signatureGenerator = !kgen.generator<<> !kgen.new_signature<() -> ()>>
kgen.generator @kernel() {
  kgen.return
}
