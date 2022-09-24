// RUN: kgen-opt %s -mlir-print-op-generic | kgen-opt -mlir-print-op-generic | FileCheck %s

// CHECK: "kgen.generator"
// CHECK: constraints = #kgen<constraints[]>
// CHECK-SAME: paramDecls = #kgen<param.decls[]>
// CHECK-SAME: resultParamTypes = #kgen<type.array[]>
kgen.generator @kernel<>() {
  kgen.return
}
