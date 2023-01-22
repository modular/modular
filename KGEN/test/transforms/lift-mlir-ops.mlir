// RUN: kgen-opt %s -lift-mlir-ops | FileCheck %s

// CHECK-LABEL: kgen.generator @pop.array.get1
// CHECK-SAME: <*"index", size, type: type>(%arg0: !pop.array<size, type>) -> !kgen.paramref<type>
  // CHECK-NEXT: 0 = pop.array.get %arg0[*"index"] {{.*}} : !pop.array<size, type>
  // CHECK-NEXT: return %0 : !kgen.paramref<type>

// CHECK-LABEL: kgen.generator @index.cmp0
// CHECK-SAME: (%arg0: index, %arg1: index) -> i1
  // CHECK-NEXT: index.cmp slt(%arg0, %arg1)

// CHECK-LABEL: @mlirOperationExpr
kgen.generator @mlirOperationExpr() {
  // CHECK-NEXT: indexCmp1: (index, index) -> i1 = <@index.cmp0>
  kgen.param.declare indexCmp1: (index, index) -> i1 =
    <#kgen.param.mlir_op<"index.cmp", {pred = #index<cmp_predicate slt>}>>
  // CHECK-NEXT: indexCmp2: (index, index) -> i1 = <@index.cmp0>
  kgen.param.declare indexCmp2: (index, index) -> i1 =
    <#kgen.param.mlir_op<"index.cmp", {pred = #index<cmp_predicate slt>}>>
  // CHECK-NEXT: arrayGetParam: <*"index", size, type: type>(!pop.array<size, type>) -> !kgen.paramref<type> = <@pop.array.get1>
  kgen.param.declare arrayGetParam: <*"index", size, type: type>(!pop.array<size, type>) -> !kgen.paramref<type> =
    <#kgen.param.mlir_op<"pop.array.get", {}>>
  kgen.return
}
