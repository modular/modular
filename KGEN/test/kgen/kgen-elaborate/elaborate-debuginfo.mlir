// RUN: kgen-opt %s -split-input-file -elaborate-generators="elaborate-debuginfo=true allow-multiple-primary-impls=true" -mlir-print-debuginfo | FileCheck %s

// CHECK-LABEL: kgen.func @loc_ref
kgen.generator @loc_ref() {
  kgen.param.if <0> {
    kgen.param.yield
  } else {
    kgen.param.declare A = <1>
    // CHECK: constant = <1> loc([[LOC:#.*]])
    kgen.param.constant = <1> loc(fused<#kgen.param.decl.ref<"A">:index>["a":0:0])
    kgen.param.yield
  }
  kgen.return
}

// CHECK: [[LOC]] = loc(fused<1 : index>

// -----

// The checks here are just doubling down on the same checks we already do in DebugInfo::ValueOp verification.

#file = #debuginfo.file<"test.mlir" in "">
!unresolved = !debuginfo.unresolved<!kgen.variadic<ty>>
#forkParamSp = #debuginfo.subprogram<file = #file, name = <"fork_param">> : !debuginfo.subroutine<(!unresolved) -> (): DW_CC_normal>
#local_variable = #debuginfo.local_variable<scope = #forkParamSp, name = "0"> : !unresolved

// CHECK-DAG: [[LOCALVAR2:#.*]] = #debuginfo.local_variable{{.*}}scope = [[SUBPROGRAM2:#.*]],
// CHECK-DAG: [[LOCALVAR3:#.*]] = #debuginfo.local_variable{{.*}}scope = [[SUBPROGRAM3:#.*]],

// CHECK-DAG: debuginfo.value [[LOCALVAR2]] = %arg0 : !kgen.variadic<index> loc([[LOC2:#.*]])
// CHECK-DAG: debuginfo.value [[LOCALVAR3]] = %arg0 : !kgen.variadic<index> loc([[LOC3:#.*]])

// CHECK-DAG: [[LOC2]] = loc(fused<[[SUBPROGRAM2]]>[{{.*}}])
// CHECK-DAG: [[LOC3]] = loc(fused<[[SUBPROGRAM3]]>[{{.*}}])

kgen.generator @fork_param<ty: type>(%x: !kgen.variadic<ty>) -> index {
  debuginfo.value #local_variable = %x : !kgen.variadic<ty> loc(#locX)
  kgen.param.if <0> {
    kgen.param.yield loc(#locVar)
  } else {
    kgen.param.fork N = <[2, 3]> loc(#locVar)
    %0 = kgen.param.constant = <N> loc(#locVar)
    kgen.return %0 : index loc(#locVar)
  } loc(#locVar)
  %1 = kgen.param.constant = <1> loc(#locVar)
  kgen.return %1 : index loc(#locVar)
} loc(#locFunc)

kgen.generator @driver() -> index {
  %0 = kgen.param.constant: index = <0>
  %1 = pop.variadic.create [%0, %0] : !kgen.variadic<index>
  %2 = kgen.call @fork_param<:type index>(%1) : (!kgen.variadic<index>) -> index
  kgen.return %2 : index
}

#locFunc = loc(fused<#forkParamSp>["test.mlir":0:0])
#locX = loc(fused<#forkParamSp>["test.mlir":1:0])
#locVar = loc(fused<#forkParamSp>["test.mlir":2:0])
