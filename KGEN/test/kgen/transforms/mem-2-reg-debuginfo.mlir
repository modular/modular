// RUN: kgen-opt -split-input-file -mem-2-reg -allow-unregistered-dialect -mlir-print-debuginfo %s | FileCheck %s

#callerSp = #debuginfo.subprogram<name = <"mem2reg_valueop">> : !debuginfo.subroutine<(index) -> (): DW_CC_normal>
#local_variable = #debuginfo.local_variable<scope = #callerSp, name = "0"> : !debuginfo.ti.ptr<index>

#loc0 = loc(fused<#callerSp>["foo.mlir":0:0])
#loc1 = loc(fused<#callerSp>["foo.mlir":1:0])
#loc2 = loc(fused<#callerSp>["foo.mlir":2:0])
#loc3 = loc(fused<#callerSp>["foo.mlir":3:0])

// CHECK-DAG: ![[INDEX_TYPE:.*]] = !debuginfo.unresolved<index>
// CHECK-DAG: ![[PTR_TYPE:.*]] = !debuginfo.ti.ptr<index>
// CHECK: #[[IRVALUE_EXPR:.*]] = #debuginfo.expr.irvalue : ![[INDEX_TYPE]]
// CHECK: #[[REFOF_EXPR:.*]] = #debuginfo.expr.refof<#[[IRVALUE_EXPR]]> : ![[PTR_TYPE]]

// CHECK-LABEL: @mem2reg_valueop_no_undef
kgen.func @mem2reg_valueop_no_undef(%arg0: index, %arg1: index) {
  // CHECK-NOT: kgen.undef : index
  // CHECK: debuginfo.value #local_variable #[[REFOF_EXPR]] = %arg0 : index loc(#[[LOC_STORE0:.*]])
  // CHECK: debuginfo.value #local_variable #[[REFOF_EXPR]] = %arg1 : index loc(#[[LOC_STORE1:.*]])
  %0 = pop.stack_allocation 1 x index loc(#loc0)
  debuginfo.value #local_variable = %0 : !kgen.pointer<index> loc(#loc0)
  pop.store %arg0, %0 : !kgen.pointer<index> loc(#loc1)
  pop.store %arg1, %0 : !kgen.pointer<index> loc(#loc2)
  kgen.return loc(#loc0)
} loc(#loc0)

// CHECK-LABEL: @mem2reg_valueop_with_initial_undef
kgen.func @mem2reg_valueop_with_initial_undef(%arg0: index, %arg1: index) -> index {
  // CHECK: %[[UNDEF_VAL:.*]] = kgen.undef : index loc(#[[LOC_UNDEF:.*]])
  // CHECK: debuginfo.value #local_variable #[[REFOF_EXPR]] = %[[UNDEF_VAL]] : index loc(#[[LOC_UNDEF]])
  // CHECK: debuginfo.value #local_variable #[[REFOF_EXPR]] = %arg0 : index loc(#[[LOC_STORE0:.*]])
  // CHECK: debuginfo.value #local_variable #[[REFOF_EXPR]] = %arg1 : index loc(#[[LOC_STORE1:.*]])
  %0 = pop.stack_allocation 1 x index loc(#loc0)
  debuginfo.value #local_variable = %0 : !kgen.pointer<index> loc(#loc0)
  %1 = pop.load %0 : !kgen.pointer<index> loc(#loc3) // loading undef
  pop.store %arg0, %0 : !kgen.pointer<index> loc(#loc1)
  pop.store %arg1, %0 : !kgen.pointer<index> loc(#loc2)
  kgen.return %1 : index loc(#loc0)
} loc(#loc0)

// CHECK: #[[LOC_STORE0_RAW:.*]] = loc("foo.mlir":1:0)
// CHECK: #[[LOC_STORE1_RAW:.*]] = loc("foo.mlir":2:0)
// CHECK: #[[LOC_UNDEF_RAW:.*]] = loc("foo.mlir":3:0)
// CHECK: #[[LOC_STORE0]] = loc(fused<{{.*}}>[#[[LOC_STORE0_RAW]]])
// CHECK: #[[LOC_STORE1]] = loc(fused<{{.*}}>[#[[LOC_STORE1_RAW]]])
// CHECK: #[[LOC_UNDEF]] = loc(fused<{{.*}}>[#[[LOC_UNDEF_RAW]]])
