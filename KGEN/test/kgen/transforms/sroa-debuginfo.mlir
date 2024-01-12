
// RUN: kgen-opt -sroa -split-input-file %s | FileCheck %s

!subroutine = !debuginfo.subroutine<() -> (): DW_CC_normal>
!member0 = !debuginfo.member<first: index>
!member1 = !debuginfo.member<second: index>
!struct = !debuginfo.struct<Foo(!member0, !member1)>
!ptr = !debuginfo.ti.ptr<!struct>
#subprogram = #debuginfo.subprogram<name = <"__next__">> : !subroutine
#local_variable = #debuginfo.local_variable<scope = #subprogram, name = "self"> : !ptr

#fileLoc = loc("foo.mlir":0:0)
#loc = loc(fused<#subprogram>[#fileLoc])

// CHECK-DAG: ![[INDEX:.*]] = !debuginfo.unresolved<index>
// CHECK-DAG: ![[STRUCT:.*]] = !debuginfo.struct<Foo(!{{.*}}, !{{.*}})>
// CHECK-DAG: ![[INDEX_PTR:.*]] = !debuginfo.ti.ptr<index>
// CHECK-DAG: ![[STRUCT_PTR:.*]] = !debuginfo.ti.ptr<![[STRUCT]]>

// CHECK-DAG: #[[IRVAL:.*]] = #debuginfo.expr.irvalue : ![[INDEX_PTR]]
// CHECK-DAG: #[[DEREF:.*]] = #debuginfo.expr.deref<#[[IRVAL]]> : ![[INDEX]]
// CHECK-DAG: #[[AGG0:.*]] = #debuginfo.expr.agg<#[[DEREF]], 0> : ![[STRUCT]]
// CHECK-DAG: #[[AGG1:.*]] = #debuginfo.expr.agg<#[[DEREF]], 1> : ![[STRUCT]]
// CHECK-DAG: #[[REF0:.*]] = #debuginfo.expr.refof<#[[AGG0]]> : ![[STRUCT_PTR]]
// CHECK-DAG: #[[REF1:.*]] = #debuginfo.expr.refof<#[[AGG1]]> : ![[STRUCT_PTR]]

// CHECK-DAG: #[[VAR:.*]] = #debuginfo.local_variable<{{.*}}, name = "self"> : ![[STRUCT_PTR]]

// CHECK-LABEL: @sroa_valueop
kgen.func @sroa_valueop() {
  // CHECK-NEXT: %0 = pop.stack_allocation 1 x index
  // CHECK-NEXT: %1 = pop.stack_allocation 1 x index
  %0 = pop.stack_allocation 1 x !kgen.struct<(index, index)> loc(#loc)
  // CHECK-NEXT: debuginfo.value #[[VAR]] #[[REF0]] = %0 : !kgen.pointer<index>
  // CHECK-NEXT: debuginfo.value #[[VAR]] #[[REF1]] = %1 : !kgen.pointer<index>
  debuginfo.value #local_variable = %0 : !kgen.pointer<struct<(index, index)>> loc(#loc)
  // CHECK-NEXT: kgen.return
  kgen.return loc(#loc)
} loc(#loc)

// -----

#sp = #debuginfo.subprogram<name = <"max">> : !debuginfo.subroutine<() -> (): DW_CC_normal>
!member0 = !debuginfo.member<first: index>
!member1 = !debuginfo.member<second: index>
!struct = !debuginfo.struct<Foo(!member0, !member1)>
#local_variable = #debuginfo.local_variable<scope = #sp, name = "x"> : !struct

#loc = loc(fused<#sp>["foo.mojo":0:0])

// CHECK-DAG: ![[INDEX:.*]] = !debuginfo.unresolved<index>
// CHECK-DAG: ![[STRUCT:.*]] = !debuginfo.struct<Foo(!{{.*}}, !{{.*}})>

// CHECK-DAG: #[[IRVAL:.*]] = #debuginfo.expr.irvalue : ![[INDEX]]
// CHECK-DAG: #[[AGG0:.*]] = #debuginfo.expr.agg<#[[IRVAL]], 0> : ![[STRUCT]]
// CHECK-DAG: #[[AGG1:.*]] = #debuginfo.expr.agg<#[[IRVAL]], 1> : ![[STRUCT]]

// CHECK-DAG: #[[VAR:.*]] = #debuginfo.local_variable<{{.*}}, name = "x"> : ![[STRUCT]]

// CHECK-LABEL: @load_debug_var
kgen.func @load_debug_var(%arg0: !kgen.struct<(index, index)>) {
  // CHECK-COUNT-2: pop.stack_allocation 1 x index
  %0 = pop.stack_allocation 1 x struct<(index, index)> loc(#loc)
  pop.store %arg0, %0 : !kgen.pointer<struct<(index, index)>> loc(#loc)
  %1 = pop.load %0 : !kgen.pointer<struct<(index, index)>> loc(#loc)
  // CHECK: [[VALUE0:%.*]] = pop.load
  // CHECK-NEXT: debuginfo.value #[[VAR]] #[[AGG0]] = [[VALUE0]]
  // CHECK: [[VALUE1:%.*]] = pop.load
  // CHECK-NEXT: debuginfo.value #[[VAR]] #[[AGG1]] = [[VALUE1]]
  debuginfo.value #local_variable = %1 : !kgen.struct<(index, index)> loc(#loc)
  kgen.return loc(#loc)
} loc(#loc)
