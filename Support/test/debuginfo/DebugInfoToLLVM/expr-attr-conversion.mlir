// RUN: support-dialect-opt %s -convert-debuginfo-to-llvm | FileCheck %s

#file = #debuginfo.file<"foo.c" in "/mlir/">
#compile_unit = #debuginfo.compile_unit<
  sourceLanguage = DW_LANG_Mojo,
  file = #file,
  producer = "MLIR",
  isOptimized = true,
  emissionKind = Full
>
#subprogram = #debuginfo.subprogram<
  compileUnit = #compile_unit,
  scope = #file,
  name = <"foo">,
  linkageName = "foo",
  file = #file,
  line = 10,
  scopeLine = 10,
  subprogramFlags = Definition
> : !debuginfo.subroutine<() -> (): DW_CC_normal>
// CHECK-DAG: #[[LOCALVAR:.*]] = #llvm.di_local_variable<{{.*}}"foo"
#local_variable = #debuginfo.local_variable<
  scope = #subprogram,
  name = "foo",
  file = #file,
  line = 10,
  arg = 0,
  alignInBits = 32
> : !debuginfo.unresolved<i32>
// CHECK-DAG: #[[LOCALVAR1:.*]] = #llvm.di_local_variable<{{.*}}"foo1"
#local_variable1 = #debuginfo.local_variable<
  scope = #subprogram,
  name = "foo1",
  file = #file,
  line = 11,
  arg = 0,
  alignInBits = 32
> : !debuginfo.unresolved<i32>
// COM: This will get removed as LLVM does not support implicit pointer yet.
// COM: CHECK-DAG: #[[LOCALVAR_PTR:.*]] = #llvm.di_local_variable<{{.*}}"fooptr"
#local_variable_ptr = #debuginfo.local_variable<
  scope = #subprogram,
  name = "fooptr",
  file = #file,
  line = 10,
  arg = 0,
  alignInBits = 64
> : !debuginfo.ptr<i32 {sizeInBits = 64, alignInBits = 64}>

#trivial_expr = #debuginfo.expr.irvalue : !debuginfo.unresolved<i32>
#refof_expr = #debuginfo.expr.refof<#trivial_expr> : !debuginfo.ptr<i32 {sizeInBits = 64, alignInBits = 64}>
#deref_expr = #debuginfo.expr.deref<#debuginfo.expr.irvalue : !debuginfo.ptr<i32 {sizeInBits = 64, alignInBits = 64}>> : !debuginfo.unresolved<i32>

func.func @foo(%arg0: i32, %arg1: i32, %arg2: !llvm.ptr) {
  // CHECK: llvm.intr.dbg.value #[[LOCALVAR]] =
  debuginfo.value #local_variable #trivial_expr = %arg0 : i32
  // COM: This will get removed as LLVM does not support implicit pointer yet.
  // COM: CHECK: llvm.intr.dbg.value #[[LOCALVAR_PTR]] #llvm.di_expression<[DW_OP_LLVM_implicit_pointer]>
  debuginfo.value #local_variable_ptr #refof_expr = %arg1 : i32
  // COM: This will get converted to declare because local_variable1 only has one debug value.
  // CHECK: llvm.intr.dbg.declare #[[LOCALVAR1]] #llvm.di_expression<[DW_OP_deref]>
  debuginfo.value #local_variable1 #deref_expr = %arg2 : !llvm.ptr
  // COM: This expr will be kept as a value since #local_variable is referenced multiple times.
  // CHECK: llvm.intr.dbg.value #[[LOCALVAR]] #llvm.di_expression<[DW_OP_deref]>
  debuginfo.value #local_variable #deref_expr = %arg2 : !llvm.ptr
  return
}
