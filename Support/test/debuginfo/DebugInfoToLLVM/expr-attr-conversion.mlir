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
// CHECK: #[[LOCALVAR:.*]] = #llvm.di_local_variable<{{.*}}"foo"
#local_variable = #debuginfo.local_variable<
  scope = #subprogram,
  name = "foo",
  file = #file,
  line = 10,
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
  // CHECK: llvm.intr.dbg{{.*}} #[[LOCALVAR]] =
  debuginfo.value #local_variable #trivial_expr = %arg0 : i32
  // COM: This will get removed as LLVM does not support implicit pointer yet.
  // COM: CHECK: llvm.intr.dbg.value #[[LOCALVAR_PTR]] #llvm.di_expr<[4100]>
  debuginfo.value #local_variable_ptr #refof_expr = %arg1 : i32
  // CHECK: llvm.intr.dbg{{.*}} #[[LOCALVAR]] #llvm.di_expr<[6]>
  debuginfo.value #local_variable #deref_expr = %arg2 : !llvm.ptr
  return
}
