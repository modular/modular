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

// CHECK-DAG: #[[STRUCT_TYPE:.*]] = #llvm.di_composite_type<{{.*}}name = "MyStruct",{{.*}}sizeInBits = 64, alignInBits = 32
!struct = !debuginfo.struct<MyStruct(
            !debuginfo.member<first: !debuginfo.unresolved<i8>>,
            !debuginfo.member<second: !debuginfo.unresolved<i32>>
          )>
// CHECK-DAG: #[[LOCALVAR_STRUCT:.*]] = #llvm.di_local_variable<{{.*}}name = "foostruct"{{.*}}type = #[[STRUCT_TYPE]]
#local_variable_struct = #debuginfo.local_variable<
  scope = #subprogram,
  name = "foostruct",
  file = #file,
  line = 10,
  arg = 0,
  alignInBits = 64
> : !struct

// CHECK-DAG: #[[OUTER_STRUCT_TYPE:.*]] = #llvm.di_composite_type<{{.*}}name = "MyOuterStruct",{{.*}}sizeInBits = 96, alignInBits = 32
!outer_struct = !debuginfo.struct<MyOuterStruct(
  !debuginfo.member<first: !debuginfo.unresolved<i32>>,
  !debuginfo.member<second: !struct>
)>
// CHECK-DAG: #[[LOCALVAR_OUTER_STRUCT:.*]] = #llvm.di_local_variable<{{.*}}name = "fooouterstruct"{{.*}}type = #[[OUTER_STRUCT_TYPE]]
#local_variable_outer_struct = #debuginfo.local_variable<
  scope = #subprogram,
  name = "fooouterstruct",
  file = #file,
  line = 10,
  arg = 0,
  alignInBits = 64
> : !outer_struct

// CHECK-DAG: #[[OUTER_STRUCT_TYPE_2:.*]] = #llvm.di_composite_type<{{.*}}name = "MyOuterStruct2",{{.*}}sizeInBits = 128, alignInBits = 32
!outer_struct_2 = !debuginfo.struct<MyOuterStruct2(
  !debuginfo.member<first: !debuginfo.unresolved<i32>>,
  !debuginfo.member<second: !outer_struct>
)>
// CHECK-DAG: #[[LOCALVAR_OUTER_STRUCT_2:.*]] = #llvm.di_local_variable<{{.*}}name = "fooouterstruct2"{{.*}}type = #[[OUTER_STRUCT_TYPE_2]]
#local_variable_outer_struct_2 = #debuginfo.local_variable<
  scope = #subprogram,
  name = "fooouterstruct2",
  file = #file,
  line = 10,
  arg = 0,
  alignInBits = 64
> : !outer_struct_2

#trivial_expr = #debuginfo.expr.irvalue : !debuginfo.unresolved<i32>
#refof_expr = #debuginfo.expr.refof<#trivial_expr> : !debuginfo.ptr<i32 {sizeInBits = 64, alignInBits = 64}>
#deref_expr = #debuginfo.expr.deref<#debuginfo.expr.irvalue : !debuginfo.ptr<i32 {sizeInBits = 64, alignInBits = 64}>> : !debuginfo.unresolved<i32>
#agg_expr = #debuginfo.expr.agg<#deref_expr, 1> : !struct
#outer_agg_expr = #debuginfo.expr.agg<#agg_expr, 1> : !outer_struct
#outer_agg_expr_2 = #debuginfo.expr.agg<#outer_agg_expr, 1> : !outer_struct_2

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
  // CHECK: llvm.intr.dbg.declare #[[LOCALVAR_STRUCT]] #llvm.di_expression<[DW_OP_deref, DW_OP_LLVM_fragment(32, 32)]>
  debuginfo.value #local_variable_struct #agg_expr = %arg2 : !llvm.ptr
  return
}

func.func @simplify(%arg0: !llvm.ptr) {
  // CHECK: #[[LOCALVAR_OUTER_STRUCT]] #llvm.di_expression<[DW_OP_deref, DW_OP_LLVM_fragment(64, 32)]>
  debuginfo.value #local_variable_outer_struct #outer_agg_expr = %arg0 : !llvm.ptr
  // CHECK: #[[LOCALVAR_OUTER_STRUCT_2]] #llvm.di_expression<[DW_OP_deref, DW_OP_LLVM_fragment(96, 32)]>
  debuginfo.value #local_variable_outer_struct_2 #outer_agg_expr_2 = %arg0 : !llvm.ptr
  return
}
