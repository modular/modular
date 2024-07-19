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

#trivial_expr = #debuginfo.expr.irvalue : !debuginfo.unresolved<i32>
#refof_expr = #debuginfo.expr.refof<#trivial_expr> : !debuginfo.ptr<i32 {sizeInBits = 64, alignInBits = 64}>
#deref_expr = #debuginfo.expr.deref<#debuginfo.expr.irvalue : !debuginfo.ptr<i32 {sizeInBits = 64, alignInBits = 64}>> : !debuginfo.unresolved<i32>
#agg_expr = #debuginfo.expr.agg<#deref_expr, 1> : !struct

func.func @foo() {
  %v0 = llvm.mlir.constant(2: i32) : i32
  %v1 = llvm.mlir.constant(3: i32) : i32
  %v2 = llvm.inttoptr %v0 : i32 to !llvm.ptr

  // CHECK: %[[COUNT:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[ALLOC:.*]] = llvm.alloca %[[COUNT]] x i32 {alignment = 1 : i64} : (i32) -> !llvm.ptr
  // CHECK: %[[V0:.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK: llvm.store volatile %[[V0]], %[[ALLOC]] {alignment = 1 : i64} : i32, !llvm.ptr
  // CHECK: %[[V1:.*]] = llvm.mlir.constant(3 : i32) : i32
  // CHECK: llvm.intr.dbg.value #[[LOCALVAR]] #llvm.di_expression<[DW_OP_deref]> = %[[ALLOC]]
  debuginfo.value #local_variable #trivial_expr = %v0 : i32
  // COM: This will get removed as LLVM does not support implicit pointer yet.
  // COM: CHECK: llvm.intr.dbg.value #[[LOCALVAR_PTR]] #llvm.di_expression<[DW_OP_LLVM_implicit_pointer]>
  debuginfo.value #local_variable_ptr #refof_expr = %v1 : i32
  // COM: This expr will be removed since it begins with a deref.
  // CHECK: llvm.intr.dbg.declare #[[LOCALVAR1]] =
  debuginfo.value #local_variable1 #deref_expr = %v2 : !llvm.ptr
  // COM: This expr will be kept as a value since #local_variable is referenced multiple times.
  // CHECK: llvm.intr.dbg.value #[[LOCALVAR]] #llvm.di_expression<[DW_OP_deref]>
  debuginfo.value #local_variable #deref_expr = %v2 : !llvm.ptr
  // CHECK: llvm.intr.dbg.value #[[LOCALVAR_STRUCT]] #llvm.di_expression<[DW_OP_deref, DW_OP_LLVM_fragment(32, 32)]>
  debuginfo.value #local_variable_struct #agg_expr = %v2 : !llvm.ptr
  return
}

!struct_with_zero_sized_fields = !debuginfo.struct<MyStruct(
            !debuginfo.member<first: !debuginfo.struct<EmptyStruct()>>,
            !debuginfo.member<second: !debuginfo.unresolved<i32>>
          )>
#local_variable_struct_zsf = #debuginfo.local_variable<
  scope = #subprogram,
  name = "foostruct_zsf",
  file = #file,
  line = 10,
  arg = 0,
  alignInBits = 32
> : !struct_with_zero_sized_fields

#zsf_second = #debuginfo.expr.agg<
  #debuginfo.expr.irvalue : !debuginfo.unresolved<i32>, 1
> : !struct_with_zero_sized_fields

// CHECK-LABEL: @simplify
func.func @simplify() {
  %v1 = llvm.mlir.constant(1: i32) : i32
  // COM: There should be no more DI expr on the second field as it covers the entire struct.
  // CHECK: llvm.intr.dbg.value #[[LOCALVAR_STRUCT_ZSF:[^ ]+]] = {{.*}} : i32
  debuginfo.value #local_variable_struct_zsf #zsf_second = %v1 : i32
  return
}
