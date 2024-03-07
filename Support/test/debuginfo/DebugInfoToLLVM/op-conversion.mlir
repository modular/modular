// RUN: support-dialect-opt %s -convert-debuginfo-to-llvm -allow-unregistered-dialect -mlir-print-debuginfo | FileCheck %s

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
> : !debuginfo.subroutine<(!debuginfo.unresolved<i32>) -> (): DW_CC_normal>
// CHECK-DAG: #[[LOCAL_VAR:.*]] = #llvm.di_local_variable<scope = {{.*}}, name = "foo"
#local_variable = #debuginfo.local_variable<
  scope = #subprogram,
  name = "foo",
  file = #file,
  line = 10,
  arg = 0,
  alignInBits = 32
> : !debuginfo.unresolved<i32>
// CHECK-DAG: #[[LOCAL_VAR2:.*]] = #llvm.di_local_variable<scope = {{.*}}, name = "foo_2"
#local_variable_2 = #debuginfo.local_variable<
  scope = #subprogram,
  name = "foo_2",
  file = #file,
  line = 10,
  arg = 0,
  alignInBits = 64
> : !debuginfo.unresolved<!llvm.ptr>
// CHECK-DAG: #[[LOCAL_VAR3:.*]] = #llvm.di_local_variable<scope = {{.*}}, name = "foo_3"
#local_variable_3 = #debuginfo.local_variable<
  scope = #subprogram,
  name = "foo_3",
  file = #file,
  line = 10,
  arg = 0,
  alignInBits = 64
> : !debuginfo.unresolved<!llvm.ptr>
#deref_expr = #debuginfo.expr.deref<#debuginfo.expr.irvalue : !debuginfo.ptr<!llvm.ptr {sizeInBits = 64, alignInBits = 64}>> : !debuginfo.unresolved<!llvm.ptr>

// CHECK-LABEL: func @simple
func.func @simple() {
  // CHECK: %[[VAL:.*]] = llvm.mlir.constant(0 : i32) : i32
  %value = llvm.mlir.constant(0 : i32) : i32

  // CHECK: llvm.intr.dbg.value #{{.*}} = %[[VAL]] : i32
  debuginfo.value #local_variable = %value : i32
  return
}

// Test translation of dbg.value to dbg.addr.

// CHECK-LABEL: func @value_to_addr_arg
// CHECK-SAME: (%[[ARG:.*]]: i32 loc({{.*}}))
func.func @value_to_addr_arg(%arg: i32) -> i32 {
  // CHECK: %[[COUNT:.*]] = llvm.mlir.constant(1 : i32) : i32 loc(#[[LOC_UNKNOWN:.*]])
  // CHECK: %[[ALLOC:.*]] = llvm.alloca %[[COUNT]] x i32 : (i32) -> !llvm.ptr loc(#[[LOC_UNKNOWN]])
  // CHECK: llvm.intr.dbg.declare #{{.*}} = %[[ALLOC]] : !llvm.ptr
  // CHECK: llvm.store %[[ARG]], %[[ALLOC]] : i32, !llvm.ptr loc(#[[LOC_STORE:.*]])
  // CHECK: %[[RESULT:.*]] = llvm.load %[[ALLOC]] : !llvm.ptr -> i32
  // CHECK: return %[[RESULT]] : i32

  debuginfo.value #local_variable = %arg : i32
  return %arg : i32
}
// CHECK-NEXT } loc(#[[LOC_STORE]])

// CHECK-LABEL: func @value_to_addr_op
func.func @value_to_addr_op() -> i32 {
  // CHECK: %[[COUNT:.*]] = llvm.mlir.constant(1 : i32) : i32 loc(#[[LOC_UNKNOWN]])
  // CHECK: %[[ALLOC:.*]] = llvm.alloca %[[COUNT]] x i32 : (i32) -> !llvm.ptr loc(#[[LOC_UNKNOWN]])
  // CHECK: %[[VALUE:.*]] = "test.op"() : () -> i32
  // CHECK: llvm.store %[[VALUE]], %[[ALLOC]] : i32, !llvm.ptr
  // CHECK: llvm.intr.dbg.declare #{{.*}} = %[[ALLOC]] : !llvm.ptr
  // CHECK: %[[RESULT:.*]] = llvm.load %[[ALLOC]] : !llvm.ptr
  // CHECK: return %[[RESULT]] : i32

  %value = "test.op"() : () -> i32
  debuginfo.value #local_variable = %value : i32
  return %value : i32
}

// CHECK-LABEL: func @value_with_two_nontrivial_ops
func.func @value_with_two_nontrivial_ops() -> (i32, i32) {
  // CHECK: %[[VALUE1:.*]] = "test.op"() : () -> i32
  // CHECK: llvm.intr.dbg.value #[[LOCAL_VAR]] = %[[VALUE1]] : i32
  // CHECK: %[[VALUE2:.*]] = "test.op2"() : () -> i32
  // CHECK: llvm.intr.dbg.value #[[LOCAL_VAR]] = %[[VALUE2]] : i32
  // CHECK: return %[[VALUE1]], %[[VALUE2]] : i32, i32

  %value1 = "test.op"() : () -> i32
  debuginfo.value #local_variable = %value1 : i32
  %value2 = "test.op2"() : () -> i32
  debuginfo.value #local_variable = %value2 : i32
  return %value1, %value2 : i32, i32
}

// CHECK-LABEL: func @value_with_one_undef_op
func.func @value_with_one_undef_op() -> i32 {
  // CHECK: %[[COUNT:.*]] = llvm.mlir.constant(1 : i32) : i32 loc(#[[LOC_UNKNOWN]])
  // CHECK: %[[ALLOC:.*]] = llvm.alloca %[[COUNT]] x i32 : (i32) -> !llvm.ptr loc(#[[LOC_UNKNOWN]])
  // CHECK: %[[VALUE:.*]] = "test.op"() : () -> i32
  // CHECK: llvm.store %[[VALUE]], %[[ALLOC]] : i32, !llvm.ptr
  // CHECK: llvm.intr.dbg.value #[[LOCAL_VAR:.*]] #llvm.di_expression<[DW_OP_deref]> = %[[ALLOC]] : !llvm.ptr
  // CHECK: %[[UNDEF:.*]] = llvm.mlir.undef : i32
  // CHECK: llvm.intr.dbg.value #[[LOCAL_VAR]] = %[[UNDEF]] : i32
  // CHECK: %[[RESULT:.*]] = llvm.load %[[ALLOC]] : !llvm.ptr
  // CHECK: return %[[RESULT]] : i32

  %value = "test.op"() : () -> i32
  debuginfo.value #local_variable = %value : i32
  %undef = llvm.mlir.undef : i32
  debuginfo.value #local_variable = %undef : i32
  return %value : i32
}

// CHECK-LABEL: func @value_with_two_undef_ops
func.func @value_with_two_undef_ops() -> i32 {
  // CHECK: %[[COUNT:.*]] = llvm.mlir.constant(1 : i32) : i32 loc(#[[LOC_UNKNOWN]])
  // CHECK: %[[ALLOC:.*]] = llvm.alloca %[[COUNT]] x i32 : (i32) -> !llvm.ptr loc(#[[LOC_UNKNOWN]])
  // CHECK: %[[UNDEF1:.*]] = llvm.mlir.undef : i32
  // CHECK: llvm.intr.dbg.value #[[LOCAL_VAR:.*]] = %[[UNDEF1]] : i32
  // CHECK: %[[VALUE:.*]] = "test.op"() : () -> i32
  // CHECK: llvm.store %[[VALUE]], %[[ALLOC]] : i32, !llvm.ptr
  // CHECK: llvm.intr.dbg.value #[[LOCAL_VAR]] #llvm.di_expression<[DW_OP_deref]> = %[[ALLOC]] : !llvm.ptr
  // CHECK: %[[UNDEF2:.*]] = llvm.mlir.undef : i32
  // CHECK: llvm.intr.dbg.value #[[LOCAL_VAR]] = %[[UNDEF2]] : i32
  // CHECK: %[[RESULT:.*]] = llvm.load %[[ALLOC]] : !llvm.ptr
  // CHECK: return %[[RESULT]] : i32

  %undef1 = llvm.mlir.undef : i32
  debuginfo.value #local_variable = %undef1 : i32
  %value = "test.op"() : () -> i32
  debuginfo.value #local_variable = %value : i32
  %undef2 = llvm.mlir.undef : i32
  debuginfo.value #local_variable = %undef2 : i32
  return %value : i32
}

// CHECK-LABEL: func @undef_values_only
func.func @undef_values_only() -> (i32, i32) {
  // CHECK: %[[COUNT:.*]] = llvm.mlir.constant(1 : i32) : i32 loc(#[[LOC_UNKNOWN]])
  // CHECK: %[[ALLOC:.*]] = llvm.alloca %[[COUNT]] x i32 : (i32) -> !llvm.ptr loc(#[[LOC_UNKNOWN]])
  // CHECK: %[[UNDEF1:.*]] = llvm.mlir.undef : i32
  // CHECK: llvm.store %[[UNDEF1]], %[[ALLOC]] : i32, !llvm.ptr
  // CHECK: llvm.intr.dbg.value #[[LOCAL_VAR]] #llvm.di_expression<[DW_OP_deref]> = %[[ALLOC]] : !llvm.ptr
  // CHECK: %[[UNDEF2:.*]] = llvm.mlir.undef : i32
  // CHECK: llvm.intr.dbg.value #[[LOCAL_VAR]] = %[[UNDEF2]] : i32
  // CHECK: %[[RESULT:.*]] = llvm.load %[[ALLOC]] : !llvm.ptr
  // CHECK: return %[[RESULT]], %[[UNDEF2]] : i32, i32

  %undef1 = llvm.mlir.undef : i32
  debuginfo.value #local_variable = %undef1 : i32
  %undef2 = llvm.mlir.undef : i32
  debuginfo.value #local_variable = %undef2 : i32
  return %undef1, %undef2 : i32, i32
}

// CHECK-LABEL: func @two_value_to_addr_op
func.func @two_value_to_addr_op() -> !llvm.ptr {
  // CHECK: %[[COUNT:.*]] = llvm.mlir.constant(1 : i32) : i32 loc(#[[LOC_UNKNOWN]])
  // CHECK: %[[ALLOC:.*]] = llvm.alloca %[[COUNT]] x !llvm.ptr : (i32) -> !llvm.ptr loc(#[[LOC_UNKNOWN]])
  // CHECK: %[[VALUE:.*]] = "test.op"() : () -> !llvm.ptr
  // CHECK: llvm.store %[[VALUE]], %[[ALLOC]] : !llvm.ptr, !llvm.ptr
  // CHECK: llvm.intr.dbg.declare #{{.*}} = %[[ALLOC]] : !llvm.ptr
  // CHECK: llvm.intr.dbg.declare #{{.*}} = %[[ALLOC]] : !llvm.ptr
  // CHECK: %[[RESULT:.*]] = llvm.load %[[ALLOC]] : !llvm.ptr
  // CHECK: return %[[RESULT]] : !llvm.ptr

  %value = "test.op"() : () -> !llvm.ptr
  debuginfo.value #local_variable_2 = %value : !llvm.ptr
  debuginfo.value #local_variable_3 = %value : !llvm.ptr
  return %value : !llvm.ptr
}

// CHECK-LABEL: func @one_value_one_value_and_undef
func.func @one_value_one_value_and_undef() -> !llvm.ptr {
  // CHECK: %[[COUNT:.*]] = llvm.mlir.constant(1 : i32) : i32 loc(#[[LOC_UNKNOWN]])
  // CHECK: %[[ALLOC:.*]] = llvm.alloca %[[COUNT]] x !llvm.ptr : (i32) -> !llvm.ptr loc(#[[LOC_UNKNOWN]])
  // CHECK: %[[VALUE:.*]] = "test.op"() : () -> !llvm.ptr
  // CHECK: llvm.store %[[VALUE]], %[[ALLOC]] : !llvm.ptr, !llvm.ptr
  // CHECK: llvm.intr.dbg.declare #[[LOCAL_VAR2]] = %[[ALLOC]] : !llvm.ptr
  // CHECK: llvm.intr.dbg.value #[[LOCAL_VAR3]] #llvm.di_expression<[DW_OP_deref]> = %[[ALLOC]] : !llvm.ptr
  // CHECK: %[[UNDEF:.*]] = llvm.mlir.undef : !llvm.ptr
  // CHECK: llvm.intr.dbg.value #[[LOCAL_VAR3]] = %[[UNDEF]] : !llvm.ptr
  // CHECK: %[[RESULT:.*]] = llvm.load %[[ALLOC]] : !llvm.ptr
  // CHECK: return %[[RESULT]] : !llvm.ptr

  %value = "test.op"() : () -> !llvm.ptr
  debuginfo.value #local_variable_2 = %value : !llvm.ptr
  debuginfo.value #local_variable_3 = %value : !llvm.ptr
  %undef = llvm.mlir.undef : !llvm.ptr
  debuginfo.value #local_variable_3 = %undef : !llvm.ptr
  return %value : !llvm.ptr
}

// CHECK-LABEL: func @one_value_one_deref_to_addr_op
func.func @one_value_one_deref_to_addr_op() -> !llvm.ptr {
  // CHECK: %[[COUNT:.*]] = llvm.mlir.constant(1 : i32) : i32 loc(#[[LOC_UNKNOWN]])
  // CHECK: %[[ALLOC:.*]] = llvm.alloca %[[COUNT]] x !llvm.ptr : (i32) -> !llvm.ptr loc(#[[LOC_UNKNOWN]])
  // CHECK: %[[VALUE:.*]] = "test.op"() : () -> !llvm.ptr
  // CHECK: llvm.store %[[VALUE]], %[[ALLOC]] : !llvm.ptr, !llvm.ptr
  // CHECK: llvm.intr.dbg.declare #{{.*}} = %[[ALLOC]] : !llvm.ptr
  // CHECK: llvm.intr.dbg.declare #{{.*}} = %[[VALUE]] : !llvm.ptr
  // CHECK: %[[RESULT:.*]] = llvm.load %[[ALLOC]] : !llvm.ptr
  // CHECK: return %[[RESULT]] : !llvm.ptr

  %value = "test.op"() : () -> !llvm.ptr
  debuginfo.value #local_variable_2 = %value : !llvm.ptr
  debuginfo.value #local_variable_3 #deref_expr = %value : !llvm.ptr
  return %value : !llvm.ptr
}

// CHECK-LABEL: func @one_deref_one_value_to_addr_op
func.func @one_deref_one_value_to_addr_op() -> !llvm.ptr {
  // CHECK: %[[COUNT:.*]] = llvm.mlir.constant(1 : i32) : i32 loc(#[[LOC_UNKNOWN]])
  // CHECK: %[[ALLOC:.*]] = llvm.alloca %[[COUNT]] x !llvm.ptr : (i32) -> !llvm.ptr loc(#[[LOC_UNKNOWN]])
  // CHECK: %[[VALUE:.*]] = "test.op"() : () -> !llvm.ptr
  // CHECK: llvm.store %[[VALUE]], %[[ALLOC]] : !llvm.ptr, !llvm.ptr
  // CHECK: llvm.intr.dbg.declare #{{.*}} = %[[VALUE]] : !llvm.ptr
  // CHECK: llvm.intr.dbg.declare #{{.*}} = %[[ALLOC]] : !llvm.ptr
  // CHECK: %[[RESULT:.*]] = llvm.load %[[ALLOC]] : !llvm.ptr
  // CHECK: return %[[RESULT]] : !llvm.ptr

  %value = "test.op"() : () -> !llvm.ptr
  debuginfo.value #local_variable_2 #deref_expr = %value : !llvm.ptr
  debuginfo.value #local_variable_3 = %value : !llvm.ptr
  return %value : !llvm.ptr
}

// CHECK-LABEL: @block_arguments
llvm.func @block_arguments() {
  %0 = llvm.mlir.constant(0 : i32) : i32
  llvm.br ^bb1(%0 : i32)
// CHECK: fused<#di_subprogram>
^bb1(%arg0: i32 loc(fused<#subprogram>["foo.mlir":0:0])):
  llvm.return
}

// CHECK: #[[LOC_UNKNOWN]] = loc(unknown)
