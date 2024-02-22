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
#local_variable = #debuginfo.local_variable<
  scope = #subprogram,
  name = "foo",
  file = #file,
  line = 10,
  arg = 0,
  alignInBits = 32
> : !debuginfo.unresolved<i32>
#local_variable_2 = #debuginfo.local_variable<
  scope = #subprogram,
  name = "foo_2",
  file = #file,
  line = 10,
  arg = 0,
  alignInBits = 64
> : !debuginfo.unresolved<!llvm.ptr>
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
