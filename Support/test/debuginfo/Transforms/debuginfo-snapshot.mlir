// RUN: support-dialect-opt %s -debuginfo-snapshot='filename=%/t' -mlir-print-debuginfo -allow-unregistered-dialect | FileCheck %s

// CHECK-DAG: ![[BASIC:.*]] = !debuginfo.basic<i32 {sizeInBits = 32, alignInBits = 32, encoding = DW_ATE_unsigned}>
// CHECK-DAG: ![[BASIC1:.*]] = !debuginfo.basic<i1 {sizeInBits = 8, alignInBits = 8, encoding = DW_ATE_unsigned}>
// CHECK-DAG: ![[SUBROUTINE:.*]] = !debuginfo.subroutine<(!basic, !basic1) -> (!basic): DW_CC_normal>

// CHECK-DAG: #[[FILE:.*]] = #debuginfo.file<"{{.*}}" in "{{.*}}">
// CHECK-DAG: #[[CU:.*]] = #debuginfo.compile_unit<sourceLanguage = DW_LANG_Mojo, file = #[[FILE]], producer = "MLIR", isOptimized = true, emissionKind = Full, nameTableKind = None>

// CHECK-DAG: #[[SP:.*]] = #debuginfo.subprogram<compileUnit = #[[CU]], scope = #[[FILE]], name = <"foo">, linkageName = "foo", file = #[[FILE]], line = {{.*}}, scopeLine = {{.*}}, subprogramFlags = "Definition|Optimized"> : ![[SUBROUTINE]]

// CHECK-DAG: #[[ARG1_VAR:.*]] = #debuginfo.local_variable<scope = #[[SP]], name = "arg0", file = #[[FILE]], line = {{.*}}, arg = 1, flags = Zero> : ![[BASIC]]
// CHECK-DAG: #[[ARG2_VAR:.*]] = #debuginfo.local_variable<scope = #[[SP]], name = "arg1", file = #[[FILE]], line = {{.*}}, arg = 2, flags = Zero> : ![[BASIC1]]

// CHECK-DAG: #[[BB_ARG_VAR:.*]] = #debuginfo.local_variable<scope = #[[SP]], name = "0", file = #[[FILE]], line = {{.*}}> : ![[BASIC]]

// CHECK-DAG: #[[OP_VALUES_VAR_0:.*]] = #debuginfo.local_variable<scope = #[[SP]], name = "1#0", file = #[[FILE]], line = {{.*}}> : ![[BASIC]]
// CHECK-DAG: #[[OP_VALUES_VAR_1:.*]] = #debuginfo.local_variable<scope = #[[SP]], name = "1#1", file = #[[FILE]], line = {{.*}}> : ![[BASIC]]
// CHECK-DAG: #[[OP_VALUES_VAR_2:.*]] = #debuginfo.local_variable<scope = #[[SP]], name = "1#2", file = #[[FILE]], line = {{.*}}> : ![[BASIC]]

// CHECK-DAG: #[[BB_ARG2_VAR:.*]] = #debuginfo.local_variable<scope = #[[SP]], name = "2", file = #[[FILE]], line = {{.*}}> : ![[BASIC]]

// CHECK-LABEL: func.func @foo(
// CHECK-SAME:   %[[ARG1:.*]]: i32
// CHECK-SAME:   %[[ARG2:.*]]: i1
// CHECK-SAME: ) -> i32
func.func @foo(%arg: i32, %cond: i1) -> i32 {
  // CHECK: debuginfo.value #local_variable = %[[ARG1]] : i32
  // CHECK: debuginfo.value #local_variable1 = %[[ARG2]] : i1
  llvm.cond_br %cond, ^bb1(%arg: i32), ^bb2(%arg: i32)

// CHECK: ^bb1(%[[BB_ARG:.*]]: i32 {{.*}}):
^bb1(%arg1: i32):
  // CHECK: debuginfo.value #[[BB_ARG_VAR]] = %[[BB_ARG]] : i32
  // CHECK: %[[OP_VALUES:.*]]:3 = "test.op"
  // CHECK: debuginfo.value #[[OP_VALUES_VAR_0]] = %[[OP_VALUES]]#0 : i32
  // CHECK: debuginfo.value #[[OP_VALUES_VAR_1]] = %[[OP_VALUES]]#1 : i32
  // CHECK: debuginfo.value #[[OP_VALUES_VAR_2]] = %[[OP_VALUES]]#2 : i32
  %values:3 = "test.op"(%arg1) : (i32) -> (i32, i32, i32)
  llvm.br ^bb2(%values#0: i32)

// CHECK: ^bb2(%[[BB_ARG2:.*]]: i32 {{.*}}):
^bb2(%arg2: i32):
  // CHECK: debuginfo.value #[[BB_ARG2_VAR]] = %[[BB_ARG2]] : i32
  return %arg2 : i32
}
