# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -debug-level full -mlir-print-debuginfo %s | FileCheck %s

# ===----------------------------------------------------------------------=== #
# Actual tests
# ===----------------------------------------------------------------------=== #

# Check single file debug info generation.

# CHECK-DAG: ![[INT_TYPE:.*]] = !debuginfo.unresolved<index>
# CHECK-DAG: ![[SP_TYPE:.*]] = !debuginfo.subroutine<(index, index) -> (index): DW_CC_normal>
# CHECK-DAG: #power_name = #debuginfo.source_name<(fn)"power"(<"index">, <"index">) from <(module)"debuginfo">>
# CHECK-DAG: #[[SP:.*]] = #debuginfo.subprogram<compileUnit = #{{.*}}, scope = #{{.*}}, name = #power_name, linkageName = "power{{.*}}", file = #{{.*}}, line = [[LN:[0-9]+]], scopeLine = [[LN]], subprogramFlags = "Definition|Optimized"> : ![[SP_TYPE]]
# CHECK-DAG: #[[LHS_VAR:.*]] = #debuginfo.local_variable<scope = #[[SP]], name = "lhs", file = #{{.*}}, line = [[LN]], arg = 1> : ![[INT_TYPE]]
# CHECK-DAG: #[[RHS_VAR:.*]] = #debuginfo.local_variable<scope = #[[SP]], name = "rhs", file = #{{.*}}, line = [[LN]], arg = 2> : ![[INT_TYPE]]


# CHECK-LABEL: lit.func @"power
fn power(lhs: int, rhs: int) -> int:
    # CHECK: debuginfo.value #[[LHS_VAR]] = %lhs
    # CHECK: debuginfo.value #[[RHS_VAR]] = %rhs
    return lhs
