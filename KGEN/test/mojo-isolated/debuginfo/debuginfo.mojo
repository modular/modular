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


@value
struct MemPair:
    var x: Int
    var y: Int


# CHECK-LABEL: lit.func @"power
fn power(lhs: int, rhs: int) -> MemPair:
    return MemPair(lhs, rhs)
    # CHECK: lit.end_func
    # CHECK-NEXT: } loc(#[[LOC_FUNC:.*]])


# CHECK: #power_name = #debuginfo.source_name<(fn)"power"(<"index">, <"index">) from <(module)"debuginfo">>
# CHECK: ![[SP_TYPE:.*]] = !debuginfo.subroutine<(index, index, !lit.ref<!MemPair, {{.*}}>) -> (!kgen.none): DW_CC_normal>
# CHECK: #[[SP:.*]] = #debuginfo.subprogram<compileUnit = #{{.*}}, scope = #{{.*}}, name = #power_name, linkageName = "power{{.*}}", file = #{{.*}}, line = [[LN:[0-9]+]], scopeLine = [[LN]], subprogramFlags = "Definition|Optimized"> : ![[SP_TYPE]]
# CHECK: #[[LOC_FUNC]] = loc(fused<#[[SP]]>
