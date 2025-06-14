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


@fieldwise_init
struct MemPair:
    var x: Int
    var y: Int


# CHECK-LABEL: lit.fn @"power
fn power(lhs: Int, rhs: Int) -> MemPair:
    return MemPair(lhs, rhs)
    # CHECK: lit.end_fn
    # CHECK-NEXT: } loc(#[[LOC_FUNC:.*]])


# CHECK: ![[SP_TYPE:.*]] = !debuginfo.subroutine<(!Int, !Int, !lit.ref<!MemPair, mut *"__result__`">) -> (!kgen.none): DW_CC_normal>
# CHECK: #power_name = #debuginfo.source_name<(fn)"power"(#Int_name, #Int_name) from <(module)"debuginfo">>
# CHECK: #[[SP:.*]] = #debuginfo.subprogram<compileUnit = #{{.*}}, scope = #{{.*}}, sourceName = #power_name, linkageName = "power{{.*}}", file = #{{.*}}, line = [[LN:[0-9]+]], scopeLine = [[LN]], subprogramFlags = "Definition|Optimized"> : ![[SP_TYPE]]
# CHECK: #[[LOC_FUNC]] = loc(fused<#[[SP]]>
