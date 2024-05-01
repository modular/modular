# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -debug-level full -mlir-print-debuginfo %s --kgen-print-inline-vtables | FileCheck %s


##===----------------------------------------------------------------------===##
# fn/def
##===----------------------------------------------------------------------===##


# CHECK-LABEL: lit.func @"testAlwaysInline
# CHECK-SAME: always_inline
@always_inline
fn testAlwaysInline():
    # CHECK: lit.return {{.*}} loc(#[[LOC_INLINE:.+]])
    pass


# CHECK-LABEL: lit.func @"testAlwaysInlineNoDebug
# CHECK-SAME: always_inline_no_debug
@always_inline("nodebug")
fn testAlwaysInlineNoDebug():
    # CHECK: lit.return {{.*}} loc(#[[LOC_INLINE_NODEBUG:.+]])
    pass


# CHECK-DAG: #[[LOC_INLINE_NODEBUG]] = loc("{{.+}}":{{[0-9]+}}:{{[0-9]+}})
# CHECK-DAG: #[[LOC_INLINE]] = loc(fused<
