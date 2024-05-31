# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -debug-level full -mlir-print-debuginfo %s --kgen-print-inline-type-values -split-input-file | FileCheck %s


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

# // -----


# CHECK-LABEL: lit.func @"testImplicitVarDeclScope
def testImplicitVarDeclScope():
    # CHECK-DAG: lit.var.decl "outer" {{.*}} loc(#[[LOC_OUTER:.+]])
    # CHECK-DAG: lit.var.decl "inner" {{.*}} loc(#[[LOC_INNER:.+]])
    outer = 8
    if True:
        inner = 5


# CHECK-LABEL: lit.func @"testImplicitVarDeclScopeNoDebug
# CHECK-SAME: always_inline_no_debug
@always_inline("nodebug")
def testImplicitVarDeclScopeNoDebug():
    # CHECK-DAG: lit.var.decl "inner" {{.*}} loc(#[[LOC_INNER_NODEBUG:.+]])
    if True:
        inner = 5


# CHECK-DAG: #[[SP:.+]] = #debuginfo.subprogram<{{.*}}linkageName = "testImplicitVarDeclScope()"
# CHECK-DAG: #[[LOC_OUTER]] = loc(fused<#[[SP]]>
# CHECK-DAG: #[[LOC_INNER]] = loc(fused<#[[SP]]>
# CHECK-DAG: #[[LOC_INNER_NODEBUG]] = loc("{{.*}}":{{[0-9]+}}:{{[0-9]+}})
