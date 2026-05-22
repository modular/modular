# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -debug-level full -mlir-print-debuginfo %s --kgen-print-inline-type-values -split-input-file | FileCheck %s


##===----------------------------------------------------------------------===##
# def/def
##===----------------------------------------------------------------------===##


# CHECK-LABEL: lit.fn @"testAlwaysInline
# CHECK-SAME: always_inline
@always_inline
def testAlwaysInline():
    # CHECK: lit.return {{.*}} loc(#[[LOC_INLINE:.+]])
    pass


# CHECK-LABEL: lit.fn @"testAlwaysInlineNoDebug
# CHECK-SAME: always_inline_no_debug
@always_inline("nodebug")
def testAlwaysInlineNoDebug():
    # CHECK: lit.return {{.*}} loc(#[[LOC_INLINE_NODEBUG:.+]])
    pass


# CHECK-DAG: #[[LOC_INLINE_NODEBUG]] = loc("{{.+}}":{{[0-9]+}}:{{[0-9]+}})
# CHECK-DAG: #[[LOC_INLINE]] = loc(fused<

# // -----


# CHECK-LABEL: lit.fn @"testImplicitVarDeclScope
def testImplicitVarDeclScope() raises:
    # CHECK-DAG: lit.var.decl "outer" {{.*}} loc(#[[LOC_OUTER:.+]])
    # CHECK-DAG: lit.var.decl "inner" {{.*}} loc(#[[LOC_INNER:.+]])
    outer = 8
    if True:
        inner = 5


# CHECK-LABEL: lit.fn @"testImplicitVarDeclScopeNoDebug
# CHECK-SAME: always_inline_no_debug
@always_inline("nodebug")
def testImplicitVarDeclScopeNoDebug() raises:
    # CHECK-DAG: lit.var.decl "inner" {{.*}} loc(#[[LOC_INNER_NODEBUG:.+]])
    if True:
        inner = 5


# CHECK-DAG: #[[SP:.+]] = #debuginfo.subprogram<{{.*}}linkageName = "testImplicitVarDeclScope()"
# CHECK-DAG: #[[LOC_OUTER]] = loc(fused<#[[SP]]>
# CHECK-DAG: #[[LOC_INNER]] = loc(fused<#[[SP]]>
# CHECK-DAG: #[[LOC_INNER_NODEBUG]] = loc("{{.*}}":{{[0-9]+}}:{{[0-9]+}})

# // -----


# CHECK-DAG: lit.fn @"fn_where_clause{{.*}}, #[[LOC_WHERE_FN:loc[0-9]+]]>}{{.*}} attributes
def fn_where_clause[x: Int]() where x:
    pass


# COM: Make sure this is a FileLineColLoc and not a FusedLoc.
# CHECK-DAG: #[[LOC_WHERE_FN]] = loc("{{.*}}":{{[0-9]+}}:{{[0-9]+}})
