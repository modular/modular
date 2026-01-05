# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s

##===----------------------------------------------------------------------===##
# parameter-if
##===----------------------------------------------------------------------===##


struct PStruct[*a: Int]:
    @always_inline("builtin")
    @staticmethod
    fn predicate() -> Bool:
        comptime size = Int(
            mlir_value=__mlir_attr[`#kgen.variadic.size<`, Self.a, `> :index`]
        )
        comptime result = size == 2
        return result


# CHECK-LABEL: lit.fn @"double_where_clause
# CHECK-SAME: where {<eq(#kgen.variadic.size<#kgen.param.decl.ref<"x.a`"> : !kgen.variadic<!Int>>, 2), #{{[[:alnum:]]+}}>, <eq(#kgen.variadic.size<#kgen.param.decl.ref<"y.a`2"> : !kgen.variadic<!Int>>, 2), #{{[[:alnum:]]+}}>}
fn double_where_clause(
    x: PStruct[...], y: PStruct[...]
) where type_of(x).predicate() where type_of(y).predicate():
    pass


# CHECK-LABEL: lit.fn @"test_nested_double_where_clause
fn test_nested_double_where_clause(x: PStruct[...], y: PStruct[...]):
    @parameter
    if type_of(x).predicate():

        @parameter
        if type_of(y).predicate():
            # CHECK: lit.call {{.*}}@"double_where_clause
            double_where_clause(x, y)
