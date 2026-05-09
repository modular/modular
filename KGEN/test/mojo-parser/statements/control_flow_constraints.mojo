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
    def predicate() -> Bool:
        comptime size = Self.a.size
        comptime result = size == 2
        return result


# CHECK-LABEL: lit.fn @"double_where_clause
# CHECK-SAME: where {<
# CHECK-SAME: ::@PStruct::@"predicate()"
# CHECK-SAME: eq(#kgen.param_list.size<:param_list<!Int> *"x.a.values``">, 2)> : i1>, #{{[[:alnum:]]+}}>, <
# CHECK-SAME: ::@PStruct::@"predicate()"
# CHECK-SAME: eq(#kgen.param_list.size<:param_list<!Int> *"y.a.values``3">, 2)> : i1>, #{{[[:alnum:]]+}}>}
def double_where_clause(
    x: PStruct[...], y: PStruct[...]
) where type_of(x).predicate() where type_of(y).predicate():
    pass


# CHECK-LABEL: lit.fn @"test_nested_double_where_clause
def test_nested_double_where_clause(x: PStruct[...], y: PStruct[...]):
    comptime if type_of(x).predicate():
        comptime if type_of(y).predicate():
            # CHECK: lit.call {{.*}}@"double_where_clause
            double_where_clause(x, y)
