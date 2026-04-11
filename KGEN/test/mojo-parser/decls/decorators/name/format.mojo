# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s

# Test that @__name can accept a t-string argument directly, producing a
# parametric linkage name expression.


# CHECK: lit.fn @"parametric_tstring[::Int,::String]()"
# CHECK-SAME: linkageName = #kgen.linkage_name<#kgen.param.expr<data_to_str
# CHECK-SAME: __make_tstring{{.*}}:string "my_name_{}_{}"
@__name(t"my_name_{A}_{B}")
@no_inline
def parametric_tstring[A: Int, B: String]():
    pass


# Test a t-string with no interpolations. Even with a static template the
# linkage name is represented as a param.expr because it goes through
# StringSlice construction.

# CHECK: lit.fn @"static_tstring()"
# CHECK-SAME: linkageName = #kgen.linkage_name<#kgen.param.expr<data_to_str
@__name(t"static_name")
def static_tstring():
    pass
