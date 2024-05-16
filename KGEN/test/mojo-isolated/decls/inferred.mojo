# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | kgen-opt -verify-parameters | FileCheck %s

##===----------------------------------------------------------------------===##
# inferred Parameters
##===----------------------------------------------------------------------===##


trait SomeTrait:
    pass


@register_passable("trivial")
struct ParamType[x: int](SomeTrait):
    pass


@register_passable("trivial")
struct DependentParam[x: int, y: ParamType[x]]:
    pass


fn inferred_param_from_arg[inferred x: int](y: ParamType[x]):
    pass


fn inferred_param_from_param[inferred x: int, y: ParamType[x]]():
    pass


fn inferred_param_variadic[inferred x: int, *y: ParamType[x]]():
    pass


fn inferred_with_default[inferred x: int, y: ParamType[x], z: int = `1`]():
    pass


fn inferred_trait[inferred T: SomeTrait, y: T]():
    pass


fn inferred_dependent_param[
    inferred x: int, inferred y: ParamType[x], z: DependentParam[x, y]
]():
    pass


# CHECK-LABEL: lit.func @"test_inferred_params
fn test_inferred_params[x: int, y: ParamType[x], z: DependentParam[x, y]]():
    # CHECK: inferred_param_from_arg{{.*}}<x>(%0)
    inferred_param_from_arg(y)
    # CHECK: inferred_param_from_param{{.*}}<x, :[[PARAMTYPE:@.*ParamType]]<x> y>
    inferred_param_from_param[y]()
    # CHECK: inferred_param_variadic{{.*}}<x, :variadic<[[PARAMTYPE]]<x>> [y, y]>
    inferred_param_variadic[y, y]()
    # CHECK: inferred_trait{{.*}}<:!SomeTrait {{#.*}}, :[[PARAMTYPE]]<x> y>
    inferred_trait[y]()
    # CHECK: inferred_with_default{{.*}}<x, :[[PARAMTYPE]]<x> y, 1>
    inferred_with_default[y]()
    # CHECK: inferred_with_default{{.*}}<x, :[[PARAMTYPE]]<x> y, 2>
    inferred_with_default[y, `2`]()
    # CHECK: inferred_dependent_param{{.*}}<x, :[[PARAMTYPE]]<x> y, :{{@.*DependentParam}}<x, :[[PARAMTYPE]]<x> y> z>
    inferred_dependent_param[z]()
