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


fn inferred_param_from_arg[x: int, //](y: ParamType[x]):
    pass


fn inferred_param_from_param[x: int, //, y: ParamType[x]]():
    pass


fn inferred_param_variadic[x: int, //, *y: ParamType[x]]():
    pass


fn inferred_with_default[x: int, //, y: ParamType[x], z: int = `1`]():
    pass


fn inferred_trait[T: SomeTrait, //, y: T]():
    pass


fn inferred_dependent_param[
    x: int, y: ParamType[x], //, z: DependentParam[x, y]
]():
    pass


fn inferred_partial[x: int, //, y: int](z: ParamType[x]):
    pass


fn inferred_partial_dependent[x: int, //, y: int, z: ParamType[x]]():
    pass


struct InferredStruct[x: int, //, y: int, z: ParamType[x]]:
    pass


struct InferredStructConversion[
    x: int, //, y: AnyTrivialRegType, z: ParamType[x]
]:
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

    # CHECK: alias.decl [[PARTIALLY_BOUND:.*]]: !lit.signature<<"x": index, +>("z": !lit.struct<#ParamType <*(0,0)>>)
    alias partially_bound = inferred_partial[`1`]
    # CHECK: call[!lit.signature<("z": !lit.struct<#ParamType <x>>) -> !kgen.none>:
    # CHECK-SAME: bind_signature(:{{.*}} [[PARTIALLY_BOUND]], x)]
    partially_bound(y)

    # CHECK: alias.decl [[PARTIALLY_BOUND:.*]]: !lit.signature<<"x": index, +, "z": [[PARAMTYPE]]<*(0,0)>>
    # CHECK-SAME: inferred_partial_dependent{{.*}}<?, 1, :[[PARAMTYPE]]<?> ?>
    alias partially_bound_dependent = inferred_partial_dependent[`1`]
    # CHECK-NEXT: !lit.signature<() -> !kgen.none> = <bind_signature(:{{.*}} [[PARTIALLY_BOUND]], x, y)>
    alias fully_bound = partially_bound_dependent[y]

    # CHECK: alias.decl [[PARTIALLY_BOUND:.*]]: anystruct<#InferredStruct <?, 1, :[[PARAMTYPE]]<?> ?>,
    # CHECK-SAME: <"x": index, +, "z": [[PARAMTYPE]]<*(0,0)>>> = <{{.*}}@InferredStruct<?, 1, :[[PARAMTYPE]]<?> ?>>
    alias partially_bound_type = InferredStruct[`1`]
    # CHECK-NEXT: anystruct<#InferredStruct <x, 1, :[[PARAMTYPE]]<x> y>> = <#lit.bind_type<:{{.*}} [[PARTIALLY_BOUND]], [x, y]>>
    alias fully_bound_type = partially_bound_type[y]

    # CHECK-NEXT: InferredStructConversion<x, :type !Int, :[[PARAMTYPE]]<x> y>
    var inferred_type: InferredStructConversion[Int, y]

##===----------------------------------------------------------------------===##
# Inferred Self parameters
##===----------------------------------------------------------------------===##

trait FancyTrait(CollectionElement):
    fn __eq__(self, other: Self) -> Bool: pass

@value
struct MyOptional[T: CollectionElement]:

    fn __eq__[U: FancyTrait](self: MyOptional[U], rhs: MyOptional[U]) -> Bool:
        pass

  # CHECK-LABEL: lit.func @"__ne__
    fn __ne__[U: FancyTrait](self: MyOptional[U], rhs: MyOptional[U]) -> Bool:
        # CHECK-NEXT: lit.call {{.*}}MyOptional::@"__eq__{{.*}}(%self, %rhs)
        return not (self == rhs)

# CHECK-LABEL: lit.func @"testMyOptional
fn testMyOptional(a: MyOptional[Int]):
    # CHECK-NEXT: lit.call {{.*}}MyOptional::@"__eq__{{.*}}(%a, %a)
    _ = a.__eq__(a)
    # CHECK-NEXT: lit.call {{.*}}MyOptional::@"__eq__{{.*}}(%a, %a)
    _ = MyOptional.__eq__(a, a)
    # CHECK-NEXT: lit.call {{.*}}MyOptional::@"__eq__{{.*}}(%a, %a)
    _ = a == a



# CHECK-LABEL: lit.func @"findall
# CHECK-NEXT: %anonymous2A = lit.var.decl "anonymous*" synth : !lit.ref<{{.*}}@Reference<{{.*}} :lifetime<0> *"self`2x", 
# CHECK-NEXT: lit.call {{.*}}Reference::@"__init__{{.*}}(%anonymous2A, %self)
struct DefBoxInference:
    def findall(self) -> DefBoxInferenceIter[__lifetime_of(self)]:
        return DefBoxInferenceIter(self)


@value
struct DefBoxInferenceIter[
    lifetime: ImmutableLifetime,
]:
    fn __init__(inout self, regex: Reference[DefBoxInference, lifetime]):
        pass