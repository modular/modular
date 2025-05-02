# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s

struct TakesIntParam[a: Int]: pass

trait SomeTrait:
    pass


@register_passable("trivial")
struct ParamType[x: Index](SomeTrait):
    pass

##===----------------------------------------------------------------------===##
# inferred Parameters
##===----------------------------------------------------------------------===##

@register_passable("trivial")
struct DependentParam[x: Index, y: ParamType[x]]:
    pass


fn inferred_param_from_arg[x: Index, //](y: ParamType[x]):
    pass


fn inferred_param_from_param[x: Index, //, y: ParamType[x]]():
    pass


fn inferred_param_variadic[x: Index, //, *y: ParamType[x]]():
    pass


fn inferred_with_default[x: Index, //, y: ParamType[x], z: Index = `1`]():
    pass


fn inferred_trait[T: SomeTrait, //, y: T]():
    pass


fn inferred_dependent_param[
    x: Index, y: ParamType[x], //, z: DependentParam[x, y]
]():
    pass


fn inferred_partial[x: Index, //, y: Index](z: ParamType[x]):
    pass


fn inferred_partial_dependent[x: Index, //, y: Index, z: ParamType[x]]():
    pass


struct InferredStruct[x: Index, //, y: Index, z: ParamType[x]]:
    pass


struct InferredStructConversion[
    x: Index, //, y: AnyTrivialRegType, z: ParamType[x]
]:
    pass


# CHECK-LABEL: lit.fn @"test_inferred_params
fn test_inferred_params[x: Index, y: ParamType[x], z: DependentParam[x, y]]():
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
    # CHECK: inferred_dependent_param{{.*}}<x, :[[PARAMTYPE]]<x> y, :{{@.*DependentParam}}<x, :[[PARAMTYPE]]<x> y> z>
    inferred_dependent_param[x=x, z]()

    # CHECK: alias.decl [[PARTIALLY_BOUND:.*]]: !lit.generator<<"x": index, +>("z": !lit.struct<#ParamType <*(0,0)>>)
    alias partially_bound = inferred_partial[`1`]
    # CHECK: lit.call @inferred::@"inferred_partial{{.*}}"<x, 1>(
    partially_bound(y)

    # CHECK: alias.decl [[PARTIALLY_BOUND:.*]]: !lit.generator<<"x": index, +, "z": [[PARAMTYPE]]<*(0,0)>>
    # CHECK-SAME: inferred_partial_dependent{{.*}}<?, 1, :[[PARAMTYPE]]<?> ?>
    alias partially_bound_dependent = inferred_partial_dependent[`1`]
    # CHECK-NEXT: !lit.generator<() -> !kgen.none> = <{{.*}}inferred_partial_dependent{{.*}}<x, 1, :@inferred::@ParamType<x> y>>
    alias fully_bound = partially_bound_dependent[y]

    # CHECK: alias.decl [[PARTIALLY_BOUND:.*]]: meta<!lit.struct<#InferredStruct <?, 1, :[[PARAMTYPE]]<?> ?>,
    # CHECK-SAME: <"x": index, +, "z": [[PARAMTYPE]]<*(0,0)>>>> = <{{.*}}@InferredStruct<?, 1, :[[PARAMTYPE]]<?> ?>>
    alias partially_bound_type = InferredStruct[`1`]
    # CHECK-NEXT: partially_bound_explicit_inferred{{.*}} = <@inferred::@InferredStruct<1, 2, :@inferred::@ParamType<1> ?>>
    alias partially_bound_explicit_inferred = InferredStruct[x=`1`, `2`]
    # CHECK-NEXT: fully_bound_type{{.*}}<@inferred::@InferredStruct<x, 1, :@inferred::@ParamType<x> y>>
    alias fully_bound_type = partially_bound_type[y]

    # CHECK-NEXT: InferredStructConversion<x, :type !Int, :[[PARAMTYPE]]<x> y>
    var inferred_type: InferredStructConversion[Int, y]


# Multiply should work even though it is @always_inline("builtin")
fn mul2_caller[n: Int, t: TakesIntParam[n * 2]](): return mul2_callee[t]()
fn mul2_callee[n: Int, //, some_t: TakesIntParam[n * 2]]():
    pass

##===----------------------------------------------------------------------===##
# Inferred Self parameters
##===----------------------------------------------------------------------===##

trait FancyTrait(Copyable, Movable):
    fn __eq__(self, other: Self) -> Bool: pass

@value
struct MyOptional[T: Copyable & Movable]:

    fn __eq__[U: FancyTrait](self: MyOptional[U], rhs: MyOptional[U]) -> Bool:
        pass

  # CHECK-LABEL: lit.fn @"__ne__
    fn __ne__[U: FancyTrait](self: MyOptional[U], rhs: MyOptional[U]) -> Bool:
        # CHECK-NEXT: lit.call {{.*}}MyOptional::@"__eq__{{.*}}(%self, %rhs)
        return not (self == rhs)

# CHECK-LABEL: lit.fn @"testMyOptional
fn testMyOptional(a: MyOptional[Int]):
    # CHECK-NEXT: lit.call {{.*}}MyOptional::@"__eq__{{.*}}(%a, %a)
    _ = a.__eq__(a)
    # CHECK-NEXT: lit.call {{.*}}MyOptional::@"__eq__{{.*}}(%a, %a)
    _ = MyOptional.__eq__(a, a)
    # CHECK-NEXT: lit.call {{.*}}MyOptional::@"__eq__{{.*}}(%a, %a)
    _ = a == a



# CHECK-LABEL: lit.fn @"findall
# CHECK-NEXT: lit.call @stdlib::@builtin::@stubs::@Pointer::@"__init__{{.*}}(%self)
struct DefBoxInference:
    def findall(self) -> DefBoxInferenceIter[__origin_of(self)]:
        return DefBoxInferenceIter[__origin_of(self)](Pointer(to=self))


@value
struct DefBoxInferenceIter[
    origin: ImmutableOrigin,
]:
    @implicit
    fn __init__(out self, regex: Pointer[DefBoxInference, origin]):
        pass
