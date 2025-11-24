# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s


struct TakesIntParam[a: Int]:
    pass


trait SomeTrait:
    pass


@register_passable("trivial")
struct ParamType[x: Int](SomeTrait):
    pass


##===----------------------------------------------------------------------===##
# inferred Parameters
##===----------------------------------------------------------------------===##


@register_passable("trivial")
struct DependentParam[x: Int, y: ParamType[x]]:
    pass


fn inferred_param_from_arg[x: Int, //](y: ParamType[x]):
    pass


fn inferred_param_from_param[x: Int, //, y: ParamType[x]]():
    pass


fn inferred_param_variadic[x: Int, //, *y: ParamType[x]]():
    pass


fn inferred_with_default[x: Int, //, y: ParamType[x], z: Int = 1]():
    pass


fn inferred_trait[T: SomeTrait, //, y: T]():
    pass


fn inferred_dependent_param[
    x: Int, y: ParamType[x], //, z: DependentParam[x, y]
]():
    pass


fn inferred_partial[x: Int, //, y: Int](z: ParamType[x]):
    pass


fn inferred_partial_dependent[x: Int, //, y: Int, z: ParamType[x]]():
    pass


struct InferredStruct[x: Int, //, y: Int, z: ParamType[x]]:
    pass


struct InferredStructConversion[
    x: Int, //, y: AnyTrivialRegType, z: ParamType[x]
]:
    pass


# CHECK-LABEL: lit.fn @"test_inferred_params
fn test_inferred_params[x: Int, y: ParamType[x], z: DependentParam[x, y]]():
    # CHECK: inferred_param_from_arg{{.*}}<:!Int x>(%0)
    inferred_param_from_arg(y)
    # CHECK: inferred_param_from_param{{.*}}<:!Int x, :!lit.struct<#ParamType <:!Int x>> y>
    inferred_param_from_param[y]()
    # CHECK: inferred_param_variadic{{.*}}<:!Int x, :variadic<!lit.struct<#ParamType <:!Int x>>> [y, y]>
    inferred_param_variadic[y, y]()
    # CHECK: inferred_trait{{.*}}<:!SomeTrait @inferred::@ParamType<:!Int x>, :!lit.struct<#ParamType <:!Int x>> y>
    inferred_trait[y]()
    # CHECK: inferred_with_default{{.*}}<:!Int x, :!lit.struct<#ParamType <:!Int x>> y, :!Int {1}>()
    inferred_with_default[y]()
    # CHECK: inferred_with_default{{.*}}<:!Int x, :!lit.struct<#ParamType <:!Int x>> y, :!Int {2}>
    inferred_with_default[y, 2]()
    # CHECK: inferred_dependent_param{{.*}}<:!Int x, :!lit.struct<#ParamType <:!Int x>> y, :!lit.struct<#DependentParam <:!Int x, :!lit.struct<#ParamType <:!Int x>> y>> z>()
    inferred_dependent_param[z]()
    # CHECK: inferred_dependent_param{{.*}}<:!Int x, :!lit.struct<#ParamType <:!Int x>> y, :!lit.struct<#DependentParam <:!Int x, :!lit.struct<#ParamType <:!Int x>> y>> z>()
    inferred_dependent_param[x=x, z]()

    # CHECK: alias.decl [[PARTIALLY_BOUND:.*]]: !lit.generator<<"x": !Int, +>("z": !lit.struct<#ParamType <:!Int *(0,0)>>)
    comptime partially_bound = inferred_partial[1]
    # CHECK: lit.call @inferred::@"inferred_partial{{.*}}"<:!Int x, :!Int {1}>(
    partially_bound(y)

    # CHECK: alias.decl [[PARTIALLY_BOUND:.*]]: !lit.generator<<"x": !Int, +, "z": !lit.struct<#ParamType <:!Int *(0,0)>>
    # CHECK-SAME: inferred_partial_dependent{{.*}}<:!Int ?, :!Int {1}, :!lit.struct<#ParamType <:!Int ?>> ?>>
    comptime partially_bound_dependent = inferred_partial_dependent[1]
    # CHECK-NEXT: !lit.generator<() -> !kgen.none> = <{{.*}}inferred_partial_dependent{{.*}}<:!Int x, :!Int {1}, :!lit.struct<#ParamType <:!Int x>> y>>
    comptime fully_bound = partially_bound_dependent[y]

    # CHECK: alias.decl [[PARTIALLY_BOUND:.*]]: meta<!lit.struct<#InferredStruct <:!Int ?, :!Int {1}, :!lit.struct<#ParamType <:!Int ?>> ?>,
    # CHECK-SAME: <"x": !Int, +, "z": !lit.struct<#ParamType <:!Int *(0,0)>>>>> = <{{.*}}@InferredStruct<:!Int ?, :!Int {1}, :!lit.struct<#ParamType <:!Int ?>> ?>>
    comptime partially_bound_type = InferredStruct[1]
    # CHECK-NEXT: partially_bound_explicit_inferred{{.*}} = <@inferred::@InferredStruct<:!Int {1}, :!Int {2}, :!lit.struct<#ParamType <:!Int {1}>> ?>>
    comptime partially_bound_explicit_inferred = InferredStruct[x=1, 2]
    # CHECK-NEXT: fully_bound_type{{.*}}<@inferred::@InferredStruct<:!Int x, :!Int {1}, :!lit.struct<#ParamType <:!Int x>> y>>
    comptime fully_bound_type = partially_bound_type[y]

    # CHECK-NEXT: #InferredStructConversion <:!Int x, :type !Int, :!lit.struct<#ParamType <:!Int x>> y>>
    var inferred_type: InferredStructConversion[Int, y]


# Multiply should work even though it is @always_inline("builtin")
fn mul2_caller[n: Int, t: TakesIntParam[n * 2]]():
    return mul2_callee[t]()


fn mul2_callee[n: Int, //, some_t: TakesIntParam[n * 2]]():
    pass


##===----------------------------------------------------------------------===##
# Inferred Self parameters
##===----------------------------------------------------------------------===##


trait FancyTrait(ImplicitlyCopyable, Movable):
    fn __eq__(self, other: Self) -> Bool:
        ...


struct MyFancyStruct(FancyTrait):
    fn __eq__(self, other: Self) -> Bool:
        return False


@fieldwise_init
struct MyOptional[T: ImplicitlyCopyable & Movable]:
    fn __eq__[U: FancyTrait](self: MyOptional[U], rhs: MyOptional[U]) -> Bool:
        pass

    # CHECK-LABEL: lit.fn @"__ne__
    fn __ne__[U: FancyTrait](self: MyOptional[U], rhs: MyOptional[U]) -> Bool:
        # CHECK-NEXT: lit.call {{.*}}MyOptional::@"__eq__{{.*}}(%self, %rhs)
        return not (self == rhs)


# CHECK-LABEL: lit.fn @"testMyOptional
fn testMyOptional(a: MyOptional[MyFancyStruct]):
    # CHECK-NEXT: lit.call {{.*}}MyOptional::@"__eq__{{.*}}(%a, %a)
    _ = a.__eq__(a)
    # CHECK: lit.call {{.*}}MyOptional::@"__eq__{{.*}}(%a, %a)
    _ = MyOptional.__eq__(a, a)
    # CHECK: lit.call {{.*}}MyOptional::@"__eq__{{.*}}(%a, %a)
    _ = a == a


# CHECK-LABEL: lit.fn @"findall
# CHECK-NEXT: lit.call @stdlib::@builtin::@stubs::@Pointer::@"__init__{{.*}}(%self)
struct DefBoxInference:
    def findall(self) -> DefBoxInferenceIter[origin_of(self)]:
        return DefBoxInferenceIter[origin_of(self)](Pointer(to=self))


@fieldwise_init
struct DefBoxInferenceIter[
    origin: ImmutOrigin,
]:
    @implicit
    fn __init__(out self, regex: Pointer[DefBoxInference, Self.origin]):
        pass
