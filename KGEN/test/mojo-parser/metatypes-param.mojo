# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate -import-mojo --mojo-disable-builtins %s | FileCheck %s

alias int = __mlir_type.index
alias one = __mlir_attr.`1:index`
alias two = __mlir_attr.`2:index`


# CHECK-LABEL: lit.struct.decl @Param
@value
@register_passable("trivial")
struct Param[x: int]:
    alias value = x

    @staticmethod
    fn foo():
        pass

    # COM: Test self type of parametric struct.
    # CHECK-LABEL: lit.func @"self_type
    # CHECK-SAME: -> !kgen.declref<[[SELF:.*]], !lit.metatype<[[SELF]]>>
    @staticmethod
    fn self_type() -> Self:
        pass


# CHECK-LABEL: lit.struct.decl @TwoParam
@value
@register_passable("trivial")
struct TwoParam[x: int, y: int]:
    alias first = x
    alias second = y

    @staticmethod
    fn foo():
        pass


# CHECK-LABEL: fully_bound_alias
fn fully_bound_alias():
    # COM: Test alias to a fully bound parametric type.
    # CHECK: BoundType: metatype<{{.*}}@Param<1>> = <{{.*}}@Param<1>>
    alias BoundType = Param[one]
    # CHECK: alias_value = <1>
    alias alias_value = BoundType.value
    # CHECK: call {{.*}}@Param::@"foo()"<1>
    BoundType.foo()
    # CHECK: call {{.*}}@Param::@"self_type()"<1>{{.*}} -> !kgen.declref<{{.*}}@Param<1>, !lit.metatype<{{.*}}@Param<1>>
    _ = BoundType.self_type()


# CHECK-LABEL: unbound_alias
fn unbound_alias():
    # COM: Test alias to a fully unbound parametric type.
    # CHECK: Unbound: metatype<{{.*}}@Param<?>, <"x": index>> = <{{.*}}@Param<?>>
    alias Unbound = Param
    # CHECK: unbound_value = <2>
    alias unbound_value = Unbound[two].value
    # CHECK: call {{.*}}@Param::@"foo()"<2>
    Unbound[two].foo()
    # CHECK: unbound_function: !lit.signature<<index, |>() -> !kgen.none> = <{{.*}}@Param::@"foo()"<?>>
    alias unbound_function = Unbound.foo

    # COM: Test fully unbound alias can be fully bound.
    # CHECK: BoundFromUnbound: metatype<{{.*}}@Param<1>> =
    # CHECK-SAME: #lit.bind_type<:metatype<{{.*}}@Param<?>, <"x": index>> {{.*}}Unbound, [1]>
    alias BoundFromUnbound = Unbound[one]


# CHECK-LABEL: partially_bound_alias
fn partially_bound_alias():
    # COM: Test partially binding a type.
    # CHECK: PartiallyBound: metatype<{{.*}}@TwoParam<1, ?>, <"y": index>> = <{{.*}}@TwoParam<1, ?>>
    alias PartiallyBound = TwoParam[one]

    # COM: Test taking a function from a partially bound type.
    # CHECK: PartiallyBoundFn: !lit.signature<<index, |>() -> !kgen.none> = <{{.*}}@TwoParam::@"foo()"<1, ?>>
    alias PartiallyBoundFn = PartiallyBound.foo
    # CHECK: FullyBoundFn: {{.*}} = <bind_signature({{.*}}PartiallyBoundFn, 2)>
    alias FullyBoundFn = PartiallyBoundFn[two]

    # COM: Test fully binding a partially bound type.
    # CHECK: BoundFromPartial: metatype<{{.*}}@TwoParam<1, 2>> =
    # CHECK-SAME: #lit.bind_type<:metatype<{{.*}}@TwoParam<1, ?>, <"y": index>> {{.*}}PartiallyBound, [2]>
    alias BoundFromPartial = PartiallyBound[two]
    # CHECK: first = <1>
    alias first = BoundFromPartial.first
    # CHECK: second = <2>
    alias second = BoundFromPartial.second
    # CHECK: fn_from_bound: !lit.signature<() -> !kgen.none> = <{{.*}}@TwoParam::@"foo()"<1, 2>>
    alias fn_from_bound = BoundFromPartial.foo


# CHECK-LABEL: partially_bound_kw
fn partially_bound_kw():
    # COM: Test partially binding the parameters out-of-order with keywords.
    # CHECK: TwoParam<?, 1>
    alias PartiallyBound = TwoParam[y=one]
    # CHECK: TwoParam<2, 1>
    alias FullyBound = PartiallyBound[x=two]

    # COM: Test emission of fully bound type.
    # CHECK: :metatype<{{.*}}@TwoParam<2, 1>> {{.*}}FullyBound
    var expr_type: FullyBound


# CHECK-LABEL: lit.func @"partial_autoparam
# CHECK-SAME: <[[X:.*]][{{.*}}]>(%value[value]: !kgen.declref<{{.*}}@TwoParam<[[X]], 1>
fn partial_autoparam(value: TwoParam[y=one]):
    alias first = value.x
    alias second = value.y


# CHECK-LABEL: lit.struct.decl @ParamVarArg
# CHECK-SAME: <[[F:.*]][F], [[I:.*]][I]
@value
@register_passable("trivial")
struct ParamVarArg[F: int, *I: int]:
    # CHECK-LABEL: lit.func @"self_type
    # CHECK-SAME: @ParamVarArg<[[F]], :variadic<index> [[I]]>, !lit.metatype<{{.*}}@ParamVarArg<[[F]], :variadic<index> [[I]]>>
    @staticmethod
    fn self_type() -> Self:
        # CHECK: Unbound: {{.*}}@ParamVarArg<?, :variadic<index> ?>, <"F": index, "I": variadic<index>> param_vararg>
        alias Unbound = ParamVarArg
        # CHECK: BoundSome: {{.*}}@ParamVarArg<1, :variadic<index> []>
        # CHECK: BoundMore: {{.*}}@ParamVarArg<1, :variadic<index> [2, 1]>
        alias BoundSome = Unbound[one]
        alias BoundMore = Unbound[one, two, one]
