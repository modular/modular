# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | kgen-opt -verify-parameters | FileCheck %s

# ===----------------------------------------------------------------------=== #
# Actual tests
# ===----------------------------------------------------------------------=== #


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
    # CHECK-SAME: -> !lit.struct<[[SELF:.*]]>
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
    # CHECK: BoundType{{.*}}: anystruct<{{.*}}Param <1>> = <{{.*}}@Param<1>>
    alias BoundType = Param[`1`]
    # CHECK: alias_value{{.*}} = <1>
    alias alias_value = BoundType.value
    # CHECK: call {{.*}}@Param::@"foo()"<1>
    BoundType.foo()
    # CHECK: call {{.*}}@Param::@"self_type()"<1>{{.*}} -> !lit.struct<#Param <1>>
    _ = BoundType.self_type()


# CHECK-LABEL: unbound_alias
fn unbound_alias():
    # COM: Test alias to a fully unbound parametric type.
    # CHECK: [[UNBOUND:\*"Unbound.*]]: anystruct<{{.*}}Param <?>, <"x": index>> = <{{.*}}@Param<?>>
    alias Unbound = Param
    # CHECK: unbound_value{{.*}} = <2>
    alias unbound_value = Unbound[`2`].value
    # CHECK: call {{.*}}@Param::@"foo()"<2>
    Unbound[`2`].foo()
    # CHECK: unbound_function{{.*}}: !lit.signature<<index, |>() -> !kgen.none> = <{{.*}}@Param::@"foo()"<?>>
    alias unbound_function = Unbound.foo

    # COM: Test fully unbound alias can be fully bound.
    # CHECK: BoundFromUnbound{{.*}}: anystruct<#Param <1>> =
    # CHECK-SAME: #lit.bind_type<:anystruct<#Param <?>, <"x": index>> {{.*}}[[UNBOUND]], [1]>
    alias BoundFromUnbound = Unbound[`1`]


# CHECK-LABEL: partially_bound_alias
fn partially_bound_alias():
    # COM: Test partially binding a type.
    # CHECK: [[PBOUND:\*"PartiallyBound.*]]: anystruct<#TwoParam <1, ?>, <"y": index>> = <{{.*}}@TwoParam<1, ?>>
    alias PartiallyBound = TwoParam[`1`]

    # COM: Test taking a function from a partially bound type.
    # CHECK: [[PBOUND_FN:\*"PartiallyBoundFn.*]]: !lit.signature<<index, |>() -> !kgen.none> = <{{.*}}@TwoParam::@"foo()"<1, ?>>
    alias PartiallyBoundFn = PartiallyBound.foo
    # CHECK: FullyBoundFn{{.*}}: {{.*}} = <bind_signature({{.*}}[[PBOUND_FN]], 2)>
    alias FullyBoundFn = PartiallyBoundFn[`2`]

    # COM: Test fully binding a partially bound type.
    # CHECK: *"BoundFromPartial`3": anystruct<#TwoParam <1, 2>> =
    # CHECK-SAME: #lit.bind_type<:anystruct<#TwoParam <1, ?>, <"y": index>> {{.*}}[[PBOUND]], [2]>
    alias BoundFromPartial = PartiallyBound[`2`]
    # CHECK: first{{.*}} = <1>
    alias first = BoundFromPartial.first
    # CHECK: second{{.*}} = <2>
    alias second = BoundFromPartial.second
    # CHECK: fn_from_bound{{.*}}: !lit.signature<() -> !kgen.none> = <{{.*}}@TwoParam::@"foo()"<1, 2>>
    alias fn_from_bound = BoundFromPartial.foo


# CHECK-LABEL: partially_bound_kw
fn partially_bound_kw():
    # COM: Test partially binding the parameters out-of-order with keywords.
    # CHECK: TwoParam <?, 1>
    alias PartiallyBound = TwoParam[y=`1`]
    # CHECK: TwoParam <2, 1>
    alias FullyBound = PartiallyBound[x=`2`]

    # COM: Test emission of fully bound type.
    # CHECK: :anystruct<#TwoParam <2, 1>> {{.*}}FullyBound
    var expr_type: FullyBound


# CHECK-LABEL: lit.func @"partial_autoparam
# CHECK-SAME: <?, [[X:.*]]>(%value: !lit.struct<#TwoParam <[[X]], 1>
fn partial_autoparam(value: TwoParam[y=`1`]):
    alias first = value.x
    alias second = value.y


# CHECK-LABEL: lit.struct.decl @ParamVarArg<F, I: variadic<index> var>
@value
@register_passable("trivial")
struct ParamVarArg[F: int, *I: int]:
    # CHECK-LABEL: lit.func @"self_type
    # CHECK-SAME: #ParamVarArg <F, :variadic<index> I>
    @staticmethod
    fn self_type() -> Self:
        # CHECK: Unbound{{.*}}: {{.*}}ParamVarArg <?, :variadic<index> ?>, <"F": index, "I": variadic<index> var>>
        alias Unbound = ParamVarArg
        # CHECK: BoundSome{{.*}}: {{.*}}ParamVarArg <1, :variadic<index> []>
        # CHECK: BoundMore{{.*}}: {{.*}}ParamVarArg <1, :variadic<index> [2, 1]>
        alias BoundSome = Unbound[`1`]
        alias BoundMore = Unbound[`1`, `2`, `1`]


@register_passable
struct ParamType[x: __mlir_type.index]:
    pass


struct DependentParam[
    a: __mlir_type.index, b: __mlir_type.index, c: ParamType[b]
]:
    pass


# CHECK-LABEL: lit.func @"direct_binding
fn direct_binding():
    # Test direct bind of StructType
    # CHECK: alias.decl *"a{{.*}} anystruct<[[DEP:.*]]<?, ?, :[[PT:.*]]<?> ?>, <"a": index, "b": index, "c": [[PT]]<*(0,1)>
    alias a = DependentParam
    # CHECK: alias.decl *"b{{.*}} anystruct<[[DEP]]<1, ?, :[[PT]]<?> ?>, <"b": index, "c": [[PT]]<*(0,0)>
    alias b = DependentParam[__mlir_attr.`1:index`]
    # CHECK: alias.decl *"c{{.*}} anystruct<[[DEP]]<1, 2, :[[PT]]<2> ?>, <"c": [[PT]]<2>
    alias c = DependentParam[__mlir_attr.`1:index`, __mlir_attr.`2:index`]

    # Test partial bind of StructType
    # CHECK: alias.decl *"d{{.*}} anystruct<[[DEP]]<1, 2, :[[PT]]<2> ?>, <"c": [[PT]]<2>
    alias d = DependentParam[__mlir_attr.`1:index`][__mlir_attr.`2:index`]


# CHECK: lit.func @"indirect_binding
fn indirect_binding():
    # CHECK: alias.decl [[a:\*"a.*"]]: anystruct
    alias a = DependentParam
    # Test indirect binds.
    # CHECK: alias.decl [[b:\*"b.*"]]: anystruct<[[DEP]]<1, ?, :[[PT]]<?> ?>, <"b": index, "c": [[PT]]<*(0,0)>{{.*}} = <#lit.bind_type<{{.*}} [[a]], [1, ?, ?]>>
    alias b = a[__mlir_attr.`1:index`]
    # CHECK: alias.decl [[c:\*"c.*"]]: anystruct<[[DEP]]<1, 2, :[[PT]]<2> ?>, <"c": [[PT]]<2>{{.*}} = <#lit.bind_type<{{.*}} [[b]], [2, ?]>>
    alias c = b[__mlir_attr.`2:index`]
    # CHECK: alias.decl [[d:\*"d.*"]]: anystruct<[[DEP]]<1, 2, :[[PT]]<2> *?>> = <#lit.bind_type<{{.*}} [[c]], [*?]>>
    alias d = c[
        __mlir_attr[`#kgen.unknown : `, ParamType[__mlir_attr.`2:index`]]
    ]
