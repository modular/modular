# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s

# ===----------------------------------------------------------------------=== #
# Actual tests
# ===----------------------------------------------------------------------=== #


# CHECK-LABEL: lit.struct.decl @Param
@value
@register_passable("trivial")
struct Param[x: Index]:
    alias value = x

    @staticmethod
    fn foo():
        pass

    # COM: Test self type of parametric struct.
    # CHECK-LABEL: lit.fn @"self_type
    # CHECK-SAME: -> !lit.struct<[[SELF:.*]]>
    @staticmethod
    fn self_type() -> Self:
        pass


# CHECK-LABEL: lit.struct.decl @TwoParam
@value
@register_passable("trivial")
struct TwoParam[x: Index, y: Index]:
    alias first = x
    alias second = y

    @staticmethod
    fn foo():
        pass


# CHECK-LABEL: fully_bound_alias
fn fully_bound_alias():
    # COM: Test alias to a fully bound parametric type.
    # CHECK: BoundType{{.*}}: meta<!lit.struct<{{.*}}Param <1>>> = <{{.*}}@Param<1>>
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
    # CHECK: [[UNBOUND:\*"Unbound.*]]: meta<!lit.struct<{{.*}}Param <?>, <"x": index>>> = <{{.*}}@Param<?>>
    alias Unbound = Param
    # CHECK: unbound_value{{.*}} = <2>
    alias unbound_value = Unbound[`2`].value
    # CHECK: call {{.*}}@Param::@"foo()"<2>
    Unbound[`2`].foo()
    # CHECK: unbound_function{{.*}}: !lit.generator<<index, |>() -> !kgen.none> = <{{.*}}@Param::@"foo()"<?>>
    alias unbound_function = Unbound.foo

    # COM: Test fully unbound alias can be fully bound.
    # CHECK: BoundFromUnbound{{.*}}: meta<!lit.struct<#Param <1>>> =
    # CHECK-SAME: <@metatypes_param::@Param<1>>
    alias BoundFromUnbound = Unbound[`1`]


# CHECK-LABEL: partially_bound_alias
fn partially_bound_alias():
    # COM: Test partially binding a type.
    # CHECK: [[PBOUND:\*"PartiallyBound.*]]: meta<!lit.struct<#TwoParam <1, ?>, <"y": index>>> = <{{.*}}@TwoParam<1, ?>>
    alias PartiallyBound = TwoParam[`1`]

    # COM: Test taking a function from a partially bound type.
    # CHECK: [[PBOUND_FN:\*"PartiallyBoundFn.*]]: !lit.generator<<index, |>() -> !kgen.none> = <{{.*}}@TwoParam::@"foo()"<1, ?>>
    alias PartiallyBoundFn = PartiallyBound.foo
    # CHECK: FullyBoundFn{{.*}}: {{.*}} =   <@metatypes_param::@TwoParam::@"foo()"<1, 2>>
    alias FullyBoundFn = PartiallyBoundFn[`2`]

    # COM: Test fully binding a partially bound type.
    # CHECK: *"BoundFromPartial`3": meta<!lit.struct<#TwoParam <1, 2>>> =
    # CHECK-SAME: <@metatypes_param::@TwoParam<1, 2>>
    alias BoundFromPartial = PartiallyBound[`2`]
    # CHECK: first{{.*}} = <1>
    alias first = BoundFromPartial.first
    # CHECK: second{{.*}} = <2>
    alias second = BoundFromPartial.second
    # CHECK: fn_from_bound{{.*}}: !lit.generator<() -> !kgen.none> = <{{.*}}@TwoParam::@"foo()"<1, 2>>
    alias fn_from_bound = BoundFromPartial.foo


# CHECK-LABEL: partially_bound_kw
fn partially_bound_kw():
    # COM: Test partially binding the parameters out-of-order with keywords.
    # CHECK: TwoParam <?, 1>
    alias PartiallyBound = TwoParam[y=`1`]
    # CHECK: TwoParam <2, 1>
    alias FullyBound = PartiallyBound[x=`2`]

    # COM: Test emission of fully bound type.
    # CHECK: expr_type{{.*}}@TwoParam<2, 1>
    var expr_type: FullyBound


# CHECK-LABEL: lit.fn @"partial_autoparam
# CHECK-SAME: <?, [[X:.*]]>(%value: !lit.struct<#TwoParam <[[X]], 1>
fn partial_autoparam(value: TwoParam[y=`1`]):
    alias first = value.x
    alias second = value.y


# CHECK-LABEL: lit.struct.decl @ParamVarArg<F, I: variadic<index> pos_vararg>
@value
@register_passable("trivial")
struct ParamVarArg[F: Index, *I: Index]:
    # CHECK-LABEL: lit.fn @"self_type
    # CHECK-SAME: #ParamVarArg <F, :variadic<index> I>
    @staticmethod
    fn self_type() -> Self:
        # CHECK: lit.alias.decl {{.*}}Unbound{{.*}}: {{.*}}ParamVarArg <?, :variadic<index> ?>, <"F": index, "I": variadic<index> pos_vararg>>
        alias Unbound = ParamVarArg
        # CHECK: lit.alias.decl {{.*}}BoundSome{{.*}}: {{.*}}ParamVarArg <1, :variadic<index> ?>
        alias BoundSome = Unbound[`1`]
        # CHECK: lit.alias.decl {{.*}}BoundFinal{{.*}}: {{.*}}ParamVarArg <1, :variadic<index> [3, 4]>
        alias BoundFinal = BoundSome[`3`, `4`]

        # CHECK: BoundMore{{.*}}: {{.*}}ParamVarArg <1, :variadic<index> [2, 1]>
        alias BoundMore = Unbound[`1`, `2`, `1`]



@register_passable
struct ParamType[x: __mlir_type.index]:
    pass


struct DependentParam[
    a: __mlir_type.index, b: __mlir_type.index, c: ParamType[b]
]:
    pass


# CHECK-LABEL: lit.fn @"direct_binding
fn direct_binding():
    # Test direct bind of StructType
    # CHECK: alias.decl *"a{{.*}} meta<!lit.struct<[[DEP:.*]]<?, ?, :[[PT:.*]]<?> ?>, <"a": index, "b": index, "c": [[PT]]<*(0,1)>>
    alias a = DependentParam
    # CHECK: alias.decl *"b{{.*}} meta<!lit.struct<[[DEP]]<1, ?, :[[PT]]<?> ?>, <"b": index, "c": [[PT]]<*(0,0)>>>
    alias b = DependentParam[__mlir_attr.`1:index`]
    # CHECK: alias.decl *"c{{.*}} meta<!lit.struct<[[DEP]]<1, 2, :[[PT]]<2> ?>, <"c": [[PT]]<2>>>
    alias c = DependentParam[__mlir_attr.`1:index`, __mlir_attr.`2:index`]

    # Test partial bind of StructType
    # CHECK: alias.decl *"d{{.*}} meta<!lit.struct<[[DEP]]<1, 2, :[[PT]]<2> ?>, <"c": [[PT]]<2>>>
    alias d = DependentParam[__mlir_attr.`1:index`][__mlir_attr.`2:index`]


# CHECK: lit.fn @"indirect_binding
fn indirect_binding():
    # CHECK: alias.decl [[a:\*"a.*"]]: meta
    alias a = DependentParam
    # Test indirect binds.
    # CHECK: alias.decl [[b:\*"b.*"]]: meta<!lit.struct<[[DEP]]<1, ?, :[[PT]]<?> ?>, <"b": index, "c": [[PT]]<*(0,0)>{{.*}}> = <@metatypes_param::@DependentParam<1, ?, :@metatypes_param::@ParamType<?> ?>>
    alias b = a[__mlir_attr.`1:index`]
    # CHECK: alias.decl [[c:\*"c.*"]]: meta<!lit.struct<[[DEP]]<1, 2, :[[PT]]<2> ?>, <"c": [[PT]]<2>{{.*}}> = <@metatypes_param::@DependentParam<1, 2, :@metatypes_param::@ParamType<2> ?>>
    alias c = b[__mlir_attr.`2:index`]
    # CHECK: alias.decl [[d:\*"d.*"]]: meta<!lit.struct<[[DEP]]<1, 2, :[[PT]]<2> *?>>> = <@metatypes_param::@DependentParam<1, 2, :@metatypes_param::@ParamType<2> *?>>
    alias d = c[
        __mlir_attr[`#kgen.unknown : `, ParamType[__mlir_attr.`2:index`]]
    ]
