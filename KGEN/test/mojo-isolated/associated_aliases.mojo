# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated --mojo-disable-builtins -split-input-file %s | FileCheck %s


# Tests that we correctly call get_vtable_entry when looking up a trait's alias.


@fieldwise_init
@register_passable("trivial")
struct ZInt:
    pass


# CHECK-LABEL: lit.trait.decl @TraitWithAlias
trait TraitWithAlias:
    # CHECK-NEXT: lit.alias.decl *"N`1": !ZInt
    alias N: ZInt


struct StructWithMatchingAlias(TraitWithAlias):
    alias N: ZInt = ZInt()

    fn __init__(out self):
        pass


# CHECK-LABEL: lit.fn @"getNFromTraitWithAlias
fn getNFromTraitWithAlias[T: TraitWithAlias](t: T) -> ZInt:
    # CHECK-NEXT: lit.alias.decl [[X:.*]]: !ZInt = <get_vtable_entry(:!TraitWithAlias T, "N")>
    # CHECK-NEXT: kgen.param.constant: !ZInt = <get_vtable_entry(:!TraitWithAlias T, "N")>
    alias X = T.N
    return X


# // -----

# Tests that we create a #kgen.type for StructWithMatchingAlias for
# TraitWithAlias, and it contains an entry for `N` of the right type.`

# CHECK-DAG: #[[StructWithMatchingAlias_VTable:.*]] = #kgen.type<!StructWithMatchingAlias, {"N" : !ZInt = {{.*}} : !TraitWithAlias


@fieldwise_init
@register_passable("trivial")
struct ZInt:
    pass


# CHECK-LABEL: lit.trait.decl @TraitWithAlias
trait TraitWithAlias:
    # CHECK-NEXT: lit.alias.decl *"N`1": !ZInt
    alias N: ZInt


struct StructWithMatchingAlias(TraitWithAlias):
    alias N: ZInt = ZInt()

    fn __init__(out self):
        pass


# CHECK-LABEL: lit.fn @"getNFromTraitWithAlias
fn getNFromTraitWithAlias[T: TraitWithAlias](t: T) -> ZInt:
    # CHECK-NEXT: lit.alias.decl [[X:.*]]: !ZInt = <get_vtable_entry(:!TraitWithAlias T, "N")>
    # CHECK-NEXT: kgen.param.constant: !ZInt = <get_vtable_entry(:!TraitWithAlias T, "N")>
    alias X = T.N
    return X


# CHECK-LABEL: lit.fn export @"testTraitWithAliasAndStructWithMatchingAlias
@export
fn testTraitWithAliasAndStructWithMatchingAlias():
    # CHECK: {{.*}} = lit.call @associated_aliases::@"getNFromTraitWithAlias{{.*}}<:!TraitWithAlias #[[StructWithMatchingAlias_VTable]]>(%1)
    _ = getNFromTraitWithAlias(StructWithMatchingAlias())


# // -----

# Tests that we correctly call get_vtable_entry when looking up a trait's alias,
# even when we're looking up an alias that originally came from a grandparent.


@fieldwise_init
@register_passable("trivial")
struct ZInt:
    pass


# CHECK-LABEL: lit.trait.decl @TraitWithAlias
trait TraitWithAlias:
    # CHECK-NEXT: lit.alias.decl *"N`1": !ZInt
    alias N: ZInt


trait TraitWithTypeAlias:
    alias T: TraitWithAlias


trait TraitWithSameTypeAlias(TraitWithTypeAlias):
    # TODO(MOCO-1992): Make it so we can omit this.
    alias T: TraitWithAlias


# CHECK-LABEL: lit.fn @"testTraitWithRefinedTypeAlias
fn testTraitWithRefinedTypeAlias[T: TraitWithSameTypeAlias]():
    # CHECK-NEXT: !TraitWithAlias = <get_vtable_entry(:!TraitWithSameTypeAlias T, "T")>
    alias MyT: TraitWithAlias = T.T


# // -----

# Tests that a trait can have a method that returns a generic struct with an
# input parameter-value that's a trait alias.

# CHECK-DAG: #[[ExplicitStructWithAliasMethod_VTable:.*]] = #kgen.type<!ExplicitStructWithAliasMethod, {"T" : !ATrait = !ZInt{{.*}} : !TraitWithAliasReturnMethod


@fieldwise_init
@register_passable("trivial")
struct ZInt:
    pass


trait ATrait:
    pass


@value
struct SIMD[T: ATrait]:
    pass


trait TraitWithAliasReturnMethod:
    alias T: ATrait

    fn bork(self) -> SIMD[T]:
        ...


# CHECK-LABEL: lit.struct.decl @ExplicitStructWithAliasMethod
@value
struct ExplicitStructWithAliasMethod(TraitWithAliasReturnMethod):
    # TODO(MOCO-1993): Make it so we don't have to say `: ATrait` here.
    alias T: ATrait = ZInt

    fn bork(self) -> SIMD[ZInt]:
        return SIMD[ZInt]()


# CHECK-LABEL: lit.fn @"testUpcastingExplicitStructWithAliasMethod
fn testUpcastingExplicitStructWithAliasMethod():
    # CHECK:       {{.*}}lit.call @associated_aliases::@"receiveTraitWithAliasReturnMethod{{.*}}<:!TraitWithAliasReturnMethod #[[ExplicitStructWithAliasMethod_VTable]]>
    receiveTraitWithAliasReturnMethod(ExplicitStructWithAliasMethod())


fn receiveTraitWithAliasReturnMethod[X: TraitWithAliasReturnMethod](t: X):
    pass


# // -----

# Tests that a trait can have a method that returns a generic struct with an
# input parameter-value that's a trait alias. Does it with implicit conformance.
# TODO: Once implicit conformance is gone, we can remove this test.

# CHECK-DAG: #[[ImplicitStructWithAliasMethod_VTable:.*]] = #kgen.type<!ImplicitStructWithAliasMethod, {"T" : !ATrait = !ZInt{{.*}} : !TraitWithAliasReturnMethod


@fieldwise_init
@register_passable("trivial")
struct ZInt:
    pass


trait ATrait:
    pass


@value
struct SIMD[T: ATrait]:
    pass


trait TraitWithAliasReturnMethod:
    alias T: ATrait

    fn bork(self) -> SIMD[T]:
        ...


# CHECK-LABEL: lit.struct.decl @ImplicitStructWithAliasMethod
@value
struct ImplicitStructWithAliasMethod:
    alias T: ATrait = ZInt

    fn bork(self) -> SIMD[ZInt]:
        return SIMD[ZInt]()


# CHECK-LABEL: lit.fn @"testUpcastingImplicitStructWithAliasMethod
fn testUpcastingImplicitStructWithAliasMethod():
    # CHECK: {{.*}}lit.call @associated_aliases::@"receiveTraitWithAliasReturnMethod{{.*}}<:!TraitWithAliasReturnMethod #[[ImplicitStructWithAliasMethod_VTable]]>
    receiveTraitWithAliasReturnMethod(ImplicitStructWithAliasMethod())


fn receiveTraitWithAliasReturnMethod[X: TraitWithAliasReturnMethod](t: X):
    pass


# // -----

# Tests that we can call an alias-returning method on a given trait instance.


trait ATrait:
    pass


@value
struct SIMD[T: ATrait]:
    pass


trait TraitWithAliasReturnMethod:
    alias T: ATrait

    fn bork(self) -> SIMD[T]:
        ...


# CHECK-LABEL: lit.fn @"callTraitWithAliasReturnMethod
fn callTraitWithAliasReturnMethod[X: TraitWithAliasReturnMethod](t: X):
    # CHECK: {{.*}}lit.call
    # CHECK-SAME: "__result__": !lit.ref<@associated_aliases::@SIMD<:!ATrait get_vtable_entry(:!TraitWithAliasReturnMethod X, "T")>
    # CHECK-SAME: : get_vtable_entry(:!TraitWithAliasReturnMethod X, "bork")
    _ = t.bork()


# // -----

# Tests that we can upcast a generic struct to a trait, when the generic struct
# uses an input-parameter in a method override for a trait method that mentions
# a trait alias in the return.

# CHECK-DAG: #[[GenericStructWithAliasMethod_VTable:.*]] = #kgen.type<@associated_aliases::@GenericStructWithAliasMethod<:!ATrait !ZInt>, {"T" : !ATrait = !ZInt, "bork" : !lit.generator<[2]("self": {{.*}}, "__result__": !lit.ref<@associated_aliases::@SIMD<:!ATrait !ZInt>, mut *[0,1]> byref_result{{.*}} : !TraitWithAliasReturnMethod


@fieldwise_init
@register_passable("trivial")
struct ZInt:
    pass


trait ATrait:
    pass


@value
struct SIMD[T: ATrait]:
    pass


trait TraitWithAliasReturnMethod:
    alias T: ATrait

    fn bork(self) -> SIMD[T]:
        ...


# TODO(MOCO-1109): also check that this works with the thunk generation for @register_passable methods
@value
struct GenericStructWithAliasMethod[Z: ATrait](TraitWithAliasReturnMethod):
    alias T: ATrait = Z

    fn bork(self) -> SIMD[Z]:
        return SIMD[Z]()


# CHECK-LABEL: lit.fn @"testUpcastingGenericStructWithAliasMethod
fn testUpcastingGenericStructWithAliasMethod():
    # CHECK: {{.*}}lit.call @associated_aliases::@"receiveTraitWithAliasReturnMethod{{.*}}<:!TraitWithAliasReturnMethod #[[GenericStructWithAliasMethod_VTable]]>
    receiveTraitWithAliasReturnMethod(GenericStructWithAliasMethod[ZInt]())


fn receiveTraitWithAliasReturnMethod[X: TraitWithAliasReturnMethod](t: X):
    pass


# // -----

# Tests that we can upcast a struct to a trait, when the trait method mentions
# a trait alias in the return type, specifically with `Self.`.


@fieldwise_init
@register_passable("trivial")
struct ZInt:
    pass


trait ATrait:
    pass


@value
struct SIMD[T: ATrait]:
    pass


# Tests explicit mentions of `Self.thing` where `thing` is an associated alias.
# See https://linear.app/modularml/issue/MOCO-1438
trait TraitWithSelfDotAliasReturnMethod:
    alias T: ATrait

    fn bork(self) -> SIMD[Self.T]:
        ...


struct StructWithSelfDotAliasReturnMethod(TraitWithSelfDotAliasReturnMethod):
    alias T: ATrait = ZInt

    fn bork(self) -> SIMD[Self.T]:
        return SIMD[Self.T]()


fn receiveTraitWithSelfDotAliasReturnMethod[
    T: TraitWithSelfDotAliasReturnMethod
](z: T):
    _ = z.bork()


fn upcastStructWithSelfDotAliasReturnMethod(
    x: StructWithSelfDotAliasReturnMethod,
):
    receiveTraitWithSelfDotAliasReturnMethod(x)


# // -----

# Tests that we can upcast a generic struct to a trait, when the generic struct
# uses an input-parameter in a method override for a trait method that mentions
# a trait alias in the return.
# This is like the above test, but explicitly mentions `Self.` in the return.


@fieldwise_init
@register_passable("trivial")
struct ZInt:
    pass


trait ATrait:
    pass


@value
struct SIMD[T: ATrait]:
    pass


# Tests explicit mentions of `Self.thing` where `thing` is an associated alias.
# See https://linear.app/modularml/issue/MOCO-1438
trait TraitWithSelfDotAliasReturnMethod:
    alias T: ATrait

    fn bork(self) -> SIMD[Self.T]:
        ...


# TODO(MOCO-1109): also check that this works with the thunk generation for @register_passable methods
@value
struct GenericStructWithSelfDotAliasReturnMethod[Z: ATrait](
    TraitWithSelfDotAliasReturnMethod
):
    alias T: ATrait = Z

    fn bork(self) -> SIMD[Self.Z]:
        return SIMD[Z]()


fn receiveTraitWithSelfDotAliasReturnMethod[
    T: TraitWithSelfDotAliasReturnMethod
](z: T):
    _ = z.bork()


fn testUpcastingGenericStructWithSelfDotAliasReturnMethod():
    receiveTraitWithSelfDotAliasReturnMethod(
        GenericStructWithSelfDotAliasReturnMethod[ZInt]()
    )


# // -----

# Tests that we can upcast a generic struct to a trait, when the generic struct
# uses an input-parameter in a method override for a trait method that mentions
# a trait alias in an argument.

# CHECK-DAG: #[[StructWithAliasArgMethod_VTable:.*]] = #kgen.type<!StructWithAliasArgMethod,{{.*}}"lork" : !lit.generator<{{.*}}"thing": !lit.ref<@associated_aliases::@SIMD<:!ATrait !ZInt>{{.*}}> = @associated_aliases::@StructWithAliasArgMethod::@"lork({{.*}}SIMD[associated_aliases::ZInt])"}> : !TraitWithAliasArgMethod


@fieldwise_init
@register_passable("trivial")
struct ZInt:
    pass


trait ATrait:
    pass


@value
struct SIMD[T: ATrait]:
    pass


trait TraitWithAliasArgMethod:
    alias T: ATrait

    fn lork(self, thing: SIMD[T]):
        ...


@value
struct StructWithAliasArgMethod(TraitWithAliasArgMethod):
    alias T: ATrait = ZInt

    fn lork(self, thing: SIMD[ZInt]):
        pass


fn receiveTraitWithAliasArgMethod[X: TraitWithAliasArgMethod](t: X):
    pass


# CHECK-LABEL: lit.fn @"testUpcastingStructWithAliasArgMethod
fn testUpcastingStructWithAliasArgMethod():
    # CHECK: {{.*}}lit.call @associated_aliases::@"receiveTraitWithAliasArgMethod{{.*}}<:!TraitWithAliasArgMethod #[[StructWithAliasArgMethod_VTable]]
    receiveTraitWithAliasArgMethod(StructWithAliasArgMethod())


# // -----

# Tests that we can call a trait's method, when it mentions a trait alias in
# an argument type.


@fieldwise_init
@register_passable("trivial")
struct ZInt:
    pass


trait ATrait:
    pass


@value
struct SIMD[T: ATrait]:
    pass


trait TraitWithAliasArgMethod:
    alias T: ATrait

    fn lork(self, thing: SIMD[T]):
        ...


# CHECK-LABEL: lit.fn @"callTraitMethodWithAliasArg
fn callTraitMethodWithAliasArg[
    X: TraitWithAliasArgMethod
](t: X, thing: SIMD[X.T]):
    # CHECK:  %0 = lit.call
    # CHECK-SAME: "thing": !lit.ref<@associated_aliases::@SIMD<:!ATrait get_vtable_entry(:!TraitWithAliasArgMethod X, "T")>
    # CHECK-SAME: : get_vtable_entry(:!TraitWithAliasArgMethod X, "lork")
    t.lork(thing)


# TODO(MOCO-1259): Support static methods with associated aliases

# TODO(MOCO-1143): Make this work:
# struct StructWithParam[X: ZInt]:
#     pass
# # HECK-LABEL: lit.trait.decl @Spork<Self: type> {
# trait TraitWithStaticMethodUsingAlias:
#     # HECK-NEXT: lit.alias.decl N = <?>
#     alias N: ZInt
#     # HECK-LABEL: lit.fn @foo(%x: !pop.simd<N, f32>) { // #kgen.param.decl.ref<"N"> : index
#     @staticmethod
#     fn foo(x: StructWithParam[N]):
#         pass
# struct StructWithStaticMethod:
#     @staticmethod
#     fn foo(x: StructWithParam[5]):
#         pass
# fn sporkify[T: TraitWithStaticMethodUsingAlias]() -> ZInt:
#    return T.N # emits a get_vtable_value
# @export
# fn testSomething():
#     # And maybe add a test for sporkify[TraitWithStaticMethodUsingAlias]()
#     sporkify[StructWithStaticMethod]()
