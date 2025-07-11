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


# CHECK-LABEL: lit.struct.decl @StructWithMatchingAlias
struct StructWithMatchingAlias(TraitWithAlias):
    # CHECK: kgen.conformance {{.*}}::TraitWithAlias
    # CHECK: kgen.witness "N" : !ZInt = apply{{.*}}ZInt::@"__init__()"
    alias N: ZInt = ZInt()

    fn __init__(out self):
        pass


# This tests that we correctly call get_vtable_entry when looking up a trait's
# alias.
# CHECK-LABEL: lit.fn @"getNFromTraitWithAlias
fn getNFromTraitWithAlias[T: TraitWithAlias](t: T) -> ZInt:
    # CHECK-NEXT: lit.alias.decl [[X:.*]]: !ZInt = <#kgen.get_witness<#kgen.param.decl.ref<"T"> : !TraitWithAlias, "associated_aliases::TraitWithAlias", "N">>
    # CHECK-NEXT: kgen.param.constant: !ZInt = <#kgen.get_witness<#kgen.param.decl.ref<"T"> : !TraitWithAlias, "associated_aliases::TraitWithAlias", "N">>
    alias X = T.N
    return X


# This tests that we create a #kgen.type for StructWithMatchingAlias for
# TraitWithAlias, and it contains an entry for `N` of the right type.`
# CHECK-LABEL: lit.fn export @"testTraitWithAliasAndStructWithMatchingAlias
@export
fn testTraitWithAliasAndStructWithMatchingAlias():
    # CHECK: {{.*}} = lit.call @associated_aliases::@"getNFromTraitWithAlias{{.*}}<:!TraitWithAlias !StructWithMatchingAlias>(%1)
    _ = getNFromTraitWithAlias(StructWithMatchingAlias())


# // -----

# Tests that we correctly call get_vtable_entry when looking up a trait's alias,
# even when we're looking up an alias that originally came from a grandparent.
# (See also MOCO-1992)


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
    pass


# CHECK-LABEL: lit.fn @"testTraitWithRefinedTypeAlias
fn testTraitWithRefinedTypeAlias[T: TraitWithSameTypeAlias]():
    # CHECK-NEXT: !TraitWithAlias = <#kgen.get_witness<#kgen.param.decl.ref<"T"> : !TraitWithSameTypeAlias, "associated_aliases::TraitWithSameTypeAlias", "T">>
    alias MyT: TraitWithAlias = T.T


# // -----

# Tests that we can upcast a generic struct to a trait, when the generic struct
# uses an input-parameter in a method override for a trait method that mentions
# a trait alias in an argument.


@fieldwise_init
@register_passable("trivial")
struct ZInt(ATrait):
    pass


trait ATrait:
    pass


struct SIMD[T: ATrait]:
    pass


trait TraitWithAliasArgMethod:
    alias T: ATrait

    fn lork(self, thing: SIMD[T]):
        ...


@fieldwise_init
struct StructWithAliasArgMethod(TraitWithAliasArgMethod):
    # CHECK: kgen.conformance {{.*}}::TraitWithAliasArgMethod
    # CHECK: kgen.witness "T" : !ATrait = !ZInt
    # CHECK: kgen.witness "lork" : {{.*}} = {{.*}}::@StructWithAliasArgMethod::@"lork(
    alias T: ATrait = ZInt

    fn lork(self, thing: SIMD[ZInt]):
        pass


fn receiveTraitWithAliasArgMethod[X: TraitWithAliasArgMethod](t: X):
    pass


# CHECK-LABEL: lit.fn @"testUpcastingStructWithAliasArgMethod
fn testUpcastingStructWithAliasArgMethod():
    # CHECK: {{.*}}lit.call @associated_aliases::@"receiveTraitWithAliasArgMethod{{.*}}<:!TraitWithAliasArgMethod !StructWithAliasArgMethod>
    receiveTraitWithAliasArgMethod(StructWithAliasArgMethod())


# // -----

# Tests that we can upcast a generic struct to a trait, when the generic struct
# uses an input-parameter in a STATIC method override for a trait method that
# mentions a trait alias in an argument. (Similar to the last test but static)


@fieldwise_init
@register_passable("trivial")
struct ZInt(ATrait):
    pass


trait ATrait:
    pass


struct SIMD[T: ATrait]:
    pass


trait TraitWithAliasArgMethod:
    alias T: ATrait

    @staticmethod
    fn lork(thing: SIMD[T]):
        ...


@fieldwise_init
struct StructWithAliasArgMethod(TraitWithAliasArgMethod):
    alias T: ATrait = ZInt

    @staticmethod
    fn lork(thing: SIMD[ZInt]):
        pass


fn receiveTraitWithAliasArgMethod[X: TraitWithAliasArgMethod](t: X):
    pass


# CHECK-LABEL: lit.fn @"testUpcastingStructWithAliasArgMethod
fn testUpcastingStructWithAliasArgMethod():
    # CHECK: {{.*}}lit.call @associated_aliases::@"receiveTraitWithAliasArgMethod{{.*}}<:!TraitWithAliasArgMethod !StructWithAliasArgMethod>
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
    # CHECK-SAME: "thing": !lit.ref<@associated_aliases::@SIMD<:!ATrait #kgen.get_witness<#kgen.param.decl.ref<"X"> : !TraitWithAliasArgMethod, "associated_aliases::TraitWithAliasArgMethod", "T">>
    # CHECK-SAME: #kgen.get_witness<#kgen.param.decl.ref<"X"> : !TraitWithAliasArgMethod, "associated_aliases::TraitWithAliasArgMethod", "lork">
    t.lork(thing)


# // -----

# Tests that we can call a trait's STATIC method, when it mentions a trait alias
# in an argument type. Similar to the previous test but with static.


@fieldwise_init
@register_passable("trivial")
struct ZInt:
    pass


trait ATrait:
    pass


struct SIMD[T: ATrait]:
    pass


trait TraitWithAliasArgMethod:
    alias T: ATrait

    @staticmethod
    fn lork(thing: SIMD[T]):
        ...


# CHECK-LABEL: lit.fn @"callTraitMethodWithAliasArg
fn callTraitMethodWithAliasArg[
    X: TraitWithAliasArgMethod
](t: X, thing: SIMD[X.T]):
    # CHECK:  %0 = lit.call
    # CHECK-SAME: "thing": !lit.ref<@associated_aliases::@SIMD<:!ATrait #kgen.get_witness<#kgen.param.decl.ref<"X"> : !TraitWithAliasArgMethod, "associated_aliases::TraitWithAliasArgMethod", "T">>
    # CHECK-SAME: #kgen.get_witness<#kgen.param.decl.ref<"X"> : !TraitWithAliasArgMethod, "associated_aliases::TraitWithAliasArgMethod", "lork">
    t.lork(thing)


# // -----

# Tests that a trait can have a method that returns a generic struct with an
# input parameter-value that's a trait alias.


@fieldwise_init
@register_passable("trivial")
struct ZInt(ATrait):
    pass


trait ATrait:
    pass


@fieldwise_init
struct SIMD[T: ATrait]:
    pass


trait TraitWithAliasReturnMethod:
    alias T: ATrait

    fn bork(self) -> SIMD[T]:
        ...


# CHECK-LABEL: lit.struct.decl @ExplicitStructWithAliasMethod
@fieldwise_init
struct ExplicitStructWithAliasMethod(TraitWithAliasReturnMethod):
    # CHECK: kgen.conformance {{.*}}::TraitWithAliasReturnMethod
    # CHECK: kgen.witness "T" : !ATrait = !ZInt
    # CHECK: kgen.witness "bork" : {{.*}} = {{.*}}::@ExplicitStructWithAliasMethod::@"bork(
    alias T: ATrait = ZInt

    fn bork(self) -> SIMD[ZInt]:
        return SIMD[ZInt]()


# CHECK-LABEL: lit.fn @"testUpcastingExplicitStructWithAliasMethod
fn testUpcastingExplicitStructWithAliasMethod():
    # CHECK:       {{.*}}lit.call @associated_aliases::@"receiveTraitWithAliasReturnMethod{{.*}}<:!TraitWithAliasReturnMethod !ExplicitStructWithAliasMethod>
    receiveTraitWithAliasReturnMethod(ExplicitStructWithAliasMethod())


fn receiveTraitWithAliasReturnMethod[X: TraitWithAliasReturnMethod](t: X):
    pass


# // -----

# Tests that a trait can have a STATIC method that returns a generic struct with
# an input parameter-value that's a trait alias. Similar to the previous test
# but static.


@fieldwise_init
@register_passable("trivial")
struct ZInt(ATrait):
    pass


trait ATrait:
    pass


@fieldwise_init
struct SIMD[T: ATrait]:
    pass


trait TraitWithAliasReturnMethod:
    alias T: ATrait

    @staticmethod
    fn bork() -> SIMD[T]:
        ...


# CHECK-LABEL: lit.struct.decl @ExplicitStructWithAliasMethod
@fieldwise_init
struct ExplicitStructWithAliasMethod(TraitWithAliasReturnMethod):
    alias T: ATrait = ZInt

    @staticmethod
    fn bork() -> SIMD[ZInt]:
        return SIMD[ZInt]()


# CHECK-LABEL: lit.fn @"testUpcastingExplicitStructWithAliasMethod
fn testUpcastingExplicitStructWithAliasMethod():
    # CHECK:       {{.*}}lit.call @associated_aliases::@"receiveTraitWithAliasReturnMethod{{.*}}<:!TraitWithAliasReturnMethod !ExplicitStructWithAliasMethod>
    receiveTraitWithAliasReturnMethod(ExplicitStructWithAliasMethod())


fn receiveTraitWithAliasReturnMethod[X: TraitWithAliasReturnMethod](t: X):
    pass


# // -----

# Tests that we can call an alias-returning method on a given trait instance.


trait ATrait:
    pass


struct SIMD[T: ATrait]:
    pass


trait TraitWithAliasReturnMethod:
    alias T: ATrait

    fn bork(self) -> SIMD[T]:
        ...


# CHECK-LABEL: lit.fn @"callTraitWithAliasReturnMethod
fn callTraitWithAliasReturnMethod[X: TraitWithAliasReturnMethod](t: X):
    # CHECK: {{.*}}lit.call
    # CHECK-SAME: "__result__": !lit.ref<@associated_aliases::@SIMD<:!ATrait #kgen.get_witness<#kgen.param.decl.ref<"X"> : !TraitWithAliasReturnMethod, "associated_aliases::TraitWithAliasReturnMethod", "T">>
    # CHECK-SAME: #kgen.get_witness<#kgen.param.decl.ref<"X"> : !TraitWithAliasReturnMethod, "associated_aliases::TraitWithAliasReturnMethod", "bork">
    _ = t.bork()


# // -----

# Tests that we can call an alias-returning STATIC method on a given trait
# instance. Similar to the previous test but with static.


trait ATrait:
    pass


struct SIMD[T: ATrait]:
    pass


trait TraitWithAliasReturnMethod:
    alias T: ATrait

    @staticmethod
    fn bork() -> SIMD[T]:
        ...


# CHECK-LABEL: lit.fn @"callTraitWithAliasReturnMethod
fn callTraitWithAliasReturnMethod[X: TraitWithAliasReturnMethod](t: X):
    # CHECK: {{.*}}lit.call
    # CHECK-SAME: "__result__": !lit.ref<@associated_aliases::@SIMD<:!ATrait #kgen.get_witness<#kgen.param.decl.ref<"X"> : !TraitWithAliasReturnMethod, "associated_aliases::TraitWithAliasReturnMethod", "T">>
    # CHECK-SAME: #kgen.get_witness<#kgen.param.decl.ref<"X"> : !TraitWithAliasReturnMethod, "associated_aliases::TraitWithAliasReturnMethod", "bork">
    _ = t.bork()


# // -----

# Tests that we can upcast a generic struct to a trait, when the generic struct
# uses an input-parameter in a method override for a trait method that mentions
# a trait alias in the return.


@fieldwise_init
@register_passable("trivial")
struct ZInt(ATrait):
    pass


trait ATrait:
    pass


@fieldwise_init
struct SIMD[T: ATrait]:
    pass


trait TraitWithAliasReturnMethod:
    alias T: ATrait

    fn bork(self) -> SIMD[T]:
        ...


# TODO(MOCO-1109): also check that this works with the thunk generation for @register_passable methods
# CHECK-LABEL: lit.struct.decl @GenericStructWithAliasMethod<Z: !ATrait>
@fieldwise_init
struct GenericStructWithAliasMethod[Z: ATrait](TraitWithAliasReturnMethod):
    # CHECK: kgen.conformance {{.*}}::TraitWithAliasReturnMethod
    # CHECK: kgen.witness "T" : !ATrait = Z
    # CHECK: kgen.witness "bork" : {{.*}} = {{.*}}::@GenericStructWithAliasMethod::@"bork(
    alias T: ATrait = Z

    fn bork(self) -> SIMD[Z]:
        return SIMD[Z]()


# CHECK-LABEL: lit.fn @"testUpcastingGenericStructWithAliasMethod
fn testUpcastingGenericStructWithAliasMethod():
    # CHECK: {{.*}}lit.call @associated_aliases::@"receiveTraitWithAliasReturnMethod{{.*}}<:!TraitWithAliasReturnMethod @associated_aliases::@GenericStructWithAliasMethod<:!ATrait !ZInt>>
    receiveTraitWithAliasReturnMethod(GenericStructWithAliasMethod[ZInt]())


fn receiveTraitWithAliasReturnMethod[X: TraitWithAliasReturnMethod](t: X):
    pass


# // -----

# Tests that we can upcast a generic struct to a trait, when the generic struct
# uses an input-parameter in a STATIC method override for a trait method that
# mentions a trait alias in the return. Same as the last test but with static.


@fieldwise_init
@register_passable("trivial")
struct ZInt(ATrait):
    pass


trait ATrait:
    pass


@fieldwise_init
struct SIMD[T: ATrait]:
    pass


trait TraitWithAliasReturnMethod:
    alias T: ATrait

    @staticmethod
    fn bork() -> SIMD[T]:
        ...


# TODO(MOCO-1109): also check that this works with the thunk generation for @register_passable methods
@fieldwise_init
struct GenericStructWithAliasMethod[Z: ATrait](TraitWithAliasReturnMethod):
    alias T: ATrait = Z

    @staticmethod
    fn bork() -> SIMD[Z]:
        return SIMD[Z]()


# CHECK-LABEL: lit.fn @"testUpcastingGenericStructWithAliasMethod
fn testUpcastingGenericStructWithAliasMethod():
    # CHECK: {{.*}}lit.call @associated_aliases::@"receiveTraitWithAliasReturnMethod{{.*}}<:!TraitWithAliasReturnMethod @associated_aliases::@GenericStructWithAliasMethod<:!ATrait !ZInt>>
    receiveTraitWithAliasReturnMethod(GenericStructWithAliasMethod[ZInt]())


fn receiveTraitWithAliasReturnMethod[X: TraitWithAliasReturnMethod](t: X):
    pass


# // -----

# Tests that we can upcast a struct to a trait, when the trait method mentions
# a trait alias in the return type, specifically with `Self.`.


@fieldwise_init
@register_passable("trivial")
struct ZInt(ATrait):
    pass


trait ATrait:
    pass


@fieldwise_init
struct SIMD[T: ATrait]:
    pass


# Tests explicit mentions of `Self.T` where `T` is an associated alias.
# See MOCO-1438.
trait TraitWithSelfDotAliasReturnMethod:
    alias T: ATrait

    fn bork(self) -> SIMD[Self.T]:
        ...


# CHECK-LABEL: lit.struct.decl @StructWithSelfDotAliasReturnMethod
struct StructWithSelfDotAliasReturnMethod(TraitWithSelfDotAliasReturnMethod):
    # CHECK: kgen.conformance {{.*}}::TraitWithSelfDotAliasReturnMethod
    # CHECK: kgen.witness "T" : !ATrait = !ZInt
    # CHECK: kgen.witness "bork" : {{.*}} = {{.*}}::@StructWithSelfDotAliasReturnMethod::@"bork(
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

# Tests that we can upcast a struct to a trait, when the trait STATIC method
# mentions a trait alias in the return type, specifically with `Self.`. Same as
# the last test but with static.


@fieldwise_init
@register_passable("trivial")
struct ZInt(ATrait):
    pass


trait ATrait:
    pass


@fieldwise_init
struct SIMD[T: ATrait]:
    pass


# Tests explicit mentions of `Self.T` where `T` is an associated alias.
# See MOCO-1438
trait TraitWithSelfDotAliasReturnMethod:
    alias T: ATrait

    @staticmethod
    fn bork() -> SIMD[Self.T]:
        ...


struct StructWithSelfDotAliasReturnMethod(TraitWithSelfDotAliasReturnMethod):
    alias T: ATrait = ZInt

    @staticmethod
    fn bork() -> SIMD[Self.T]:
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
struct ZInt(ATrait):
    pass


trait ATrait:
    pass


@fieldwise_init
struct SIMD[T: ATrait]:
    pass


# Tests explicit mentions of `Self.T` where `T` is an associated alias.
# See MOCO-1438
trait TraitWithSelfDotAliasReturnMethod:
    alias T: ATrait

    fn bork(self) -> SIMD[Self.T]:
        ...


# TODO(MOCO-1109): also check that this works with the thunk generation for @register_passable methods
# CHECK-LABEL: lit.struct.decl @GenericStructWithSelfDotAliasReturnMethod<Z: !ATrait>
@fieldwise_init
struct GenericStructWithSelfDotAliasReturnMethod[Z: ATrait](
    TraitWithSelfDotAliasReturnMethod
):
    # CHECK: kgen.conformance {{.*}}::TraitWithSelfDotAliasReturnMethod
    # CHECK: kgen.witness "T" : !ATrait = Z
    # CHECK: kgen.witness "bork" : {{.*}} = {{.*}}::@GenericStructWithSelfDotAliasReturnMethod::@"bork(
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
# uses an input-parameter in a STATIC method override for a trait method that
# mentions a trait alias in the return.
# This is like some above tests, but explicitly mentions `Self.` in the return
# and with the method being static.


@fieldwise_init
@register_passable("trivial")
struct ZInt(ATrait):
    pass


trait ATrait:
    pass


@fieldwise_init
struct SIMD[T: ATrait]:
    pass


# Tests explicit mentions of `Self.T` where `T` is an associated alias.
# See MOCO-1438
trait TraitWithSelfDotAliasReturnMethod:
    alias T: ATrait

    @staticmethod
    fn bork() -> SIMD[Self.T]:
        ...


# TODO(MOCO-1109): also check that this works with the thunk generation for @register_passable methods
@fieldwise_init
struct GenericStructWithSelfDotAliasReturnMethod[Z: ATrait](
    TraitWithSelfDotAliasReturnMethod
):
    alias T: ATrait = Z

    @staticmethod
    fn bork() -> SIMD[Self.Z]:
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

# Tests that we correctly handle substituting struct alias into the "needle"
# signature when confirming that a trait's method exists in the struct (see
# SAVMBCTATBS).


trait ATrait:
    pass


trait ASubTrait(ATrait):
    pass


@register_passable("trivial")
struct ZInt(ASubTrait, ATrait):
    pass


struct SIMD[T: ATrait]:
    pass


trait TraitWithAliasReturnMethod:
    alias T: ATrait

    fn bork(self) -> SIMD[T]:
        ...


# CHECK-LABEL: lit.struct.decl @ExplicitStructWithAliasMethod
@fieldwise_init
struct ExplicitStructWithAliasMethod(TraitWithAliasReturnMethod):
    alias T: ASubTrait = ZInt

    # If we didn't follow SAVMBCTATBS, then verifyConformance would be
    # incorrectly checking for the existence of
    # `fn bork(self) -> SIMD[:ASubTrait ZInt]:` which is actually malformed
    # because SIMD takes an ATrait, not a ASubTrait.
    fn bork(self) -> SIMD[ZInt]:
        ...


# // -----

# Tests that we correctly handle substituting struct alias into the "needle"
# signature when confirming that a trait's STATIC method exists in the struct
# (see SAVMBCTATBS). Same as the previous test but with static.


trait ATrait:
    pass


trait ASubTrait(ATrait):
    pass


@register_passable("trivial")
struct ZInt(ASubTrait, ATrait):
    pass


struct SIMD[T: ATrait]:
    pass


trait TraitWithAliasReturnMethod:
    alias T: ATrait

    @staticmethod
    fn bork() -> SIMD[T]:
        ...


# CHECK-LABEL: lit.struct.decl @ExplicitStructWithAliasMethod
@fieldwise_init
struct ExplicitStructWithAliasMethod(TraitWithAliasReturnMethod):
    alias T: ASubTrait = ZInt

    # If we didn't follow SAVMBCTATBS, then verifyConformance would be
    # incorrectly checking for the existence of
    # `fn bork(self) -> SIMD[:ASubTrait ZInt]:` which is actually malformed
    # because SIMD takes an ATrait, not a ASubTrait.
    @staticmethod
    fn bork() -> SIMD[ZInt]:
        ...


# // -----

# Tests that we can call a static method that has an associated alias in it.


struct ZInt:
    pass


@fieldwise_init
struct Zcalar[X: ZInt]:
    pass


trait FooTrait:
    alias dtype: ZInt

    @staticmethod
    fn foo(x: Zcalar[dtype]):
        ...


fn bar[foo: FooTrait]():
    p0 = Zcalar[foo.dtype]()
    foo.foo(p0)


# // -----

# Sub-Trait Alias Type Can Be More Specific (STATCBMS):
# This test shows that a sub-trait's alias can have a more specific type than
# the super-trait's alias.
# See original bug:
# https://linear.app/modularml/issue/MOCO-1869/bug-trait-refinement-does-not-correctly-propagate-associated-type


# This is the important check, it makes sure that we:
#  * Pull it out of the vtable as a BB
#  * Upcast it to an AA


trait AA:
    fn __init__(out self):
        ...


trait BB(AA):
    fn __init__(out self):
        ...


trait A:
    alias Type: AA


trait B(A):
    alias Type: BB


fn fa[T: A]() -> T.Type:
    return T.Type()


# CHECK-LABEL: lit.fn @"fb
fn fb[T: B]() -> T.Type:
    # CHECK: %[[REBIND:.*]] = kgen.rebind {{.*}} : !lit.ref<:!BB #kgen.get_witness<#kgen.param.decl.ref<"T"> : !B, "associated_aliases::B", "Type">
    # CHECK-SAME: to !lit.ref<:!AA #kgen.get_witness<#kgen.type<!kgen.param<:!B T>> : !A, "associated_aliases::A", "Type">
    # CHECK: lit.call{{.*}}fa{{.*}}(%[[REBIND]])
    return fa[T]()


# // -----

# Tests that a trait or struct can declare an alias that is a more specific type
# of *both* of its parent traits' aliases' types.
# This is like the above STATCBMS test but with two parent traits.


trait AA:
    fn __init__(out self):
        ...


trait BB:
    fn __init__(out self):
        ...


trait CC(AA, BB):
    fn __init__(out self):
        ...


trait A:
    alias Type: AA


trait B:
    alias Type: BB


trait TraitWithExplicitOverride(A, B):
    alias Type: CC


fn receiveTraitWithExplicitOverride[T: TraitWithExplicitOverride]():
    alias cc: CC = T.Type


struct StructWithExplicitOverride(A, B):
    alias Type: CC = CC()


fn receiveStructWithExplicitOverride[T: StructWithExplicitOverride]():
    alias cc: CC = T.Type


# TODO(MOCO-2123): Make this work:
# trait TraitWithNoOverride(A, B):
#     pass
# fn receiveTraitWithNoOverride[T: TraitWithNoOverride]():
#     alias Something = T.Type


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
