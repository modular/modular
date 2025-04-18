# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated -split-input-file %s | kgen-opt | FileCheck %s

# CHECK-DAG: #[[Int_VTable:.*]] = #kgen.type<index,{{.*}} : !ATrait
# CHECK-DAG: #[[ExplicitStructWithAliasMethod_VTable:.*]] = #kgen.type<!ExplicitStructWithAliasMethod, {"T" : !ATrait = #[[Int_VTable]]{{.*}} : !TraitWithAliasReturnMethod
# CHECK-DAG: #[[ImplicitStructWithAliasMethod_VTable:.*]] = #kgen.type<!ImplicitStructWithAliasMethod, {"T" : !ATrait = #[[Int_VTable]]{{.*}} : !TraitWithAliasReturnMethod
# CHECK-DAG: #[[GenericStructWithAliasMethod_VTable:.*]] = #kgen.type<@associated_aliases::@GenericStructWithAliasMethod<:!ATrait #[[Int_VTable]]>, {"T" : !ATrait = #[[Int_VTable]], "bork" : !lit.generator<[2]("self": {{.*}}, "__result__": !lit.ref<@associated_aliases::@SIMD<:!ATrait #[[Int_VTable]]>, mut *[0,1]> byref_result{{.*}} : !TraitWithAliasReturnMethod
# CHECK-DAG: #[[StructWithMatchingAlias_VTable:.*]] = #kgen.type<!StructWithMatchingAlias, {"N" : !Int = {42}, {{.*}} : !TraitWithAlias
# CHECK-DAG: #[[StructWithAliasArgMethod_VTable:.*]] = #kgen.type<!StructWithAliasArgMethod,{{.*}}"lork" : !lit.generator<{{.*}}"thing": !lit.ref<@associated_aliases::@SIMD<:!ATrait #type_value>{{.*}}> = @associated_aliases::@StructWithAliasArgMethod::@"lork({{.*}}SIMD[__mlir_type.index])",{{.*}}> : !TraitWithAliasArgMethod

alias Index = __mlir_type.index


# CHECK-LABEL: lit.trait.decl @TraitWithAlias
trait TraitWithAlias:
    # CHECK-NEXT: lit.alias.decl *"N`1": !Int
    alias N: Int


struct StructWithMatchingAlias(TraitWithAlias):
    alias N: Int = 42

    fn __init__(out self):
        pass

# CHECK-LABEL: getNFromTraitWithAlias
fn getNFromTraitWithAlias[T: TraitWithAlias](t: T) -> Int:
    # CHECK-NEXT: lit.alias.decl [[X:.*]]: !Int = <get_vtable_entry(:!TraitWithAlias T, "N")>
    # CHECK-NEXT: kgen.param.constant: !Int = <get_vtable_entry(:!TraitWithAlias T, "N")>
    alias X = T.N
    return X


# CHECK-LABEL: testTraitWithAliasAndStructWithMatchingAlias
@export
fn testTraitWithAliasAndStructWithMatchingAlias():
    # CHECK: {{.*}} = lit.call @associated_aliases::@"getNFromTraitWithAlias{{.*}}<:!TraitWithAlias #[[StructWithMatchingAlias_VTable]]>(%1)
    _ = getNFromTraitWithAlias(StructWithMatchingAlias())


trait TraitWithTypeAlias:
    alias T: TraitWithAlias


trait TraitWithSameTypeAlias(TraitWithTypeAlias):
    alias T: TraitWithAlias


# CHECK-LABEL: testTraitWithRefinedTypeAlias
fn testTraitWithRefinedTypeAlias[T: TraitWithSameTypeAlias]():
    # CHECK-NEXT: !TraitWithAlias = <get_vtable_entry(:!TraitWithSameTypeAlias T, "T")>
    alias MyT: TraitWithAlias = T.T


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
    alias T: ATrait = Index
    fn bork(self) -> SIMD[Index]:
        return SIMD[Index]()


fn receiveTraitWithAliasReturnMethod[X: TraitWithAliasReturnMethod](t: X):
    pass


# CHECK-LABEL: lit.fn @"testUpcastingExplicitStructWithAliasMethod
fn testUpcastingExplicitStructWithAliasMethod():
    # CHECK:       {{.*}}lit.call @associated_aliases::@"receiveTraitWithAliasReturnMethod{{.*}}<:!TraitWithAliasReturnMethod #[[ExplicitStructWithAliasMethod_VTable]]>
    receiveTraitWithAliasReturnMethod(ExplicitStructWithAliasMethod())


# CHECK-LABEL: lit.struct.decl @ImplicitStructWithAliasMethod
@value
struct ImplicitStructWithAliasMethod:
    alias T: ATrait = Index
    fn bork(self) -> SIMD[Index]:
        return SIMD[Index]()


# CHECK-LABEL: lit.fn @"testUpcastingImplicitStructWithAliasMethod
fn testUpcastingImplicitStructWithAliasMethod():
    # CHECK: {{.*}}lit.call @associated_aliases::@"receiveTraitWithAliasReturnMethod{{.*}}<:!TraitWithAliasReturnMethod #[[ImplicitStructWithAliasMethod_VTable]]>
    receiveTraitWithAliasReturnMethod(ImplicitStructWithAliasMethod())


# CHECK-LABEL: lit.fn @"callTraitWithAliasReturnMethod
fn callTraitWithAliasReturnMethod[X: TraitWithAliasReturnMethod](t: X):
    # CHECK: {{.*}}lit.call
    # CHECK-SAME: "__result__": !lit.ref<@associated_aliases::@SIMD<:!ATrait get_vtable_entry(:!TraitWithAliasReturnMethod X, "T")>
    # CHECK-SAME: : get_vtable_entry(:!TraitWithAliasReturnMethod X, "bork")
    _ = t.bork()


# TODO(MOCO-1109): also check that this works with the thunk generation for @register_passable methods
@value
struct GenericStructWithAliasMethod[Z: ATrait](TraitWithAliasReturnMethod):
    alias T: ATrait = Z
    fn bork(self) -> SIMD[Z]:
        return SIMD[Z]()

# CHECK-LABEL: lit.fn @"testUpcastingGenericStructWithAliasMethod
fn testUpcastingGenericStructWithAliasMethod():
    # CHECK: {{.*}}lit.call @associated_aliases::@"receiveTraitWithAliasReturnMethod{{.*}}<:!TraitWithAliasReturnMethod #[[GenericStructWithAliasMethod_VTable]]>
    receiveTraitWithAliasReturnMethod(GenericStructWithAliasMethod[Index]())


trait TraitWithAliasArgMethod:
    alias T: ATrait
    fn lork(self, thing: SIMD[T]):
        ...


@value
struct StructWithAliasArgMethod(TraitWithAliasArgMethod):
    alias T: ATrait = Index
    fn lork(self, thing: SIMD[Index]):
        pass

fn receiveTraitWithAliasArgMethod[X: TraitWithAliasArgMethod](t: X):
    pass


# CHECK-LABEL: lit.fn @"testUpcastingStructWithAliasArgMethod
fn testUpcastingStructWithAliasArgMethod():
    # CHECK: {{.*}}lit.call @associated_aliases::@"receiveTraitWithAliasArgMethod{{.*}}<:!TraitWithAliasArgMethod #[[StructWithAliasArgMethod_VTable]]
    receiveTraitWithAliasArgMethod(StructWithAliasArgMethod())


# CHECK-LABEL: lit.fn @"callTraitMethodWithAliasArg
fn callTraitMethodWithAliasArg[X: TraitWithAliasArgMethod](t: X, thing: SIMD[X.T]):
    # CHECK:  %0 = lit.call
    # CHECK-SAME: "thing": !lit.ref<@associated_aliases::@SIMD<:!ATrait get_vtable_entry(:!TraitWithAliasArgMethod X, "T")>
    # CHECK-SAME: : get_vtable_entry(:!TraitWithAliasArgMethod X, "lork")
    t.lork(thing)


# Tests explicit mentions of `Self.thing` where `thing` is an associated alias.
# See https://linear.app/modularml/issue/MOCO-1438
trait TraitWithSelfDotAliasReturnMethod:
    alias T: ATrait

    fn bork(self) -> SIMD[Self.T]:
        ...


struct StructWithSelfDotAliasReturnMethod(TraitWithSelfDotAliasReturnMethod):
    alias T: ATrait = Index

    fn bork(self) -> SIMD[Self.T]:
        return SIMD[Self.T]()

fn receiveTraitWithSelfDotAliasReturnMethod[T: TraitWithSelfDotAliasReturnMethod](z: T):
    _ = z.bork()

fn callTraitWithSelfDotAliasReturnMethod(x: StructWithSelfDotAliasReturnMethod):
    receiveTraitWithSelfDotAliasReturnMethod(x)


# TODO(MOCO-1109): also check that this works with the thunk generation for @register_passable methods
@value
struct GenericStructWithSelfDotAliasReturnMethod[Z: ATrait](TraitWithSelfDotAliasReturnMethod):
    alias T: ATrait = Z
    fn bork(self) -> SIMD[Self.Z]:
        return SIMD[Z]()

fn testUpcastingGenericStructWithSelfDotAliasReturnMethod():
    receiveTraitWithSelfDotAliasReturnMethod(GenericStructWithSelfDotAliasReturnMethod[Index]())


# TODO(MOCO-1259): Support static methods with associated aliases

# TODO(MOCO-1143): Make this work:
# struct StructWithParam[X: Int]:
#     pass
# # HECK-LABEL: lit.trait.decl @Spork<Self: type> {
# trait TraitWithStaticMethodUsingAlias:
#     # HECK-NEXT: lit.alias.decl N = <?>
#     alias N: Int
#     # HECK-LABEL: lit.fn @foo(%x: !pop.simd<N, f32>) { // #kgen.param.decl.ref<"N"> : index
#     @staticmethod
#     fn foo(x: StructWithParam[N]):
#         pass
# struct StructWithStaticMethod:
#     @staticmethod
#     fn foo(x: StructWithParam[5]):
#         pass
# fn sporkify[T: TraitWithStaticMethodUsingAlias]() -> Int:
#    return T.N # emits a get_vtable_value
# @export
# fn testSomething():
#     # And maybe add a test for sporkify[TraitWithStaticMethodUsingAlias]()
#     sporkify[StructWithStaticMethod]()
