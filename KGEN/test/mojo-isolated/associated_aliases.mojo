# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated -split-input-file %s | kgen-opt | FileCheck %s

# CHECK-DAG: #[[Int_VTable:.*]] = #kgen.type<index,{{.*}} : !ATrait
# CHECK-DAG: #[[ExplicitStructWithAliasMethod_VTable:.*]] = #kgen.type<!ExplicitStructWithAliasMethod, {"T" : !ATrait = #[[Int_VTable]]{{.*}} : !TraitWithAliasMethod
# CHECK-DAG: #[[ImplicitStructWithAliasMethod_VTable:.*]] = #kgen.type<!ImplicitStructWithAliasMethod, {"T" : !ATrait = #[[Int_VTable]]{{.*}} : !TraitWithAliasMethod
# CHECK-DAG: #[[GenericStructWithAliasMethod_VTable:.*]] = #kgen.type<@associated_aliases::@GenericStructWithAliasMethod<:!ATrait #[[Int_VTable]]>, {"T" : !ATrait = #[[Int_VTable]], "bork" : !lit.signature<[2]("self": {{.*}}, "__result__": !lit.ref<@associated_aliases::@SIMD<:!ATrait #[[Int_VTable]]>, mut *[0,1]> byref_result{{.*}} : !TraitWithAliasMethod
# CHECK-DAG: #[[StructWithMatchingAlias_VTable:.*]] = #kgen.type<!StructWithMatchingAlias, {"N" : !Int = {42}, {{.*}} : !TraitWithAlias
# CHECK-DAG: #[[StructWithAliasArgMethod_VTable:.*]] = #kgen.type<!StructWithAliasArgMethod,{{.*}}"lork" : !lit.signature<{{.*}}"thing": !lit.ref<@associated_aliases::@SIMD<:!ATrait #type_value>{{.*}}> = @associated_aliases::@StructWithAliasArgMethod::@"lork({{.*}}SIMD[__mlir_type.index])",{{.*}}> : !TraitWithAliasArgMethod

alias int = __mlir_type.index


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
    # CHECK-NEXT: lit.alias.decl [[X:.*]]: !Int = <get_type_method(:!TraitWithAlias T, "N")>
    # CHECK-NEXT: {{.*}} = kgen.param.constant: !Int = <[[X]]>
    alias X = T.N
    return X


# CHECK-LABEL: testTraitWithAliasAndStructWithMatchingAlias
@export
fn testTraitWithAliasAndStructWithMatchingAlias():
    # CHECK: {{.*}} = lit.call @associated_aliases::@"getNFromTraitWithAlias{{.*}}<:!TraitWithAlias #[[StructWithMatchingAlias_VTable]]>(%1)
    _ = getNFromTraitWithAlias(StructWithMatchingAlias())


trait ATrait:
    pass


@value
struct SIMD[T: ATrait]:
    pass


trait TraitWithAliasMethod:
    alias T: ATrait
    fn bork(self) -> SIMD[T]:
        ...


# CHECK-LABEL: lit.struct.decl @ExplicitStructWithAliasMethod
@value
struct ExplicitStructWithAliasMethod(TraitWithAliasMethod):
    alias T: ATrait = int
    fn bork(self) -> SIMD[int]:
        return SIMD[int]()


fn receiveTraitWithAliasMethod[X: TraitWithAliasMethod](t: X):
    pass


# CHECK-LABEL: lit.func @"testUpcastingExplicitStructWithAliasMethod
fn testUpcastingExplicitStructWithAliasMethod():
    # CHECK:       {{.*}}lit.call @associated_aliases::@"receiveTraitWithAliasMethod{{.*}}<:!TraitWithAliasMethod #[[ExplicitStructWithAliasMethod_VTable]]>
    receiveTraitWithAliasMethod(ExplicitStructWithAliasMethod())


# CHECK-LABEL: lit.struct.decl @ImplicitStructWithAliasMethod
@value
struct ImplicitStructWithAliasMethod:
    alias T: ATrait = int
    fn bork(self) -> SIMD[int]:
        return SIMD[int]()


# CHECK-LABEL: lit.func @"testUpcastingImplicitStructWithAliasMethod
fn testUpcastingImplicitStructWithAliasMethod():
    # CHECK: {{.*}}lit.call @associated_aliases::@"receiveTraitWithAliasMethod{{.*}}<:!TraitWithAliasMethod #[[ImplicitStructWithAliasMethod_VTable]]>
    receiveTraitWithAliasMethod(ImplicitStructWithAliasMethod())


# CHECK-LABEL: lit.func @"callTraitWithAliasMethod
fn callTraitWithAliasMethod[X: TraitWithAliasMethod](t: X):
    # CHECK: {{.*}}lit.call
    # CHECK-SAME: "__result__": !lit.ref<@associated_aliases::@SIMD<:!ATrait get_type_method(:!TraitWithAliasMethod X, "T")>
    # CHECK-SAME: : get_type_method(:!TraitWithAliasMethod X, "bork")
    _ = t.bork()


# TODO(MOCO-1109): also check that this works with the thunk generation for @register_passable methods
@value
struct GenericStructWithAliasMethod[Z: ATrait](TraitWithAliasMethod):
    alias T: ATrait = Z
    fn bork(self) -> SIMD[Z]:
        return SIMD[Z]()

# CHECK-LABEL: lit.func @"testUpcastingGenericStructWithAliasMethod
fn testUpcastingGenericStructWithAliasMethod():
    # CHECK: {{.*}}lit.call @associated_aliases::@"receiveTraitWithAliasMethod{{.*}}<:!TraitWithAliasMethod #[[GenericStructWithAliasMethod_VTable]]>
    receiveTraitWithAliasMethod(GenericStructWithAliasMethod[int]())


trait TraitWithAliasArgMethod:
    alias T: ATrait
    fn lork(self, thing: SIMD[T]):
        ...


@value
struct StructWithAliasArgMethod(TraitWithAliasArgMethod):
    alias T: ATrait = int
    fn lork(self, thing: SIMD[int]):
        pass

fn receiveTraitWithAliasArgMethod[X: TraitWithAliasArgMethod](t: X):
    pass


# CHECK-LABEL: lit.func @"testUpcastingStructWithAliasArgMethod
fn testUpcastingStructWithAliasArgMethod():
    # CHECK: {{.*}}lit.call @associated_aliases::@"receiveTraitWithAliasArgMethod{{.*}}<:!TraitWithAliasArgMethod #[[StructWithAliasArgMethod_VTable]]
    receiveTraitWithAliasArgMethod(StructWithAliasArgMethod())


# CHECK-LABEL: lit.func @"callTraitMethodWithAliasArg
fn callTraitMethodWithAliasArg[X: TraitWithAliasArgMethod](t: X, thing: SIMD[X.T]):
    # CHECK:  %0 = lit.call
    # CHECK-SAME: "thing": !lit.ref<@associated_aliases::@SIMD<:!ATrait get_type_method(:!TraitWithAliasArgMethod X, "T")>
    # CHECK-SAME: : get_type_method(:!TraitWithAliasArgMethod X, "lork")
    t.lork(thing)


# TODO(MOCO-1259): Support static methods with associated aliases

# TODO(MOCO-1143): Make this work:
# struct StructWithParam[X: Int]:
#     pass
# # HECK-LABEL: lit.trait.decl @Spork<Self: type> {
# trait TraitWithStaticMethodUsingAlias:
#     # HECK-NEXT: lit.alias.decl N = <?>
#     alias N: Int
#     # HECK-LABEL: lit.func @foo(%x: !pop.simd<N, f32>) { // #kgen.param.decl.ref<"N"> : index
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
