# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated -split-input-file %s | FileCheck %s

# CHECK-DAG: #[[ExplicitStructWithAliasMethod_ATrait_VTable:.*]] = #kgen.type<!ExplicitStructWithAliasMethod, {"T" : !ATrait = #[[Int_ATrait_VTable:[a-zA-Z_]+]]
# CHECK-DAG: #[[Int_ATrait_VTable]] = #kgen.type<index,
# CHECK-DAG: #[[ImplicitStructWithAliasMethod_ATrait_VTable:.*]] = #kgen.type<!ImplicitStructWithAliasMethod, {"T" : !ATrait = #[[Int_ATrait_VTable]]
# CHECK-DAG: #[[StructWithMatchingAlias_ATrait_VTable:.*]] = #kgen.type<!StructWithMatchingAlias, {"N" : !Int = {42}, {{.*}} : !TraitWithAlias
# CHECK-DAG: #[[GenericStructWithAliasMethod_ATrait_VTable:.*]] = #kgen.type<@"associated-aliases"::@GenericStructWithAliasMethod<:!ATrait #[[Int_ATrait_VTable]]>, {"T" : !ATrait = #[[Int_ATrait_VTable]], "bork" : !lit.signature<[2]("self": {{.*}}, "__result__": !lit.ref<@"associated-aliases"::@SIMD<:!ATrait #[[Int_ATrait_VTable]]>, mut *[0,1]> byref_result


alias int = __mlir_type.index


# CHECK-LABEL: lit.trait.decl @TraitWithAlias
trait TraitWithAlias:
    # CHECK-NEXT: lit.alias.decl *"N`1": !Int
    alias N: Int


struct StructWithMatchingAlias(TraitWithAlias):
    alias N: Int = 42

    fn __init__(inout self):
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
    # CHECK: {{.*}} = lit.call @"associated-aliases"::@"getNFromTraitWithAlias{{.*}}<:!TraitWithAlias #[[StructWithMatchingAlias_ATrait_VTable]]>(%1)
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
    # CHECK:       {{.*}}lit.call @"associated-aliases"::@"receiveTraitWithAliasMethod{{.*}}<:!TraitWithAliasMethod #[[ExplicitStructWithAliasMethod_ATrait_VTable]]>
    receiveTraitWithAliasMethod(ExplicitStructWithAliasMethod())


# CHECK-LABEL: lit.struct.decl @ImplicitStructWithAliasMethod
@value
struct ImplicitStructWithAliasMethod:
    alias T: ATrait = int
    fn bork(self) -> SIMD[int]:
        return SIMD[int]()


# CHECK-LABEL: lit.func @"testUpcastingImplicitStructWithAliasMethod
fn testUpcastingImplicitStructWithAliasMethod():
    # CHECK: {{.*}}lit.call @"associated-aliases"::@"receiveTraitWithAliasMethod{{.*}}<:!TraitWithAliasMethod #[[ImplicitStructWithAliasMethod_ATrait_VTable]]>
    receiveTraitWithAliasMethod(ImplicitStructWithAliasMethod())


# CHECK-LABEL: lit.func @"callTraitWithAliasMethod
fn callTraitWithAliasMethod[X: TraitWithAliasMethod](t: X):
    # CHECK: {{.*}}lit.call
    # CHECK-SAME: "__result__": !lit.ref<@"associated-aliases"::@SIMD<:!ATrait get_type_method(:!TraitWithAliasMethod X, "T")>
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
    # CHECK: {{.*}}lit.call @"associated-aliases"::@"receiveTraitWithAliasMethod{{.*}}<:!TraitWithAliasMethod #[[GenericStructWithAliasMethod_ATrait_VTable]]>
    receiveTraitWithAliasMethod(GenericStructWithAliasMethod[int]())

# TODO(MOCO-1143): Make arguments work, like this:
# trait DType:
#     pass
# @value
# struct SIMD[T: DType]:
#     pass
# trait TraitWithAliasMethod:
#     alias T: DType
#     fn bork(self, thing: SIMD[T]) -> SIMD[T]:
#         ...
# @value
# struct StructWithAliasMethod(TraitWithAliasMethod):
#     alias T: DType = int
#     fn bork(self, thing: SIMD[int]) -> SIMD[int]:
#         return SIMD[int]()

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
