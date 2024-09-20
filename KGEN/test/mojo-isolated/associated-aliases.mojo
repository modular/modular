# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s

# CHECK: [[VTABLE:.*]] = #kgen.type<!StructWithMatchingAlias, {"N" : !Int = {42}, {{.*}} : !TraitWithAlias

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
    # CHECK: {{.*}} = lit.call @"associated-aliases"::@"getNFromTraitWithAlias{{.*}}<:!TraitWithAlias [[VTABLE]]>(%1)
    _ = getNFromTraitWithAlias(StructWithMatchingAlias())



# // -----

# TODO(MOCO-1143): Uncomment these in the upcasting PR:
# # HECK-DAG: [[VTABLE2:.*]] = #kgen.type<!StructWithAliasMethod
# # HECK-DAG: "T" : !ATrait = #[[VTABLE3:[a-zA-Z_]+]]
# # HECK-DAG: #[[VTABLE3]] = #kgen.type<index,


trait ATrait:
    pass


@value
struct SIMD[T: ATrait]:
    pass


trait TraitWithAliasMethod:
    alias T: ATrait
    fn bork(self, thing: SIMD[T]) -> SIMD[T]:
        ...


@value
struct StructWithAliasMethod(TraitWithAliasMethod):
    alias T: ATrait = int
    fn bork(self, thing: SIMD[int]) -> SIMD[int]:
        return SIMD[int]()


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
