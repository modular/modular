# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s

# CHECK: [[VTABLE:.*]] = #kgen.type<!StructWithMatchingAlias, {"N" : !Int = {42}, {{.*}} : !MyTrait

alias int = __mlir_type.index


# CHECK-LABEL: lit.trait.decl @MyTrait
trait MyTrait:
    # CHECK-NEXT: lit.alias.decl *"N`1": !Int
    alias N: Int


struct StructWithMatchingAlias(MyTrait):
    alias N: Int = 42

    fn __init__(inout self):
        pass


# CHECK-LABEL: getNFromMyTrait
fn getNFromMyTrait[T: MyTrait](t: T) -> Int:
    # CHECK-NEXT: lit.alias.decl [[X:.*]]: !Int = <get_type_method(:!MyTrait T, "N")>
    # CHECK-NEXT: {{.*}} = kgen.param.constant: !Int = <[[X]]>
    alias X = T.N
    return X


# CHECK-LABEL: testMyTraitAndStructWithMatchingAlias
@export
fn testMyTraitAndStructWithMatchingAlias():
    # CHECK: {{.*}} = lit.call @"associated-aliases"::@"getNFromMyTrait{{.*}}<:!MyTrait [[VTABLE]]>(%1)
    _ = getNFromMyTrait(StructWithMatchingAlias())


struct StructWithParam[X: Int]:
    pass


# TODO(MOCO-1143): Make this work:
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
