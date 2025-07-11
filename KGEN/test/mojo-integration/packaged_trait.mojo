# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo package %S/inputs/test_package -o %T/test_package_trait.mojopkg
# RUN: kgen-translate --mojo-enable-prebuilt-packages -import-mojo -I %T %s --kgen-print-inline-type-values | FileCheck %s

from test_package_trait.module import (
    PackageTrait,
    UseTrait,
    UseTraitReg,
    trait_method,
    contains_thunk_ref,
)


# CHECK: lit.struct.decl @MyType({{.*}}PackageTrait
struct MyType(PackageTrait):
    fn method(self):
        pass

    # CHECK: kgen.conformance {{.*}}::PackageTrait
    # CHECK: kgen.witness "method" {{.*}} = {{.*}}::@MyType::@"method


# CHECK: lit.struct.decl @MyRegType({{.*}}PackageTrait
@register_passable
struct MyRegType(PackageTrait):
    fn method(self):
        pass

    # CHECK: kgen.conformance {{.*}}::PackageTrait
    # CHECK: kgen.witness "method" : !lit.generator<[1]("self": !lit.ref<!MyRegType, imm *[0,0]> read_mem) -> !kgen.none> = {{.*}}::@MyRegType::@"method


fn bind_trait[T: PackageTrait]():
    pass


# CHECK-LABEL: lit.fn @"test
fn test():
    # CHECK-NEXT: <:!PackageTrait !MyType>
    bind_trait[MyType]()
    # CHECK-NEXT: <:!PackageTrait !MyRegType>
    bind_trait[MyRegType]()
    # CHECK-NEXT: <:!UsedInPackageTrait !UseTrait>
    trait_method[UseTrait]()
    # CHECK-NEXT: <:!UsedInPackageTrait !UseTraitReg>
    trait_method[UseTraitReg]()

    # COM: Anchor this decl reference to materialize it.
    contains_thunk_ref()


fn use_trait[T: PackageTrait](x: UseTrait, y: T):
    y.method()


# CHECK-LABEL: lit.package @test_package_trait

# CHECK: lit.trait.decl @PackageTrait
# CHECK: lit.trait.decl @UsedInPackageTrait

# CHECK-LABEL: lit.struct.decl @UseTrait
# CHECK: kgen.conformance {{.*}}::UsedInPackageTrait
# CHECK: kgen.witness "method" {{.*}} = {{.*}}::@UseTrait::@"method

# CHECK-LABEL: lit.struct.decl @UseTraitReg
# CHECK: kgen.conformance {{.*}}::UsedInPackageTrait
# CHECK: kgen.witness "method" : !lit.generator<[1]("self": !lit.ref<!UseTraitReg, imm *[0,0]> read_mem) -> !kgen.none> = {{.*}}::@UseTraitReg::@"method
