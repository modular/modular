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


# CHECK: lit.struct.decl @MyRegType({{.*}}PackageTrait
@register_passable
struct MyRegType(PackageTrait):
    fn method(self):
        pass


fn bind_trait[T: PackageTrait]():
    pass


# CHECK-LABEL: lit.fn @"test
fn test():
    # CHECK-NEXT: [!MyType{{[0-9]*}}, {"method" {{.*}}@MyType::@"method
    bind_trait[MyType]()
    # CHECK-NEXT: [!MyRegType{{[0-9]*}}, {"method" {{.*}}!lit.generator<[1]("self": !lit.ref<!MyRegType, imm *[0,0]> read_mem) -> !kgen.none>{{.*}}@MyRegType::@"method
    bind_trait[MyRegType]()
    # CHECK-NEXT: [!UseTrait{{[0-9]*}}, {"method" {{.*}}@UseTrait::@"method
    trait_method[UseTrait]()
    # CHECK-NEXT: [!UseTraitReg{{[0-9]*}}, {"method" {{.*}}!lit.generator<[1]("self": !lit.ref<!UseTraitReg, imm *[0,0]> read_mem) -> !kgen.none>{{.*}}@UseTraitReg::@"method
    trait_method[UseTraitReg]()

    # COM: Anchor this decl reference to materialize it.
    contains_thunk_ref()


fn use_trait[T: PackageTrait](x: UseTrait, y: T):
    y.method()


# CHECK: lit.trait.decl @PackageTrait
# CHECK: lit.trait.decl @UsedInPackageTrait
# CHECK: lit.struct.decl @UseTrait({{.*}}UsedInPackageTrait
