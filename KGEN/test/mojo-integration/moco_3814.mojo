# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Verify that importing a package whose struct
# has a method returning another struct from the same package succeeds.
#
# When Container is body-resolved, its nested FnOps (such as Container.get_inner)
# are placed in the IR as fully materialized ops even though they are never
# resolved. finalizeImportedBytecodeModules() erases these materialized-but-unparsed
# ops so that dangling symbol references in their type attrs do not cause
# verifySymbolUses to fail.

# RUN: mkdir -p %t.moco-3814
# RUN: mojo package %S/inputs/moco_3814_package -o %t.moco-3814/moco_3814_package.mojopkg
# RUN: kgen-translate --mojo-enable-prebuilt-packages -import-mojo -I %t.moco-3814 %s | FileCheck %s

from moco_3814_package import Container

# Calling Container.__init__ body-resolves Container, which materializes
# Container.get_inner's FnOp even though get_inner is never called here.
# Inner appears only in get_inner's return type; finalizeImportedBytecodeModules()
# must erase get_inner for verification to pass.


def test() -> Int:
    var c = Container(42)
    # CHECK: lit.call @moco_3814_package::@container::@Container::@"__init__
    return c.get_count()
    # CHECK: lit.call @moco_3814_package::@container::@Container::@"get_count


# Verify the imported package module is present in the output.
# CHECK: lit.package @moco_3814_package
# CHECK: lit.struct.decl @Container
# CHECK-NOT: get_inner
