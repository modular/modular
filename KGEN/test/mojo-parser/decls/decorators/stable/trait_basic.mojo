# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s

# Test that @stable decorator is recognized on traits and trait methods.


# CHECK: lit.trait.decl @StableTrait{{.*}}hasStableDecorator
@stable
trait StableTrait:
    pass


# UnstableTrait should not have hasStableDecorator
# CHECK: lit.trait.decl @UnstableTrait
# CHECK-NOT: hasStableDecorator
# CHECK-SAME: unspecified
trait UnstableTrait:
    pass


# The trait must be @stable to allow @stable members inside it.
# CHECK: lit.trait.decl @TraitWithStableMethod{{.*}}hasStableDecorator
@stable
trait TraitWithStableMethod:
    # CHECK: lit.fn @"stable_trait_method({{.*}}hasStableDecorator
    @stable
    def stable_trait_method(self):
        ...

    # CHECK: lit.fn @"unstable_trait_method(
    # CHECK-NOT: hasStableDecorator
    # CHECK-SAME: sourceName = "unstable_trait_method"
    def unstable_trait_method(self):
        ...
