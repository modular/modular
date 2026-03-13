# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -mojo-search-paths=%S -verify-diagnostics %s

# Test that @stable members cannot be declared in unstable structs/traits.
# The actual warnings are defined in test_std_mock/__init__.mojo, which is an
# opted-in package. This test triggers parsing of that package.
#
# In an opted-in package, types without @stable are considered unstable.
# Declaring a @stable member inside an unstable type emits a warning because
# users cannot rely on the member's stability if the containing type isn't
# stable.

from test_std_mock import *


# To trigger the trait error, we need to implement the unstable trait.
# This forces resolution of the trait's method signatures.
struct ImplementsUnstableTrait(UnstableTraitWithStableMember):
    def __init__(out self):
        pass

    def stable_method_in_unstable_trait(self):
        pass


def main():
    # Force use of the types and methods to ensure they're fully resolved.
    var x = UnstableStructWithStableMember()
    x.stable_method_in_unstable()

    var y = ImplementsUnstableTrait()
    y.stable_method_in_unstable_trait()

    # Force resolution of struct with unstable alias implementing stable trait alias.
    var z = StableStructWithUnstableAliasImpl()
    _ = z
