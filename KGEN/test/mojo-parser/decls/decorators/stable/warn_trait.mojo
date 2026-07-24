# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test that --warn-on-unstable-apis emits warnings for unstable trait
# implementation from opted-in packages.

# RUN: %parse-mojo-isolated -mojo-search-paths=%S -warn-on-unstable-apis %s 2>&1 | FileCheck %s

from test_std_mock import StableTrait, UnstableTrait


struct ImplementsStableTrait(StableTrait, Movable where False):
    """Implementing a stable trait should not trigger a warning."""

    pass


# CHECK: warning: use of unstable API 'UnstableTrait'
struct ImplementsUnstableTrait(UnstableTrait, Movable where False):
    """Implementing an unstable trait should trigger a warning."""

    pass
