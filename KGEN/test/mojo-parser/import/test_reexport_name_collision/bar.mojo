# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Defines a non-parametric function with the same name as the module.
# Used to test that re-exporting works for plain functions too,
# not just parametric ones.

fn bar() -> Int:
    return 99
