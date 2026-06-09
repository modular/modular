# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Defines a function with a DIFFERENT name than the module.
# __init__.mojo does NOT re-export anything named "qux", so importing
# "qux" from the package should still resolve to the submodule.


def qux_func() -> Int:
    return 7
