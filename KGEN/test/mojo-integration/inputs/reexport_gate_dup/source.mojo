# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# The single real definition of `foo`. It is reached from `__init__` along two
# different import paths (directly, and via `relay`), so both paths must collapse
# to this one decl.


def foo() -> Int:
    return 42
