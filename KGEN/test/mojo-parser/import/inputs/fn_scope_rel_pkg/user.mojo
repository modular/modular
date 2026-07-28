# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# A module whose function body uses a RELATIVE import (`from .util import`).
# Imported by import_dotted_in_function.mojo to verify relative imports work
# at function scope.


def rel_fn() -> Int:
    from .util import util_fn

    return util_fn()
