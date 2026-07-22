# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Sub-package wildcard-re-exported by the parent (`from .wild_pkg import *`).
Exercises the sub-package branch of the wildcard resolver (its public surface is
its own __init__'s decls)."""


# Pulled into std.import_suggestion's surface via the sub-package wildcard.
def wildpkg_symbol():
    pass
