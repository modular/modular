# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Submodule whose symbols are (or are not) re-exported by the package."""


# Re-exported by __init__ -> suggestable as `std.import_suggestion`.
def explicit_reexport():
    pass


# Re-exported by __init__ but private -> never suggested.
def _private_reexport():
    pass


# Defined here but NOT re-exported -> not public API, so not suggested even
# though it is reachable via the full submodule path.
def not_reexported_symbol():
    pass
