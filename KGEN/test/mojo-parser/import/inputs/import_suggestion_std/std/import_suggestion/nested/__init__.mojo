# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Nested sub-package: verifies the suggestion uses the deep package path."""

# An explicit re-export inside a nested sub-package's __init__; suggestable as
# the nested package path `std.import_suggestion.nested`.
from .detail import nested_reexport


# A direct declaration in a nested sub-package's __init__.
def nested_symbol():
    pass
