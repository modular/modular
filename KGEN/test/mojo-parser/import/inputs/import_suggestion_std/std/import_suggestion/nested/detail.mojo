# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Submodule whose symbol the nested package re-exports explicitly."""


# Explicitly re-exported by nested/__init__.mojo -> suggestable as
# `std.import_suggestion.nested`.
def nested_reexport():
    pass
