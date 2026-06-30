# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Re-exports `foo` from `source`. Importing `foo` from here resolves (through
# this gate) to the same `source.foo` decl as importing it directly.

from .source import foo
