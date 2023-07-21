# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from String import String

alias IntAlias = 12
"""Int alias summary

Int alias description."""

alias ExplicitIntAlias: Int = 123


fn function() -> Int:
    alias AliasInsideFunction = "sdfsdf"


alias AliasToAlias = IntAlias
