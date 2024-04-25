# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from sys.ffi import RTLD

alias IntAlias = 12
"""Int alias summary

Int alias description."""
alias ExplicitIntAlias: Int = 123


fn function() -> Int:
    alias AliasInsideFunction = "sdfsdf"


alias AliasToAlias = IntAlias


struct StructWithAlias:
    alias AliasInStruct = Int


alias AliasInStructRef = StructWithAlias.AliasInStruct

alias ExternalAlias = RTLD.LAZY
