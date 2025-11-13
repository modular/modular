# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from sys.ffi import RTLD

comptime IntAlias = 12
"""Int alias summary

Int alias description."""
comptime ExplicitIntAlias: Int = 123


fn function() -> Int:
    comptime AliasInsideFunction = "sdfsdf"


comptime AliasToAlias = IntAlias


struct StructWithAlias:
    comptime AliasInStruct = Int


comptime AliasInStructRef = StructWithAlias.AliasInStruct

comptime ExternalAlias = RTLD.LAZY
