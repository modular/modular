# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


struct SomeStruct:
    var a_field: Int
    """Summary of a_field."""

    fn __init__(inout self):
        pass


fn main():
    var someStruct = SomeStruct()
    _ = someStruct.a_field
