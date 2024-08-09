# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


struct String(KeyElement):
    fn __init__(inout self):
        pass

    fn __init__(inout self, literal: StringLiteral):
        pass

    fn __copyinit__(inout self, existing: Self):
        pass

    fn __moveinit__(inout self, owned existing: String):
        pass

    fn __len__(self) -> Int:
        return 0
