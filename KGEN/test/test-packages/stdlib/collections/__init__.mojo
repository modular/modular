# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


struct OwnedKwargsDict[V: Copyable & Movable]:
    fn __init__(out self):
        pass

    fn _insert(mut self, var key: String, var value: V):
        pass

    fn _insert(mut self, key: StringLiteral, var value: V):
        pass
