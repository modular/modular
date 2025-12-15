# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


struct OwnedKwargsDict[V: ImplicitlyCopyable]:
    fn __init__(out self):
        pass

    fn _insert(mut self, var key: String, var value: Self.V):
        pass

    fn _insert(mut self, key: StringLiteral, var value: Self.V):
        pass
