# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


struct OwnedKwargsDict[V: ImplicitlyCopyable]:
    def __init__(out self):
        pass

    def _insert(mut self, var key: String, var value: Self.V):
        pass

    def _insert(mut self, key: StringLiteral, var value: Self.V):
        pass
