# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


struct OwnedKwargsDict[V: CollectionElement]:
    fn __init__(out self):
        pass

    fn _insert(mut self, owned key: String, owned value: V):
        pass

    fn _insert(mut self, key: StringLiteral, owned value: V):
        pass
