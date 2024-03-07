# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


trait CollectionElement:
    pass


trait KeyElement:
    pass


struct Dict[K: KeyElement, V: CollectionElement]:
    fn __init__(inout self):
        pass

    fn __setitem__(inout self, key: K, value: V):
        pass
