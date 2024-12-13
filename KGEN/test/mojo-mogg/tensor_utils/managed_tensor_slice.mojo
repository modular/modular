# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


# Stub used internally for testing only
@register_passable("trivial")
struct ManagedTensorSlice[type: DType, rank: Int]:
    fn __init__(mut self):
        pass

    @staticmethod
    fn _get_dtype() -> DType:
        return Self.type

    @staticmethod
    fn _get_static_rank() -> Int:
        return Self.rank
