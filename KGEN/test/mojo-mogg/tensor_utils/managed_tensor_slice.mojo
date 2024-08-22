# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


struct ManagedTensorSlice[type: DType, rank: Int]:
    @staticmethod
    fn _get_dtype() -> DType:
        return Self.type

    @staticmethod
    fn _get_static_rank() -> Int:
        return Self.rank
