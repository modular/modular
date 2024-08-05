# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


trait _StaticTensorType:
    @staticmethod
    fn _get_dtype() -> DType:
        ...

    @staticmethod
    fn _get_static_rank() -> Int:
        ...


struct UnsafeTensorSlice[type: DType, rank: Int](_StaticTensorType):
    @staticmethod
    fn _get_dtype() -> DType:
        return Self.type

    @staticmethod
    fn _get_static_rank() -> Int:
        return Self.rank
