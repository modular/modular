# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from tensor_internal import TensorSpec
from memory import UnsafePointer


trait TensorLike:
    fn spec(self) -> TensorSpec:
        ...

    fn unsafe_ptr[type: DType](self) -> UnsafePointer[Scalar[type]]:
        pass
