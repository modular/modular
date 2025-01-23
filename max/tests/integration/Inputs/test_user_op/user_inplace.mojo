# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import compiler
from tensor_utils import ManagedTensorSlice


@compiler.register("mutable_test_op", num_dps_outputs=0)
struct MutableTestOp:
    @staticmethod
    fn execute(in_place_tensor: ManagedTensorSlice) raises:
        x = in_place_tensor._ptr.load(0)
        x += 1
        in_place_tensor._ptr.store(0, x)
