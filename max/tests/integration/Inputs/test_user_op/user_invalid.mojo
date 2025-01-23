# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import compiler
from tensor_utils import ManagedTensorSlice


@compiler.register("fails_to_elaborate")
struct FailsToElaborate:
    @uses_opaque
    @staticmethod
    fn execute(
        out: ManagedTensorSlice[DType.int32, 1],
    ):
        constrained[False, "oops"]()
