# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import compiler
from tensor import ManagedTensorSlice


@compiler.register("fails_to_elaborate")
struct FailsToElaborate:
    @staticmethod
    fn execute(
        out: ManagedTensorSlice[type = DType.int32, rank=1],
    ):
        constrained[False, "oops"]()
