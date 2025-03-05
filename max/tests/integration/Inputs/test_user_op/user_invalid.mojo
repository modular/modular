# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import compiler
from tensor import ManagedTensorSlice, OutputTensor


@compiler.register("fails_to_elaborate")
struct FailsToElaborate:
    @staticmethod
    fn execute(
        out: OutputTensor[type = DType.int32, rank=1],
    ):
        constrained[False, "oops"]()
