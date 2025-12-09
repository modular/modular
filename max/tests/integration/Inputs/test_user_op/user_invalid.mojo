# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import compiler
from tensor import OutputTensor


@compiler.register("fails_to_elaborate")
struct FailsToElaborate:
    @staticmethod
    fn execute(
        output: OutputTensor[dtype = DType.int32, rank=1],
    ):
        constrained[False, "oops"]()
