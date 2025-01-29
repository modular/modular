# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import compiler
from tensor import ManagedTensorSlice
from sys import env_get_int


@compiler.register("use_splitk_reduction_scheme")
struct UseSplitkReductionScheme:
    @staticmethod
    fn execute(
        out: ManagedTensorSlice[DType.int32, 1],
    ):
        alias split_k_reduction_scheme = env_get_int[
            "SPLITK_REDUCTION_SCHEME", 2
        ]()
        out[0] = split_k_reduction_scheme
