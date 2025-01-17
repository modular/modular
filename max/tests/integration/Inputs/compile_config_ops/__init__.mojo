# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from buffer import NDBuffer
from buffer.dimlist import DimList
from register import register_internal
from sys import env_get_int


@register_internal("expose_env")
fn expose_env(output: NDBuffer[DType.int32, 1, DimList(1)]):
    alias split_k_reduction_scheme = env_get_int["SPLITK_REDUCTION_SCHEME", 2]()
    output[0] = split_k_reduction_scheme
