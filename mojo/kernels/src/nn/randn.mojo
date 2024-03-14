# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from random import randn

from closed_source_memory.buffer import NDBuffer

from utils.list import DimList


fn random_normal[
    rank: Int,
    type: DType,
    output_shape: DimList,
    mean: Float64,
    variance: Float64,
](output: NDBuffer[type, rank, output_shape]):
    """
    Fill `output` with values generated from Normal(mean, variance) distribution.

    Args:
        output: The output buffer.
    """
    randn(output.data, output.size(), mean, variance)
