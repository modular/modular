# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from random import randn

from buffer import NDBuffer
from buffer.dimlist import DimList


fn random_normal[
    rank: Int,
    type: DType,
    output_shape: DimList,
    mean: Float64,
    variance: Float64,
](output: NDBuffer[mut=True, type, rank, _, output_shape]):
    """
    Fill `output` with values generated from Normal(mean, variance) distribution.

    Args:
        output: The output buffer.
    """
    randn(output.data, output.size(), mean, variance)
