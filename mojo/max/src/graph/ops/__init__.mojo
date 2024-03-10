# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements various ops for the graph-building APIs."""

from .casting import (
    shape_of,
    cast,
    squeeze,
    unsqueeze,
    reshape,
    reshape_like,
    transpose_matrix,
    transpose,
)
from .complex import as_complex, as_interleaved_complex, as_real, mul_complex

# TODO(33370): rename this to `ops.custom`
from .custom_ops import custom, custom_nv
from .elementwise import (
    add,
    cos,
    div,
    mul,
    pow,
    rsqrt,
    sigmoid,
    silu,
    sin,
    softmax,
    sub,
)
from .linalg import (
    band_part,
    batch_matmul,
    matmul,
    matmul_by_matrix,
    matmul_broadcast,
    outer,
)
from .lists import list
from .reduction import mean, arg_max
from .slicing import gather, slice, split, concat, stack
