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
    rebind,
    reshape,
    reshape_like,
    transpose_matrix,
    transpose,
)
from .complex import as_complex, as_interleaved_complex, as_real, mul_complex

# TODO(33370): rename this to `ops.custom`
from .custom_ops import custom, custom_nv
from .elementwise import (
    abs,
    add,
    cos,
    div,
    equal,
    erf,
    exp,
    floor,
    gelu,
    greater,
    greater_equal,
    is_inf,
    is_nan,
    log,
    log1p,
    logsoftmax,
    max,
    min,
    mod,
    mul,
    not_equal,
    pow,
    relu,
    round,
    roundeven,
    rsqrt,
    sigmoid,
    silu,
    sin,
    softmax,
    sqrt,
    sub,
    tanh,
    trunc,
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
