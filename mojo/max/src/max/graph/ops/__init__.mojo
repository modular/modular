# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements various ops used when building a graph.

Although the following modules provide a lot of the ops you want when building
a graph, you can also use functions in
[`Graph`](/max/api/mojo/graph/graph/Graph) to add constant values,
such as [`constant()`](/max/api/mojo/graph/graph/Graph#constant),
[`vector()`](/max/api/mojo/graph/graph/Graph#vector), and
[`scalar()`](/max/api/mojo/graph/graph/Graph#scalar).

The [`Symbol`](/max/api/mojo/graph/symbol/Symbol) type (returned by
all ops) also implements various dunder methods to support operations between
symbols, such as `+` add, `*` multiply, and `@` matmul, plus convenience
methods such as
[`reshape()`](/max/api/mojo/graph/symbol/Symbol#reshape) and
[`swapaxes()`](/max/api/mojo/graph/symbol/Symbol#swapaxes). """

from .casting import (
    broadcast_to,
    cast,
    rebind,
    reshape,
    reshape_like,
    shape_of,
    squeeze,
    transpose,
    transpose_matrix,
    unsqueeze,
)
from .complex import as_complex, as_interleaved_complex, as_real, mul_complex
from .convolution import avg_pool, conv2d, conv3d, max_pool

# TODO(33370): rename this to `ops.custom`
from .custom_ops import custom
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
from .linalg import band_part, layer_norm, matmul, outer, tile
from .lists import list
from .quantized_ops import qmatmul
from .reduction import arg_max, mean
from .repeat_interleave import repeat_interleave
from .slicing import concat, gather, select, slice, split, stack
