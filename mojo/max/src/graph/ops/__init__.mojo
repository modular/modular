# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements various ops used when building a graph.

Although the following modules provide a lot of the ops you want when building
a graph, you can also use functions in
[`Graph`](/engine/reference/mojo/graph/graph/Graph) to add constant values,
such as [`constant()`](/engine/reference/mojo/graph/graph/Graph#constant),
[`vector()`](/engine/reference/mojo/graph/graph/Graph#vector), and
[`scalar()`](/engine/reference/mojo/graph/graph/Graph#scalar).

The [`Symbol`](/engine/reference/mojo/graph/symbol/Symbol) type (returned by
all ops) also implements various dunder methods to support operations between
symbols, such as `+` add, `*` multiply, and `@` matmul, plus convenience
methods such as
[`reshape()`](/engine/reference/mojo/graph/symbol/Symbol#reshape) and
[`swapaxes()`](/engine/reference/mojo/graph/symbol/Symbol#swapaxes). """

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
from .convolution import conv2d, conv3d

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
