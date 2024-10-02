# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements ops used when staging a graph.

Although the following modules provide a lot of the ops you want when building
a graph, you can also use functions in
[`Graph`](/max/api/python/graph/graph/Graph) to add constant values,
such as [`constant()`](/max/api/python/graph/graph/Graph#constant),
[`vector()`](/max/api/python/graph/graph/Graph#vector), and
[`scalar()`](/max/api/python/graph/graph/Graph#scalar).

The [`TensorValue`](/max/api/python/graph/value/TensorValue) type (returned
by most ops) also implements various dunder methods to support operations
between TensorValues, such as `+` add, `*` multiply, and `@` matmul, plus
convenience methods such as
[`reshape()`](/max/api/python/graph/value/TensorValue#reshape) and
[`swapaxes()`](/max/api/python/graph/value/TensorValue#swapaxes).
"""

from .band_part import band_part
from .broadcast_to import broadcast_to
from .buffer import buffer_load, buffer_store, buffer_store_slice
from .cast import cast
from .complex import as_interleaved_complex
from .concat import concat
from .constant import constant
from .custom import custom, inplace_custom
from .debug import print
from .elementwise import *
from .gather import gather
from .layer_norm import layer_norm
from .matmul import matmul
from .outer import outer
from .quantized import dequantize, qmatmul
from .range import range
from .rebind import rebind
from .reduction import mean
from .reshape import reshape
from .select import select
from .shape_to_tensor import shape_to_tensor
from .slice_tensor import slice_tensor
from .squeeze import squeeze
from .stack import stack
from .tile import tile
from .transpose import transpose
from .unsqueeze import unsqueeze
