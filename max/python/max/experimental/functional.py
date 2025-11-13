# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Provides experimental functional APIs for tensor operations.

.. warning::
    This module contains experimental APIs that are subject to change or
    removal in future versions. Use with caution in production environments.

This module provides functional-style tensor operations that work seamlessly
with both MAX Graph construction and eager Tensor execution. All operations
are wrapped versions of the core graph operations that automatically handle
different execution contexts.
These operations can be used in both graph construction and eager execution.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TypeVar

from typing_extensions import ParamSpec

from .. import driver
from ..graph import BufferValue, Graph, TensorValue, ops
from . import tensor

Args = ParamSpec("Args")
Result = TypeVar("Result")
Op = Callable[Args, Result]


def functional(op: Op) -> Op:
    """Decorator that converts a graph operation to support multiple tensor
    types."""

    def to_tensor(
        result: driver.Tensor | tensor.Tensor | TensorValue | BufferValue,
    ) -> tensor.Tensor:
        if isinstance(result, tensor.Tensor):
            return result
        if isinstance(result, driver.Tensor):
            return tensor.Tensor(storage=result)
        return tensor.Tensor.from_graph_value(result)

    @functools.wraps(op)
    def wrapped(*args, **kwargs):  # noqa: ANN202
        try:
            graph = Graph.current
        except LookupError:
            # No graph, use Tensor compute graph.
            graph = tensor.GRAPH.graph

        with graph:
            results = op(*args, **kwargs)
        if results is None:
            return
        elif isinstance(
            results, driver.Tensor | tensor.Tensor | TensorValue | BufferValue
        ):
            return to_tensor(results)
        return [to_tensor(result) for result in results]

    return wrapped


#: Computes the absolute value element-wise.
#: See :func:`max.graph.ops.abs` for details.
abs = functional(ops.abs)
#: Adds two tensors element-wise.
#: See :func:`max.graph.ops.add` for details.
add = functional(ops.add)
#: Concatenate values from multiple devices.
#: See :func:`max.graph.ops.allgather` for details.
allgather = functional(ops.allgather)
#: Sum values from multiple devices.
#: See :func:`max.graph.ops.allreduce.sum` for details.
allreduce_sum = functional(ops.allreduce.sum)
#: Returns the indices of the maximum values along an axis.
#: See :func:`max.graph.ops.argmax` for details.
argmax = functional(ops.argmax)
#: Returns the indices of the minimum values along an axis.
#: See :func:`max.graph.ops.argmin` for details.
argmin = functional(ops.argmin)
#: Returns the indices that would sort a tensor along an axis.
#: See :func:`max.graph.ops.argsort` for details.
argsort = functional(ops.argsort)
#: Converts a tensor to interleaved complex representation.
#: See :func:`max.graph.ops.as_interleaved_complex` for details.
as_interleaved_complex = functional(ops.as_interleaved_complex)
#: Computes the inverse hyperbolic tangent element-wise.
#: See :func:`max.graph.ops.atanh` for details.
atanh = functional(ops.atanh)
#: Applies 2D average pooling.
#: See :func:`max.graph.ops.avg_pool2d` for details.
avg_pool2d = functional(ops.avg_pool2d)
#: Copies a tensor setting everything outside a central band to zero.
#: See :func:`max.graph.ops.band_part` for details.
band_part = functional(ops.band_part)
#: Broadcasts a tensor to a new shape.
#: See :func:`max.graph.ops.broadcast_to` for details.
broadcast_to = functional(ops.broadcast_to)
#: Sets a tensor buffer to new values.
#: See :func:`max.graph.ops.buffer_store` for details.
buffer_store = functional(ops.buffer_store)
#: Sets a slice of a tensor buffer to new values.
#: See :func:`max.graph.ops.buffer_store_slice` for details.
buffer_store_slice = functional(ops.buffer_store_slice)
#: Casts a tensor to a different data type.
#: See :func:`max.graph.ops.cast` for details.
cast = functional(ops.cast)
#: Splits a tensor into chunks along a dimension.
#: See :func:`max.graph.ops.chunk` for details.
chunk = functional(ops.chunk)
#: Concatenates a list of tensors along an axis.
#: See :func:`max.graph.ops.concat` for details.
concat = functional(ops.concat)
#: Creates a constant tensor.
#: See :func:`max.graph.ops.constant` for details.
constant = functional(ops.constant)
#: Creates a constant tensor from external data.
#: See :func:`max.graph.ops.constant_external` for details.
constant_external = functional(ops.constant_external)
#: Applies 2D convolution.
#: See :func:`max.graph.ops.conv2d` for details.
conv2d = functional(ops.conv2d)
#: Applies 2D transposed convolution.
#: See :func:`max.graph.ops.conv2d_transpose` for details.
conv2d_transpose = functional(ops.conv2d_transpose)
#: Applies 3D convolution.
#: See :func:`max.graph.ops.conv3d` for details.
conv3d = functional(ops.conv3d)
#: Computes the cosine element-wise.
#: See :func:`max.graph.ops.cos` for details.
cos = functional(ops.cos)
#: Computes the cumulative sum along an axis.
#: See :func:`max.graph.ops.cumsum` for details.
cumsum = functional(ops.cumsum)
#: Applies a custom operation.
#: See :func:`max.graph.ops.custom` for details.
custom = functional(ops.custom)
#: Divides two tensors element-wise.
#: See :func:`max.graph.ops.div` for details.
div = functional(ops.div)
#: Computes element-wise equality comparison.
#: See :func:`max.graph.ops.equal` for details.
equal = functional(ops.equal)
#: Computes the error function element-wise.
#: See :func:`max.graph.ops.erf` for details.
erf = functional(ops.erf)
#: Computes the exponential element-wise.
#: See :func:`max.graph.ops.exp` for details.
exp = functional(ops.exp)
#: Flattens a tensor.
#: See :func:`max.graph.ops.flatten` for details.
flatten = functional(ops.flatten)
#: Computes the floor element-wise.
#: See :func:`max.graph.ops.floor` for details.
floor = functional(ops.floor)
#: Performs tensor folding operation.
#: See :func:`max.graph.ops.fold` for details.
fold = functional(ops.fold)
#: Gathers values along an axis specified by indices.
#: See :func:`max.graph.ops.gather` for details.
gather = functional(ops.gather)
#: Gathers values using multi-dimensional indices.
#: See :func:`max.graph.ops.gather_nd` for details.
gather_nd = functional(ops.gather_nd)
#: Applies the Gaussian Error Linear Unit (GELU) activation.
#: See :func:`max.graph.ops.gelu` for details.
gelu = functional(ops.gelu)
#: Computes element-wise greater-than comparison.
#: See :func:`max.graph.ops.greater` for details.
greater = functional(ops.greater)
#: Computes element-wise greater-than-or-equal comparison.
#: See :func:`max.graph.ops.greater_equal` for details.
greater_equal = functional(ops.greater_equal)
#: Creates a Hann window.
#: See :func:`max.graph.ops.hann_window` for details.
hann_window = functional(ops.hann_window)
#: Computes the inverse real FFT.
#: See :func:`max.graph.ops.irfft` for details.
irfft = functional(ops.irfft)
#: Checks for infinite values element-wise.
#: See :func:`max.graph.ops.is_inf` for details.
is_inf = functional(ops.is_inf)
#: Checks for NaN values element-wise.
#: See :func:`max.graph.ops.is_nan` for details.
is_nan = functional(ops.is_nan)
#: Applies layer normalization.
#: See :func:`max.graph.ops.layer_norm` for details.
layer_norm = functional(ops.layer_norm)
#: Computes the natural logarithm element-wise.
#: See :func:`max.graph.ops.log` for details.
log = functional(ops.log)
#: Computes log(1 + x) element-wise.
#: See :func:`max.graph.ops.log1p` for details.
log1p = functional(ops.log1p)
#: Computes element-wise logical AND.
#: See :func:`max.graph.ops.logical_and` for details.
logical_and = functional(ops.logical_and)
#: Computes element-wise logical NOT.
#: See :func:`max.graph.ops.logical_not` for details.
logical_not = functional(ops.logical_not)
#: Computes element-wise logical OR.
#: See :func:`max.graph.ops.logical_or` for details.
logical_or = functional(ops.logical_or)
#: Computes element-wise logical XOR.
#: See :func:`max.graph.ops.logical_xor` for details.
logical_xor = functional(ops.logical_xor)
#: Applies the log softmax function.
#: See :func:`max.graph.ops.logsoftmax` for details.
logsoftmax = functional(ops.logsoftmax)
#: Scatters values according to a mask.
#: See :func:`max.graph.ops.masked_scatter` for details.
masked_scatter = functional(ops.masked_scatter)
#: Performs matrix multiplication.
#: See :func:`max.graph.ops.matmul` for details.
matmul = functional(ops.matmul)
#: Returns the maximum values along an axis.
#: See :func:`max.graph.ops.max` for details.
max = functional(ops.max)
#: Applies 2D max pooling.
#: See :func:`max.graph.ops.max_pool2d` for details.
max_pool2d = functional(ops.max_pool2d)
#: Computes the mean along specified axes.
#: See :func:`max.graph.ops.mean` for details.
mean = functional(ops.mean)
#: Returns the minimum values along an axis.
#: See :func:`max.graph.ops.min` for details.
min = functional(ops.min)
#: Computes the modulo operation element-wise.
#: See :func:`max.graph.ops.mod` for details.
mod = functional(ops.mod)
#: Multiplies two tensors element-wise.
#: See :func:`max.graph.ops.mul` for details.
mul = functional(ops.mul)
#: Negates a tensor element-wise.
#: See :func:`max.graph.ops.negate` for details.
negate = functional(ops.negate)
#: Returns the indices of non-zero elements.
#: See :func:`max.graph.ops.nonzero` for details.
nonzero = functional(ops.nonzero)
#: Computes element-wise inequality comparison.
#: See :func:`max.graph.ops.not_equal` for details.
not_equal = functional(ops.not_equal)
#: Computes the outer product of two vectors.
#: See :func:`max.graph.ops.outer` for details.
outer = functional(ops.outer)
#: Pads a tensor.
#: See :func:`max.graph.ops.pad` for details.
pad = functional(ops.pad)
#: Permutes the dimensions of a tensor.
#: See :func:`max.graph.ops.permute` for details.
permute = functional(ops.permute)
#: Raises tensor elements to a power.
#: See :func:`max.graph.ops.pow` for details.
pow = functional(ops.pow)
#: Creates a tensor with evenly spaced values.
#: See :func:`max.graph.ops.range` for details.
range = functional(ops.range)
#: Applies the ReLU activation function.
#: See :func:`max.graph.ops.relu` for details.
relu = functional(ops.relu)
#: Repeats elements of a tensor.
#: See :func:`max.graph.ops.repeat_interleave` for details.
repeat_interleave = functional(ops.repeat_interleave)
#: Reshapes a tensor to a new shape.
#: See :func:`max.graph.ops.reshape` for details.
reshape = functional(ops.reshape)
#: Rounds tensor values element-wise.
#: See :func:`max.graph.ops.round` for details.
round = functional(ops.round)
#: Computes the reciprocal square root element-wise.
#: See :func:`max.graph.ops.rsqrt` for details.
rsqrt = functional(ops.rsqrt)
#: Scatters values along an axis.
#: See :func:`max.graph.ops.scatter` for details.
scatter = functional(ops.scatter)
#: Scatters values using multi-dimensional indices.
#: See :func:`max.graph.ops.scatter_nd` for details.
scatter_nd = functional(ops.scatter_nd)
#: Applies the sigmoid activation function.
#: See :func:`max.graph.ops.sigmoid` for details.
sigmoid = functional(ops.sigmoid)
#: Applies the SiLU (Swish) activation function.
#: See :func:`max.graph.ops.silu` for details.
silu = functional(ops.silu)
#: Computes the sine element-wise.
#: See :func:`max.graph.ops.sin` for details.
sin = functional(ops.sin)
#: Slices a tensor along specified dimensions.
#: See :func:`max.graph.ops.slice_tensor` for details.
slice_tensor = functional(ops.slice_tensor)
#: Applies the softmax function.
#: See :func:`max.graph.ops.softmax` for details.
softmax = functional(ops.softmax)
#: Splits a tensor into multiple tensors.
#: See :func:`max.graph.ops.split` for details.
split = functional(ops.split)
#: Computes the square root element-wise.
#: See :func:`max.graph.ops.sqrt` for details.
sqrt = functional(ops.sqrt)
#: Removes dimensions of size 1.
#: See :func:`max.graph.ops.squeeze` for details.
squeeze = functional(ops.squeeze)
#: Stacks tensors along a new dimension.
#: See :func:`max.graph.ops.stack` for details.
stack = functional(ops.stack)
#: Subtracts two tensors element-wise.
#: See :func:`max.graph.ops.sub` for details.
sub = functional(ops.sub)
#: Computes the sum along specified axes.
#: See :func:`max.graph.ops.sum` for details.
sum = functional(ops.sum)
#: Computes the hyperbolic tangent element-wise.
#: See :func:`max.graph.ops.tanh` for details.
tanh = functional(ops.tanh)
#: Tiles a tensor by repeating it.
#: See :func:`max.graph.ops.tile` for details.
tile = functional(ops.tile)
#: Returns the k largest elements along an axis.
#: See :func:`max.graph.ops.top_k` for details.
top_k = functional(ops.top_k)
#: Transfers a tensor to a specified device.
#: See :func:`max.graph.ops.transfer_to` for details.
transfer_to = functional(ops.transfer_to)
#: Transposes a tensor.
#: See :func:`max.graph.ops.transpose` for details.
transpose = functional(ops.transpose)
#: Truncates tensor values element-wise.
#: See :func:`max.graph.ops.trunc` for details.
trunc = functional(ops.trunc)
#: Adds dimensions of size 1.
#: See :func:`max.graph.ops.unsqueeze` for details.
unsqueeze = functional(ops.unsqueeze)
#: Selects elements from two tensors based on a condition.
#: See :func:`max.graph.ops.where` for details.
where = functional(ops.where)
