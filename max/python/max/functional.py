# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

"""Provides functional APIs for tensor operations.

This module provides functional-style tensor operations that work seamlessly
with both MAX Graph construction and eager Tensor execution. All operations
are wrapped versions of the core graph operations that automatically handle
different execution contexts.
These operations can be used in both graph construction and eager execution.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
from collections.abc import (
    Callable,
    Coroutine,
    Generator,
    Iterable,
    Mapping,
    Sequence,
)
from pathlib import Path
from typing import Any, TypeAlias, TypeVar, overload

from max import _realization_context as rc
from max import driver, tensor
from max._mlir_context import MLIRThreadPoolExecutor
from max.dtype import DType
from max.graph import (
    BufferValue,
    Graph,
    TensorValue,
    TensorValueLike,
    Type,
    ops,
)
from max.graph.type import DeviceRef
from max.graph.value import Value
from typing_extensions import ParamSpec

#: Type alias for custom extensions paths, matching :obj:`engine.CustomExtensionsType`.
CustomExtensionType: TypeAlias = str | Path
CustomExtensionsType: TypeAlias = (
    Iterable[CustomExtensionType] | CustomExtensionType
)

Args = ParamSpec("Args")
Result = TypeVar("Result")
Op = Callable[Args, Result]


def _in_running_loop() -> bool:
    """Checks whether the caller is inside a running asyncio event loop.

    Returns:
        bool: True if currently inside a running event loop, False otherwise.
    """
    # - asyncio.get_event_loop().is_running() works in most scenarios
    # - asyncio.get_event_loop() raises in some environments
    # - use asyncio.get_running_loop() and check if it fails instead
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _run(coro: Coroutine[Any, Any, Result]) -> Result:
    """Runs a coroutine synchronously, handling nested event loops.

    If not inside an event loop, uses ``asyncio.run()``. If already inside
    an event loop (e.g., in Jupyter), runs the coroutine in a separate
    thread to avoid blocking.

    Args:
        coro: The coroutine to execute.

    Returns:
        The result of the coroutine.
    """
    # If there's no running loop, just use asyncio.run
    if not _in_running_loop():
        return asyncio.run(coro)

    # Run self.realize in another thread
    loop = asyncio.new_event_loop()
    with MLIRThreadPoolExecutor() as pool:
        fut = pool.submit(loop.run_until_complete, coro)
    return fut.result()


_ConvertableToTensor: TypeAlias = (
    driver.Buffer | tensor.Tensor | TensorValue | BufferValue
)


def _to_tensor(value: _ConvertableToTensor) -> tensor.Tensor:
    """Converts a tensor-like value to a Tensor.

    Args:
        value: A driver.Buffer, Tensor, TensorValue, or BufferValue.

    Returns:
        tensor.Tensor: The converted Tensor object.
    """
    if isinstance(value, tensor.Tensor):
        return value
    elif isinstance(value, driver.Buffer):
        return tensor.Tensor(storage=value)
    return tensor.Tensor.from_graph_value(value)


@overload
def _to_tensors(value: _ConvertableToTensor, /) -> tensor.Tensor: ...


@overload
def _to_tensors(value: None, /) -> None: ...


@overload
def _to_tensors(
    values: Iterable[_ConvertableToTensor],
) -> list[tensor.Tensor]: ...


def _to_tensors(values):
    """Converts one or more tensor-like values to Tensors.

    Handles single values, None, or iterables of tensor-like values.

    Args:
        values: None, a single tensor-like value, or an iterable of them.

    Returns:
        None if input is None, a single Tensor if input is a single value,
        or a list of Tensors if input is an iterable.
    """
    if values is None:
        return None
    if isinstance(values, _ConvertableToTensor):
        return _to_tensor(values)
    return [_to_tensor(value) for value in values]


def _return_tensors(op: Op[..., Any]) -> Op[..., Any]:
    """Decorator that converts operation results to Tensors.

    Wraps an operation so its return values are converted from graph values
    (TensorValue/BufferValue) or driver tensors to Tensor objects.

    Args:
        op: The operation to wrap.

    Returns:
        The wrapped operation that returns Tensor objects.
    """

    @functools.wraps(op)
    def wrapped(*args, **kwargs):  # noqa: ANN202
        results = op(*args, **kwargs)
        return _to_tensors(results)

    return wrapped


def in_graph_context() -> bool:
    """Checks whether the caller is inside a Graph context.

    Returns:
        bool: True if inside a ``with Graph(...):`` block, False otherwise.
    """
    try:
        _ = Graph.current
    except LookupError:
        return False
    else:
        return True


def functional(op: Op[..., Any]) -> Op[..., Any]:
    """Converts a graph operation to support multiple tensor types.

    Returns a wrapped op that can be called with various tensor types.
    """
    op = _return_tensors(op)

    @functools.wraps(op)
    def wrapped(*args, **kwargs):  # noqa: ANN202
        with contextlib.ExitStack() as stack:
            if tensor.current_realization_context(None) is None:
                ctx = (
                    rc.GraphRealizationContext(Graph.current)
                    if in_graph_context()
                    else rc.EagerRealizationContext()
                )
                stack.enter_context(ctx)
                stack.enter_context(tensor.realization_context(ctx))
            return op(*args, **kwargs)

    return wrapped


@contextlib.contextmanager
def lazy() -> Generator[None]:
    """Context manager for lazy tensor evaluation.

    Within this context, tensor operations are recorded but not executed.
    Tensors remain unrealized until explicitly awaited via ``await tensor.realize``
    or until their values are needed (e.g., by calling ``.item()``).

    This is particularly useful for creating tensors which may not ever
    be used. Lazy tensors that aren't used will never allocate memory or perform
    operations.

    Yields:
        None

    .. code-block:: python

        from max import functional as F
        from max.tensor import Tensor
        from max.nn import Linear

        with F.lazy():
            model = Linear(2, 3)

        print(model)  # Lazy weights not initialized
        # Executing the model would be fine! The weights would be created
        # on first use.
        # output = model(Tensor.ones([5, 2]))

        # Load pretrained weights, never creating the original random weights
        weights =  {
            "weight": Tensor.zeros([3, 2]),
            "bias": Tensor.zeros([3]),
        }
        model.load_state_dict(weights)
    """
    with rc.LazyRealizationContext() as ctx, tensor.realization_context(ctx):
        yield


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


@functional
def argmax(x: TensorValueLike, axis: int | None = -1) -> TensorValue:
    """Returns the indices of the maximum values along an axis.

    Args:
        x: The input tensor.
        axis: The axis along which to find the maximum indices. If None,
            finds the index of the maximum across all elements (flattened).

    Returns:
        A tensor containing the indices of the maximum values.
    """
    if axis is None:
        x = TensorValue(x).reshape([-1])
        axis = 0
    return ops.argmax(x, axis=axis)


@functional
def argmin(x: TensorValueLike, axis: int | None = -1) -> TensorValue:
    """Returns the indices of the minimum values along an axis.

    Args:
        x: The input tensor.
        axis: The axis along which to find the minimum indices. If None,
            finds the index of the minimum across all elements (flattened).

    Returns:
        A tensor containing the indices of the minimum values.
    """
    if axis is None:
        x = TensorValue(x).reshape([-1])
        axis = 0
    return ops.argmin(x, axis=axis)


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
#: Multiply two complex-valued tensors.
#: See :func:`max.graph.ops.complex.mul` for details.
complex_mul = functional(ops.complex.mul)
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


def _load_custom_extensions(
    graph: Graph, custom_extensions: CustomExtensionsType | None
) -> None:
    """Loads custom extensions into the graph if not already loaded.

    Checks the graph's kernel library to determine which extensions are already
    loaded, avoiding redundant loading.

    Args:
        graph: The graph to load extensions into.
        custom_extensions: Paths to custom extension libraries (.mojopkg files
            or Mojo source directories).
    """
    if custom_extensions is None:
        return

    if isinstance(custom_extensions, (str, Path)):
        custom_extensions = [custom_extensions]
    paths = [Path(p) for p in custom_extensions]

    graph._import_kernels(Path(p) for p in custom_extensions)


@functional
def custom(
    name: str,
    device: driver.Device | DeviceRef,
    values: Sequence[Value[Any]],
    out_types: Sequence[Type[Any]],
    parameters: Mapping[str, bool | int | str | DType] | None = None,
    custom_extensions: CustomExtensionsType | None = None,
) -> list[Value[Any]]:
    """Applies a custom operation with optional custom extension loading.

    Creates a node to execute a custom graph operation. The custom op should be
    registered by annotating a Mojo function with the ``@compiler.register``
    decorator.

    This function extends :func:`max.graph.ops.custom` with automatic loading
    of custom extension libraries, eliminating the need to manually import
    kernels before use.

    Example:
        .. code-block:: python

            from max import functional as F, Tensor
            from max.dtype import DType
            from max.driver import CPU

            x = Tensor.full([10], 10, dtype=DType.float32, device=CPU())
            y = Tensor.ones([10], dtype=DType.float32, device=CPU())

            # Use custom op with inline extension loading
            result = F.custom(
                "vector_sum",
                device=x.device,
                values=[x, y],
                out_types=[x.type],
                custom_extensions="ops.mojopkg"
            )[0]

    Args:
        name: The op name provided to ``@compiler.register``.
        device: Device that the op is assigned to. This becomes a ``target``
            parameter to the kernel.
        values: The op function's arguments.
        out_types: The list of op function's return types.
        parameters: Dictionary of extra parameters expected by the kernel.
        custom_extensions: Paths to custom extension libraries (``.mojopkg``
            files or Mojo source directories). Extensions are automatically
            loaded into the current graph if not already present.

    Returns:
        Symbolic values representing the outputs of the op in the graph.
        These correspond 1:1 with the types passed as ``out_types``.

    See Also:
        :func:`max.graph.ops.custom`: The underlying graph operation.
        :func:`inplace_custom`: For in-place custom operations.
    """
    graph = Graph.current
    _load_custom_extensions(graph, custom_extensions)
    return ops.custom(
        name=name,
        device=device,
        values=values,
        out_types=out_types,
        parameters=parameters,
    )


@functional
def inplace_custom(
    name: str,
    device: driver.Device | DeviceRef,
    values: Sequence[Value[Any]],
    out_types: Sequence[Type[Any]] | None = None,
    parameters: dict[str, bool | int | str | DType] | None = None,
    custom_extensions: CustomExtensionsType | None = None,
) -> list[Value[Any]]:
    """Applies an in-place custom operation with optional custom extension loading.

    Creates a node to execute an in-place custom graph operation. The custom op
    should be registered by annotating a Mojo function with the
    ``@compiler.register`` decorator.

    This function extends :func:`max.graph.ops.inplace_custom` with automatic
    loading of custom extension libraries, eliminating the need to manually
    import kernels before use.

    Example:
        .. code-block:: python

            from max import functional as F, Tensor
            from max.dtype import DType
            from max.driver import CPU

            # Create a buffer for in-place modification
            data = Tensor.zeros([10], dtype=DType.float32, device=CPU())

            # Use in-place custom op with inline extension loading
            F.inplace_custom(
                "my_inplace_op",
                device=data.device,
                values=[data],
                custom_extensions="ops.mojopkg"
            )

    Args:
        name: The op name provided to ``@compiler.register``.
        device: Device that the op is assigned to. This becomes a ``target``
            parameter to the kernel.
        values: The op function's arguments. At least one must be a
            :obj:`BufferValue` or :obj:`_OpaqueValue`.
        out_types: The list of op function's return types. Can be None if the
            operation has no outputs.
        parameters: Dictionary of extra parameters expected by the kernel.
        custom_extensions: Paths to custom extension libraries (``.mojopkg``
            files or Mojo source directories). Extensions are automatically
            loaded into the current graph if not already present.

    Returns:
        Symbolic values representing the outputs of the op in the graph.

    See Also:
        :func:`max.graph.ops.inplace_custom`: The underlying graph operation.
        :func:`custom`: For non-in-place custom operations.
    """
    graph = Graph.current
    _load_custom_extensions(graph, custom_extensions)
    return ops.inplace_custom(
        name=name,
        device=device,
        values=values,
        out_types=out_types,
        parameters=parameters,
    )


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


@functional
def max(
    x: TensorValueLike,
    y: TensorValueLike | None = None,
    /,
    axis: int | None = -1,
) -> TensorValue:
    """Returns the maximum values along an axis, or elementwise maximum of two tensors.

    Args:
        x: The input tensor.
        y: Optional second tensor for elementwise maximum.
        axis: The axis along which to compute the maximum (only for reduction).
            If None, computes the maximum across all elements (flattened).

    Returns:
        A tensor containing the maximum values.
    """
    if y is not None:
        # Elementwise max
        return ops.elementwise.max(x, y)
    # Reduction max
    if axis is None:
        x = TensorValue(x).reshape([-1])
        axis = 0
    return ops.reduction.max(x, axis=axis)


#: Applies 2D max pooling.
#: See :func:`max.graph.ops.max_pool2d` for details.
max_pool2d = functional(ops.max_pool2d)


@functional
def mean(x: TensorValueLike, axis: int | None = -1) -> TensorValue:
    """Computes the mean along specified axes.

    Args:
        x: The input tensor.
        axis: The axis along which to compute the mean. If None,
            computes the mean across all elements (flattened).

    Returns:
        A tensor containing the mean values.
    """
    if axis is None:
        x = TensorValue(x).reshape([-1])
        axis = 0
    return ops.mean(x, axis=axis)


@functional
def min(
    x: TensorValueLike,
    y: TensorValueLike | None = None,
    /,
    axis: int | None = -1,
) -> TensorValue:
    """Returns the minimum values along an axis, or elementwise minimum of two tensors.

    Args:
        x: The input tensor.
        y: Optional second tensor for elementwise minimum.
        axis: The axis along which to compute the minimum (only for reduction).
            If None, computes the minimum across all elements (flattened).

    Returns:
        A tensor containing the minimum values.
    """
    if y is not None:
        # Elementwise min
        return ops.elementwise.min(x, y)
    # Reduction min
    if axis is None:
        x = TensorValue(x).reshape([-1])
        axis = 0
    return ops.reduction.min(x, axis=axis)


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


@functional
def prod(x: TensorValueLike, axis: int | None = -1) -> TensorValue:
    """Computes the product along specified axes.

    Args:
        x: The input tensor.
        axis: The axis along which to compute the product. If None,
            computes the product across all elements (flattened).

    Returns:
        A tensor containing the product values.
    """
    if axis is None:
        x = TensorValue(x).reshape([-1])
        axis = 0
    return ops.prod(x, axis=axis)


#: Creates a tensor with evenly spaced values.
#: See :func:`max.graph.ops.range` for details.
arange = functional(ops.range)
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


def split(
    x: tensor.Tensor | TensorValue,
    split_size_or_sections: int | list[int],
    axis: int = 0,
) -> list[tensor.Tensor] | list[TensorValue]:
    """Splits a tensor into multiple tensors along a given dimension.

    This function supports two modes, matching PyTorch's behavior:

    - If ``split_size_or_sections`` is an **int**, splits into chunks of that
      size (the last chunk may be smaller if the dimension is not evenly
      divisible).
    - If ``split_size_or_sections`` is a **list of ints**, splits into chunks
      with exactly those sizes (must sum to the dimension size).

    .. code-block:: python

        from max import functional as F, Tensor

        x = Tensor.ones([10, 4])

        # Split into chunks of size 3 (last chunk is size 1)
        chunks = F.split(x, 3, axis=0)  # shapes: [3,4], [3,4], [3,4], [1,4]

        # Split into exact sizes
        chunks = F.split(x, [2, 3, 5], axis=0)  # shapes: [2,4], [3,4], [5,4]

    Args:
        x: The input tensor to split.
        split_size_or_sections: Either an int (chunk size) or a list of ints
            (exact sizes for each output tensor).
        axis: The dimension along which to split. Defaults to 0.

    Returns:
        A list of tensors resulting from the split.
    """
    # Get the dimension size along the split axis
    shape = x.shape
    dim_size = int(shape[axis])

    # Convert int split_size to list of sizes
    if isinstance(split_size_or_sections, int):
        chunk_size = split_size_or_sections
        num_full_chunks, remainder = divmod(dim_size, chunk_size)
        split_sizes = [chunk_size] * num_full_chunks
        if remainder > 0:
            split_sizes.append(remainder)
    else:
        split_sizes = list(split_size_or_sections)

    return functional(ops.split)(x, split_sizes, axis)


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


@functional
def sum(x: TensorValueLike, axis: int | None = -1) -> TensorValue:
    """Computes the sum along specified axes.

    Args:
        x: The input tensor.
        axis: The axis along which to compute the sum. If None,
            computes the sum across all elements (flattened).

    Returns:
        A tensor containing the sum values.
    """
    if axis is None:
        x = TensorValue(x).reshape([-1])
        axis = 0
    return ops.sum(x, axis=axis)


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
