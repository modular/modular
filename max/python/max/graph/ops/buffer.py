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
"""Ops for reading and writing mutable buffers in a graph."""

from max._core.dialects import kgen, mo, rmo

from ..graph import Graph
from ..type import BufferType, TensorType
from ..value import BufferValue, BufferValueLike, TensorValue, TensorValueLike
from .slice_tensor import SliceIndices, _slice_and_output_tensors
from .validation import assert_same_device


def buffer_load(x: BufferValue) -> TensorValue:
    """Loads the contents of a buffer into a value-semantic tensor.

    Copies the mutable ``x`` buffer into a new :class:`~max.graph.TensorValue`
    that you can use in value-semantic operations. To write a tensor back into a
    buffer, use :func:`buffer_store`.

    The following example reads a buffer input into a tensor:

    .. code-block:: python

        from max.dtype import DType
        from max.graph import BufferType, DeviceRef, Graph, ops

        buffer_type = BufferType(DType.float32, shape=[2, 2], device=DeviceRef.CPU())

        with Graph("load_demo", input_types=[buffer_type]) as graph:
            loaded = ops.buffer_load(graph.inputs[0].buffer)
            graph.output(loaded)

            print(f"shape: {loaded.shape}")  # Output: shape: [Dim(2), Dim(2)]

    Args:
        x: The buffer to load into a tensor.

    Returns:
        A new tensor holding a copy of the buffer's contents.
    """
    in_chain = Graph.current.device_chains[x.device]

    result, output_chain = Graph.current._add_op_generated(
        rmo.MoMutableLoadOp,
        TensorType(x.dtype, x.shape, x.device),
        mo.ChainType(),
        x,
        kgen.ParamDeclArrayAttr([]),
        in_chain,
    )

    Graph.current.device_chains[x.device] = output_chain

    return result.tensor


def buffer_store(destination: BufferValueLike, source: TensorValueLike) -> None:
    """Stores a tensor into a buffer, overwriting its contents.

    Copies the value-semantic ``source`` tensor into the mutable
    ``destination`` buffer in place. Pair it with :func:`buffer_load` to read
    the buffer back. This is how a graph mutates persistent state, such as
    writing a new entry into a key-value cache.

    The following example reads a buffer, adds one, and writes the result back:

    .. code-block:: python

        from max.dtype import DType
        from max.graph import BufferType, DeviceRef, Graph, ops

        buffer_type = BufferType(DType.float32, shape=[4], device=DeviceRef.CPU())

        with Graph("store_demo", input_types=[buffer_type]) as graph:
            state = graph.inputs[0].buffer
            updated = ops.buffer_load(state) + 1
            ops.buffer_store(state, updated)
            graph.output(updated)

    Args:
        destination: The buffer to write into.
        source: The tensor whose contents are copied into the buffer.
    """
    destination = BufferValue(destination)
    in_chain = Graph.current.device_chains[destination.device]

    output_chain = Graph.current._add_op_generated(
        rmo.MoMutableStoreOp,
        mo.ChainType(),
        destination,
        TensorValue(source),
        kgen.ParamDeclArrayAttr([]),
        in_chain,
    )[0]

    Graph.current.device_chains[destination.device] = output_chain


def buffer_create(type: BufferType) -> BufferValue:
    """Creates a new buffer of the given type.

    Allocates a fresh :class:`~max.graph.BufferValue` inside the graph, rather
    than taking one as a graph input. Use it when a graph needs scratch mutable
    state that isn't passed in from outside.

    The following example creates a buffer and reads it back:

    .. code-block:: python

        from max.dtype import DType
        from max.graph import BufferType, DeviceRef, Graph, ops

        with Graph("create_demo") as graph:
            buffer = ops.buffer_create(
                BufferType(DType.float32, shape=[4], device=DeviceRef.CPU())
            )
            graph.output(ops.buffer_load(buffer))

            print(f"shape: {buffer.shape}")  # Output: shape: [Dim(4)]

    Args:
        type: The type of the resulting :class:`~max.graph.BufferValue`.

    Returns:
        A new buffer of the requested type.
    """
    return Graph.current._add_op_generated(mo.BufferCreateOp, type)[0].buffer


def buffer_store_slice(
    destination: BufferValueLike, source: TensorValueLike, indices: SliceIndices
) -> None:
    """Stores the input tensor to into a slice in the input buffer.

    It stores the immutable input tensor `source` in the mutable tensor `destination`.
    This is semantically equivalent to a copy from `source` tensor to a slice in the
    `destination` buffer at index specified by `indices`.

    Args:
        destination: The buffer to store the tensor in.
        source: The tensor to be stored in the buffer.
        indices: The index in the buffer where the tensor should be stored
    """
    destination = BufferValue(destination)
    source = TensorValue(source)

    assert_same_device(destination=destination, source=source)
    in_chain = Graph.current.device_chains[destination.device]

    starts, stops, steps, unsqueezed_shape, squeezed_shape = (
        _slice_and_output_tensors(destination, indices)
    )

    if source.shape != squeezed_shape:
        raise ValueError(
            f"expected source to have shape {squeezed_shape}, but source had"
            f" shape {source.shape}"
        )

    output_chain = Graph.current._add_op_generated(
        rmo.MoMutableStoreSliceOp,
        mo.ChainType(),
        destination,
        source.reshape(unsqueezed_shape),
        starts,
        stops,
        steps,
        kgen.ParamDeclArrayAttr([]),
        in_chain,
    )[-1]

    Graph.current.device_chains[destination.device] = output_chain
