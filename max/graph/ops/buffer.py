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
"""Op implementation for load_buffer."""

from max._core.dialects import kgen, mo, rmo

from ..graph import Graph
from ..type import BufferType, TensorType
from ..value import BufferValue, BufferValueLike, TensorValue, TensorValueLike
from .slice_tensor import SliceIndices, _slice_and_output_tensors


def buffer_load(
    x: BufferValue,
) -> TensorValue:
    """Loads the input buffer into a tensor.

    It loads the in-place mutable tensor to an immutable tensor graph value.
    This is semantically equivalent to a copy from the mutable tensor `x` to the
    mutable value-semantic tensor `output`.

    Args:
        x: The buffer to be loaded to a tensor.

    Returns:
        A tensor graph value representing a copy of the buffer loaded.
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
    """Stores the input tensor into the in-out buffer.

    It stores the immutable input tensor `x` in the mutable tensor `y`.
    This is semantically equivalent to a copy from `x` tensor to the `y` buffer.

    Args:
        x: The tensor to be stored in the buffer.
        y: The buffer to store the tensor in.
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
    """Creates a buffer of the given type.

    Args:
        type: The type of the resulting BufferValue
    Returns:
        A new BufferValue of the requested type.
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
