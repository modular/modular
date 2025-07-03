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

from max.mlir.dialects import rmo

from ..graph import Graph
from ..type import TensorType, _ChainType
from ..value import BufferValue, TensorValue
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
    in_chain = Graph.current._current_chain

    output = Graph.current._add_op(
        rmo.mo_mutable_load,
        TensorType(x.dtype, x.shape, x.device).to_mlir(),
        _ChainType().to_mlir(),
        x,
        in_chain,
    )

    Graph.current._update_chain(output[1])

    return TensorValue(output[0])


def buffer_store(destination: BufferValue, source: TensorValue) -> None:
    """Stores the input tensor into the inout buffer.

    It stores the immutable input tensor `x` in the mutable tensor `y`.
    This is semantically equivalent to a copy from `x` tensor to the `y` buffer.

    Args:
        x: The tensor to be stored in the buffer.
        y: The buffer to store the tensor in.
    """
    in_chain = Graph.current._current_chain

    output_chain = Graph.current._add_op(
        rmo.mo_mutable_store, destination, source, in_chain
    )[0]

    Graph.current._update_chain(output_chain)


def buffer_store_slice(
    destination: BufferValue, source: TensorValue, indices: SliceIndices
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
    in_chain = Graph.current._current_chain

    starts, stops, steps, unsqueezed_shape, squeezed_shape = (
        _slice_and_output_tensors(destination, indices)
    )

    if source.shape != squeezed_shape:
        raise ValueError(
            f"expected source to have shape {squeezed_shape}, but source had"
            f" shape {source.shape}"
        )

    output_chain = Graph.current._add_op(
        rmo.mo_mutable_store_slice,
        destination,
        source.reshape(unsqueezed_shape),
        starts,
        stops,
        steps,
        in_chain,
    )[-1]

    Graph.current._update_chain(output_chain)
