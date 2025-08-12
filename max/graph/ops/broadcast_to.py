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
"""Op implementation for broadcast_to."""

from __future__ import annotations

from collections.abc import Iterable

from max.mlir.dialects import rmo

from ..dim import DimLike
from ..graph import Graph
from ..shape import Shape, ShapeLike
from ..type import TensorType
from ..value import StrongTensorValueLike, TensorValue


def broadcast_to(
    x: StrongTensorValueLike,
    shape: TensorValue | ShapeLike,
    out_dims: Iterable[DimLike] | None = None,
) -> TensorValue:
    """Broadcasts a symbolic tensor.

    Broadcasts the input tensor to the specified shape.
    Dimensions in the input must be one or match the target dimension.

    Args:
        x: The input symbolic tensor to broadcast.
            This tensor may not contain any dynamic dimensions.
        shape: The new shape as a list of dimensions.
            Dynamic dimensions are not allowed.
        out_dims: Output dims used only for tensor-valued `shape`.

    Returns:
        A symbolic tensor with the same elements as the original tensor, but
        in a new shape. Its symbolic shape is the same as :code:`shape`.

    Raises:
        ValueError: if a tensor-valued shape is passed without out_dims.
    """
    x = TensorValue(x)

    if isinstance(shape, TensorValue):
        # For tensor-valued shapes, dims need to be declared in the graph.
        # Push the onus of doing so onto the caller.
        if out_dims is None:
            message = f"must pass out_dims with tensor value shape {shape}"
            raise ValueError(message)

        return Graph.current._add_op(
            rmo.mo_broadcast_to,
            TensorType(x.dtype, shape=out_dims, device=x.device).to_mlir(),
            x,
            shape._mlir_value,
        )[0].tensor

    return Graph.current._add_op(
        rmo.broadcast_to, x, new_shape=Shape(shape).to_mlir()
    )[0].tensor
