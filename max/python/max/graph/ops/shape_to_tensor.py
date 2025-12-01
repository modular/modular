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
"""Op implementation for shape_to_tensor."""

from max.mlir.dialects import rmo

from ..graph import Graph
from ..type import Shape, ShapeLike
from ..value import TensorValue


def shape_to_tensor(shape: ShapeLike) -> TensorValue:
    """Converts a shape to a tensor.

    This is useful for using a shape attribute in an op that expects a tensor
    value.

    Args:
        shape: the shape attribute of a tensor value.

    Returns:
        The TensorValue containing the same value as `shape`.

    Example:
        >>> x = ops.constant(np.zeros((1,)), DType.int64, device=DeviceRef.CPU())
        >>> result = ops.stack([
        ...     x,
        ...     ops.shape_to_tensor(x.shape),
        ... ])
        TensorValue(dtype=int64, shape=[StaticDim(dim=2), StaticDim(dim=1)])
    """
    shape = Shape(shape)
    result = Graph.current._add_op(rmo.shape_to_tensor, shape.to_mlir())[
        0
    ].tensor

    return result
