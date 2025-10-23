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
"""Op implementation for reshape."""

from max.mlir.dialects import rmo

from ..graph import Graph
from ..type import Shape, ShapeLike
from ..value import TensorValue, TensorValueLike


def reshape(x: TensorValueLike, shape: ShapeLike) -> TensorValue:
    """Reshapes a symbolic tensor.

    The number and order of the elements in the tensor is unchanged.
    In other words, if you were to iterate over elements in the tensor
    by major dimension to minor dimension, the iteration order would stay
    the same.

    If a value of -1 is present in the shape, that dimension becomes
    an automatically calculated dimension collecting all unspecified dimensions.
    Its length becomes the number of elements in the original tensor
    divided by the product of elements of the reshape.

    Args:
        x: The input symbolic tensor to reshape.
           This tensor may not contain any dynamic dimensions.
        shape: The new shape as a list of dimensions.
               Dynamic dimensions are not allowed.
               A single dimension may be `-1`.

    Returns:
        A symbolic tensor with the same elements as the original tensor, but
        in a new shape. Its symbolic shape is the same as :code:`shape`.

    Raises:
        ValueError: if input and target shapes' number of elements mismatch.
    """
    return Graph.current._add_op(
        rmo.reshape, TensorValue(x), new_shape=Shape(shape).to_mlir()
    )[0].tensor
