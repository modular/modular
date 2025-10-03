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
"""Op implementation for squeeze."""

from ..type import Shape
from ..value import TensorValue, TensorValueLike
from .reshape import reshape


def squeeze(x: TensorValueLike, axis: int) -> TensorValue:
    """Removes a size-1 dimension from a symbolic tensor.

    Args:
        x: The input symbolic tensor to squeeze.
        axis: The dimension to remove from the input's shape. If negative, this
              indexes from the end of the tensor. For example,
              :code:`squeeze(v, -1)` squeezes the last dimension.

    Returns:
        A symbolic tensor with the same number of elements as the input tensor,
        and whose rank is 1 less than the rank of the input tensor.
    """
    v = TensorValue(x)
    # TODO (MSDK-655): Probably want to add rmo.mo_squeeze_shape here
    shape = Shape(v.shape)
    if shape[axis] != 1:
        raise ValueError(f"Squeeze dim must be 1, got {axis=}, {shape=}")
    shape.pop(axis)
    return reshape(v, shape)
