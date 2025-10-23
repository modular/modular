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

from ..value import TensorValue, TensorValueLike
from .reshape import reshape


def unsqueeze(x: TensorValueLike, axis: int) -> TensorValue:
    """Inserts a size-1 dimension into a symbolic tensor.

    Args:
        x: The input symbolic tensor to unsqueeze.
        axis: The index at which to insert a new dimension into the input's
            shape. Elements at that index or higher are shifted back.
            If negative, it indexes relative *1 plus* the rank of the tensor.
            For example, :code:`unsqueeze(v, -1)` adds a new dimension at the
            end, and :code:`unsqueeze(v, -2)` inserts the dimension immediately
            before the last dimension.

    Returns:
        A symbolic tensor with the same number of elements as the input tensor,
        whose rank is 1 larger than the rank of the input tensor. The result's
        shape at the :code:`axis` dimension is a static dimension of size 1.
    """
    x = TensorValue(x)
    rank = x.rank
    if axis < 0:
        axis += rank + 1
    if not 0 <= axis <= rank:
        raise ValueError(f"unsqueeze axis out of bounds: {axis=}, {rank=}")

    shape = x.shape
    new_shape = shape[:axis] + [1] + shape[axis:]
    return reshape(x, new_shape)
