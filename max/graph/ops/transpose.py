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
"""Op implementation for transpose."""

import numpy as np
from max.dtype import DType
from max.mlir.dialects import rmo

from ..graph import Graph
from ..type import DeviceRef
from ..value import TensorType, TensorValue, TensorValueLike
from .constant import constant
from .utils import check_axis_in_bounds


def _axis_bounds(rank: int) -> tuple[int, int]:
    if rank == 0:
        return -1, 0
    return -rank, rank - 1


def transpose(x: TensorValueLike, axis_1: int, axis_2: int) -> TensorValue:
    """Transposes two axes of a symbolic tensor.
    For more information, see :obj:`~max.graph.TensorValue.transpose()`.

    Args:
        x: The input symbolic tensor to transpose.
        axis_1: One of the two axes to transpose. If negative, this indexes
           from the end of the tensor. For example,
           :code:`transpose(v, -1, -2)` transposes the last two axes.
        axis_2: The other axis to transpose. May also be negative to index from
           the end of the tensor.

    Returns:
        A new symbolic tensor with the two specified axes transposed.
        It has the same elements and dtype, but the order of the elements
        is different according to the transposition.
    """
    v = TensorValue(x)

    rank = len(v.shape)

    check_axis_in_bounds(axis_1, rank, _axis_bounds, "axis_1")
    check_axis_in_bounds(axis_2, rank, _axis_bounds, "axis_2")

    if axis_1 < 0:
        axis_1 += rank
    if axis_2 < 0:
        axis_2 += rank

    new_shape = v.shape
    indices = np.array(range(len(new_shape)))

    # Only change the shape for non-zero rank tensors.
    if rank > 0:
        new_shape[axis_1], new_shape[axis_2] = (
            new_shape[axis_2],
            new_shape[axis_1],
        )
        indices[axis_1], indices[axis_2] = axis_2, axis_1

    return Graph.current._add_op(
        rmo.mo_transpose,
        TensorType(dtype=v.dtype, shape=new_shape, device=v.device).to_mlir(),
        v,
        constant(indices, DType.int64, DeviceRef.CPU()),
    )[0].tensor
