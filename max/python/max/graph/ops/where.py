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
"""Op implementation for where."""

from max.dtype import DType
from max.mlir.dialects import rmo

from .. import dtype_promotion
from ..graph import Graph
from ..value import TensorValue, TensorValueLike
from .validation import assert_same_device


def where(
    condition: TensorValueLike, x: TensorValueLike, y: TensorValueLike
) -> TensorValue:
    """Returns element-wise ``condition ? x : y`` for input tensors condition, x, and y.

    Args:
        condition: The condition tensor to use for selecting elementwise
                   values. This tensor must have a boolean dtype.
        x: If the condition is true at a position, the value from the same
           position in this tensor will be selected.
        y: If the condition is false at a position, the value from the same
           position in this tensor will be selected.

    Returns:
        A new symbolic tensor holding either values from either ``x`` or ``y``,
        based on the elements in ``condition``.
    """
    condition = TensorValue(condition)
    if condition.dtype != DType.bool:
        raise ValueError(
            f"Expected condition to be a boolean tensor, but got a tensor with dtype {condition.dtype}"
        )

    x, y = dtype_promotion._promote_weak_dtypes(x, y)
    assert_same_device(condition=condition, x=x, y=y)
    return Graph.current._add_op(rmo.select, condition, x, y)[0].tensor
