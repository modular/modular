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
"""Op implementation for outer."""

from .. import dtype_promotion
from ..value import TensorValue, TensorValueLike
from .reshape import reshape


def outer(lhs: TensorValueLike, rhs: TensorValueLike) -> TensorValue:
    """Computes the outer product of two symbolic vectors.

    Args:
        lhs: The left side of the product. Whatever its shape,
            it will be flattened to a rank-1 vector.
        rhs: The right side of the product. Whatever its shape,
            it will be flattened to a rank-1 vector. Must have the
            same number of elements as `lhs`.

    Returns:
        A symbolic tensor representing the
        [outer product](https://en.wikipedia.org/wiki/Outer_product)
        of the two input vectors. It will have rank 2, with the dimension
        sizes being the number of elements of `lhs` and `rhs` respectively.
    """
    lhs, rhs = dtype_promotion._promote_weak_dtypes(lhs, rhs)
    if lhs.rank != 1 or rhs.rank != 1:
        raise ValueError("outer expected 1d-tensors as inputs")
    return reshape(lhs, [-1, 1]) * reshape(rhs, [1, -1])
