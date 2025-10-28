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
"""Op implementation for argsort."""

from max.dtype import DType

from ..type import TensorType
from ..value import StrongTensorValueLike, TensorValue
from .custom import custom


def argsort(x: StrongTensorValueLike, ascending: bool = True) -> TensorValue:
    """Returns the indices that would sort a tensor.

    This function returns the indices that would sort the input tensor along
    its first dimension. The returned indices are of type int64.

    Args:
        x: Input tensor to be sorted.
        ascending: If True (default), sort in ascending order. If False, sort in
            descending order.

    Returns:
        A tensor of indices of the same shape as the input tensor.
    """
    x = TensorValue(x)
    if x.rank != 1:
        raise ValueError("argsort only implemented for input tensors of rank 1")
    return custom(
        "mx.argsort",
        x.device,
        [x],
        out_types=[
            TensorType(dtype=DType.int64, shape=x.shape, device=x.device)
        ],
        parameters={
            "ascending": ascending,
        },
    )[0].tensor
