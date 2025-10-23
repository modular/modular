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
"""Op implementation for cast."""

from max._core.dialects import mo
from max.dtype import DType

from ..graph import Graph
from ..value import StrongTensorValueLike, TensorValue


def cast(x: StrongTensorValueLike, dtype: DType) -> TensorValue:
    """Casts a symbolic tensor to a different data type.

    Args:
        x: The input tensor to cast.
        dtype: The target dtype to which the tensor is cast.

    Returns:
        A new symbolic tensor with the same shape as the input and the
        specified dtype.
    """
    x = TensorValue(x)
    cast_type = x.type.cast(dtype)
    return Graph.current._add_op_generated(mo.CastOp, cast_type, x)[0].tensor
