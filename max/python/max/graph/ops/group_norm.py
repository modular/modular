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
"""Op implementation for group_norm."""

from max._core.dialects import kgen, mo
from max.dtype import DType

from ..graph import Graph
from ..type import DeviceRef, TensorType
from ..value import TensorValue, TensorValueLike
from .constant import constant


def group_norm(
    input: TensorValueLike,
    gamma: TensorValueLike,
    beta: TensorValueLike,
    num_groups: int,
    epsilon: float,
) -> TensorValue:
    """Performs group normalization.

    Divides channels into groups and computes normalization statistics
    within each group. Useful for small batch sizes where batch
    normalization is unstable.

    Args:
        input: The input tensor of shape ``[N, C, ...]`` to normalize.
        gamma: The scale parameter of shape ``[C]``.
        beta: The bias parameter of shape ``[C]``.
        num_groups: The number of groups to divide the channels into.
        epsilon: A small value added to the denominator for numerical
            stability.

    Returns:
        A normalized tensor with the same shape as ``input``.

    Raises:
        ValueError: If the input tensor has fewer than 2 dimensions.
    """
    input = TensorValue(input)
    gamma = TensorValue(gamma)
    beta = TensorValue(beta)

    if len(input.shape) < 2:
        raise ValueError(
            f"Expected input tensor with >=2 dimensions, got shape"
            f" {input.shape}"
        )

    return Graph.current._add_op_generated(
        mo.ReduceGroupNormOp,
        result=TensorType(
            dtype=input.dtype, shape=input.shape, device=input.device
        ),
        input=input,
        gamma=gamma,
        beta=beta,
        epsilon=constant(epsilon, input.dtype, DeviceRef.CPU()),
        num_groups=constant(num_groups, DType.int32, DeviceRef.CPU()),
        output_param_decls=kgen.ParamDeclArrayAttr([]),
    )[0].tensor
