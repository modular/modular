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
"""Op implementation for layer_norm."""

from max._core.dialects import kgen, mo

from .. import dtype_promotion
from ..dim import StaticDim
from ..graph import Graph
from ..type import DeviceRef
from ..value import TensorValue, TensorValueLike
from .constant import constant


def layer_norm(
    input: TensorValue,
    gamma: TensorValueLike,
    beta: TensorValueLike,
    epsilon: float,
) -> TensorValue:
    """Performs layer normalization.

    Args:
        input: The input tensor to normalize.
        gamma: The gamma parameter of the normalization.
        beta: The beta parameter of the normalization.
        epsilon: The epsilon parameter of the normalization.

    Returns:
        A graph tensor value with the normalization applied.

    Raises:
        ValueError: If gamma size doesn't match the last dimension of input.
        ValueError: If beta size doesn't match the last dimension of input.
        ValueError: If epsilon is not positive.
    """
    if isinstance(gamma, TensorValue) and isinstance(
        input.shape[-1], StaticDim
    ):
        gamma_tensor = gamma

        # Check that gamma size matches the last dimension of input
        if gamma_tensor.shape[0] != input.shape[-1]:
            raise ValueError(
                f"Gamma size {gamma_tensor.shape[0]} does not match dimension of reduction {input.shape[-1]}."
            )

    if isinstance(beta, TensorValue) and isinstance(input.shape[-1], StaticDim):
        beta_tensor = beta

        # Check that beta size matches the last dimension of input
        if beta_tensor.shape[0] != input.shape[-1]:
            raise ValueError(
                f"Beta size {beta_tensor.shape[0]} does not match dimension of reduction {input.shape[-1]}."
            )

    # Check that epsilon is positive
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    input, gamma = dtype_promotion._promote_weak_dtypes(input, gamma)
    input, beta = dtype_promotion._promote_weak_dtypes(input, beta)
    return Graph.current._add_op_generated(
        mo.LayerNormOp,
        input._mlir_value.type,
        input,
        gamma,
        beta,
        constant(epsilon, input.dtype, DeviceRef.CPU()),
        kgen.ParamDeclArrayAttr([]),
    )[0].tensor
