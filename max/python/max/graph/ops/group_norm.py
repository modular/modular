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
    """Computes group normalization over the channel axis of ``input``.

    Splits the channel axis (axis 1) of ``input`` into ``num_groups``
    groups, computes the mean and variance within each group, and
    normalizes. ``gamma`` and ``beta`` then apply a per-channel affine
    transform. Useful when the batch axis is small enough that batch
    normalization is unstable.

    For example:

    .. code-block:: python

        from max.dtype import DType
        from max.graph import DeviceRef, Graph, TensorType, ops

        with Graph(
            "gn",
            input_types=[
                TensorType(DType.float32, ("batch", 128, 32, 32), DeviceRef.GPU()),
                TensorType(DType.float32, (128,), DeviceRef.GPU()),
                TensorType(DType.float32, (128,), DeviceRef.GPU()),
            ],
        ) as g:
            x, gamma, beta = g.inputs
            y = ops.group_norm(
                x.tensor, gamma.tensor, beta.tensor,
                num_groups=32, epsilon=1e-5,
            )
            g.output(y)

    Args:
        input: The tensor to normalize, of shape
            ``(batch, channels, ...)``.
        gamma: The per-channel scale applied after normalization. A 1-D
            tensor whose length matches the channel axis of ``input``.
        beta: The per-channel bias added after scaling. A 1-D tensor with
            the same shape as ``gamma``.
        num_groups: The number of groups to split the channel axis into.
            Must divide the channel size evenly.
        epsilon: A small positive constant added to the variance for
            numerical stability.

    Returns:
        A tensor with the same shape and dtype as ``input``.

    Raises:
        ValueError: If ``input`` has fewer than 2 dimensions.
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
