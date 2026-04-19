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
"""Op implementation for rms_norm."""

from max._core.dialects import builtin, kgen, mo

from ..graph import Graph
from ..type import DeviceRef, TensorType
from ..value import TensorValue, TensorValueLike
from .constant import constant


def rms_norm(
    input: TensorValueLike,
    weight: TensorValueLike,
    epsilon: float,
    weight_offset: float = 0.0,
    multiply_before_cast: bool = False,
) -> TensorValue:
    """Performs Root Mean Square layer normalization.

    Computes ``output = input / rms(input) * weight`` where
    ``rms(x) = sqrt(mean(x^2) + epsilon)``.

    When ``multiply_before_cast`` is ``False`` (Llama-style), the input is
    cast to the output dtype before multiplication by the weight.  When
    ``True`` (Gemma-style), the multiplication is performed before the cast.

    Args:
        input: The input tensor to normalize.
        weight: The weight tensor whose shape must match the last dimension
            of ``input``.
        epsilon: A small value added to the denominator for numerical
            stability.
        weight_offset: A value added to the weight before normalization.
            Typically ``1`` for Gemma-like normalization and ``0`` otherwise.
        multiply_before_cast: Whether to multiply before casting to the
            output dtype.

    Returns:
        A normalized tensor with the same shape and dtype as ``input``.

    Raises:
        ValueError: If weight shape doesn't match the last dimension of input.
    """
    input = TensorValue(input)
    weight = TensorValue(weight)

    if input.shape[-1:] != weight.shape:
        raise ValueError(
            f"RMSNorm: Could not apply weight shape {weight.shape} to input"
            f" shape {input.shape}, weight shape must match the final input"
            f" dimension."
        )

    return Graph.current._add_op_generated(
        mo.ReduceRmsNormOp,
        result=TensorType(
            dtype=input.dtype, shape=input.shape, device=input.device
        ),
        input=input,
        weight=weight,
        epsilon=constant(epsilon, input.dtype, DeviceRef.CPU()),
        weight_offset=constant(weight_offset, input.dtype, DeviceRef.CPU()),
        multiply_before_cast=builtin.BoolAttr(multiply_before_cast),
        output_param_decls=kgen.ParamDeclArrayAttr([]),
    )[0].tensor
