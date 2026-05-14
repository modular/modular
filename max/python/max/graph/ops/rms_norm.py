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
    """Computes root mean square normalization over the last dimension of ``input``.

    The output is ``input / rms(input) * (weight + weight_offset)`` where
    ``rms(x) = sqrt(mean(x ** 2) + epsilon)``. Reduction runs over the last
    axis of ``input`` and is broadcast back across the leading axes. See
    `Root Mean Square Layer Normalization
    <https://arxiv.org/abs/1910.07467>`_ for the original formulation.

    Two variants are supported through ``weight_offset`` and
    ``multiply_before_cast``:

    - **Llama-style** (default): ``weight_offset=0`` and
      ``multiply_before_cast=False``. The normalized input is cast to the
      output dtype before multiplication by the weight.
    - **Gemma-style**: ``weight_offset=1`` and ``multiply_before_cast=True``.
      The weight is treated as ``1 + weight`` and multiplication runs in
      the reduction dtype before casting back.

    For example:

    .. code-block:: python

        from max.dtype import DType
        from max.graph import DeviceRef, Graph, TensorType, ops

        with Graph(
            "rms",
            input_types=[
                TensorType(DType.float32, ("batch", "seq", 128), DeviceRef.GPU()),
                TensorType(DType.float32, (128,), DeviceRef.GPU()),
            ],
        ) as g:
            x, weight = g.inputs
            y_llama = ops.rms_norm(x.tensor, weight.tensor, epsilon=1e-6)
            y_gemma = ops.rms_norm(
                x.tensor, weight.tensor, epsilon=1e-6,
                weight_offset=1.0, multiply_before_cast=True,
            )
            g.output(y_llama, y_gemma)

    Args:
        input: The tensor to normalize. Reduction runs over the last axis.
        weight: The scale applied after normalization. A 1-D tensor whose
            shape matches the last dimension of ``input``.
        epsilon: A small positive constant added to the mean of squares for
            numerical stability.
        weight_offset: A value added to ``weight`` before scaling. Use
            ``1.0`` for Gemma-style normalization and ``0.0`` otherwise.
            Defaults to ``0.0``.
        multiply_before_cast: Whether to multiply by the (offset) weight
            before casting the normalized input back to the output dtype.
            Llama-style sets this to ``False``. Defaults to ``False``.

    Returns:
        A tensor with the same shape and dtype as ``input``.

    Raises:
        ValueError: If ``weight`` does not match the last dimension of
            ``input``.
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
