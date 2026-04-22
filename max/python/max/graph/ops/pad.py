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

"""Op implementation for pad."""

from collections.abc import Iterable
from typing import Literal

import numpy as np
from max._core.dialects import kgen, rmo

from .. import dtype_promotion
from ..graph import Graph
from ..type import DeviceRef, DType, Shape, TensorType
from ..value import TensorValue, TensorValueLike
from .concat import concat


def _compute_result_shape(input_shape: Shape, paddings: list[int]) -> Shape:
    assert len(paddings) == 2 * len(input_shape)

    new_shape = Shape(input_shape)
    for i, s in enumerate(new_shape):
        new_shape[i] = s + paddings[2 * i] + paddings[2 * i + 1]

    return new_shape


def pad(
    input: TensorValueLike,
    paddings: Iterable[int],
    mode: Literal["constant", "reflect", "edge"] = "constant",
    value: TensorValueLike = 0,
) -> TensorValue:
    """Pads a tensor along every dimension.

    Adds padding to the input tensor using the specified padding values and
    mode.

    Args:
        input: The input tensor to pad.
        paddings: Sequence of padding values. For a tensor with rank N,
            paddings must contain 2*N non-negative integers in the order
            ``[pad_before_dim0, pad_after_dim0, pad_before_dim1,
            pad_after_dim1, ...]``.
        mode: The padding mode.  Supported values:

            * ``"constant"`` - fill padded cells with ``value``.
            * ``"reflect"``  - reflect values about the content-region
              edges (excludes the boundary element, equivalent to
              ``numpy.pad`` with ``mode='reflect'``).
            * ``"edge"``     - repeat the nearest boundary element
              (equivalent to ``numpy.pad`` with ``mode='edge'``).
        value: The constant fill value (only used when ``mode='constant'``).
            Defaults to 0.

    Returns:
        A symbolic tensor with the same dtype as ``input``, padded along
        each dimension according to ``paddings``.

    Raises:
        ValueError: If ``mode`` is not one of the supported values, or if
            any padding value is negative.
    """
    input = TensorValue(input)
    paddings = list(paddings)

    if mode not in ("constant", "reflect", "edge"):
        raise ValueError(
            f"unsupported padding mode {mode!r}; "
            "expected 'constant', 'reflect', or 'edge'"
        )

    if any(x < 0 for x in paddings):
        raise ValueError(
            f"padding values must be non-negative but given {paddings}"
        )

    result_type = TensorType(
        input.dtype, _compute_result_shape(input.shape, paddings), input.device
    )

    promoted_paddings = [
        dtype_promotion._promote_to_strong(
            np.array([x]), DType.int64, DeviceRef.CPU()
        )
        for x in paddings
    ]
    padding_tensor = concat(promoted_paddings, axis=0)

    if mode == "constant":
        return Graph.current._add_op_generated(
            rmo.MoPadConstantOp,
            result=result_type,
            input=input,
            paddings=padding_tensor,
            constant=dtype_promotion._promote_to_strong(
                value, input.dtype, DeviceRef.CPU()
            ),
            output_param_decls=kgen.ParamDeclArrayAttr([]),
        )[0].tensor

    if mode == "reflect":
        return Graph.current._add_op_generated(
            rmo.MoPadReflectOp,
            result=result_type,
            input=input,
            paddings=padding_tensor,
            output_param_decls=kgen.ParamDeclArrayAttr([]),
        )[0].tensor

    # mode == "edge"
    return Graph.current._add_op_generated(
        rmo.MoPadRepeatOp,
        result=result_type,
        input=input,
        paddings=padding_tensor,
        output_param_decls=kgen.ParamDeclArrayAttr([]),
    )[0].tensor
