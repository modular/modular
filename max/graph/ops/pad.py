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
    mode: Literal["constant"] = "constant",
    value: TensorValueLike = 0,
) -> TensorValue:
    """Pads a tensor with constant values.

    Adds padding to the input tensor using the specified padding values.
    Currently only constant padding mode is supported.

    Args:
        input: The input tensor to pad.
        paddings: Sequence of padding values. The padding values are applied
            symmetrically to each dimension. For a tensor with rank N,
            paddings should contain 2*N values: [pad_before_dim0, pad_after_dim0,
            pad_before_dim1, pad_after_dim1, ...].
        mode: The padding mode. Currently only "constant" is supported.
        value: The constant value to use for padding.
    """
    input = TensorValue(input)
    paddings = list(paddings)

    if mode != "constant":
        raise ValueError("only constant padding supported")

    if any(x < 0 for x in paddings):
        raise ValueError(
            f"padding values must be non-negative but given {paddings}"
        )

    input = TensorValue(input)

    result_type = TensorType(
        input.dtype, _compute_result_shape(input.shape, paddings), input.device
    )

    promoted = [
        dtype_promotion._promote_to_strong(
            np.array([x]), DType.int64, DeviceRef.CPU()
        )
        for x in paddings
    ]

    padding_tensor = concat(promoted, axis=0)

    return Graph.current._add_op_generated(
        rmo.MoPadConstantOp,
        result=result_type,
        input=TensorValue(input),
        paddings=padding_tensor,
        constant=dtype_promotion._promote_to_strong(
            value, input.dtype, DeviceRef.CPU()
        ),
        output_param_decls=kgen.ParamDeclArrayAttr([]),
    )[0].tensor
