# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Op implementation for pad."""

from collections.abc import Iterable
from typing import Literal

import numpy as np
from max.mlir.dialects import rmo

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

    device = input.device or DeviceRef.CPU()

    promoted = [
        dtype_promotion._promote_to_strong(np.array([x]), DType.int64, device)
        for x in paddings
    ]

    padding_tensor = concat(promoted, axis=0)

    return Graph.current._add_op(
        rmo.mo_pad_constant,
        result=result_type.to_mlir(),
        input=TensorValue(input),
        paddings=padding_tensor,
        constant=dtype_promotion._promote_to_strong(value, input.dtype, device),
    )[0].tensor
