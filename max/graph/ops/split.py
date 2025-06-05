# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for split."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from max.dtype import DType
from max.mlir.dialects import mo

from ..graph import Graph
from ..type import DeviceRef, Dim, DimLike
from ..value import TensorType, TensorValue, TensorValueLike
from .constant import constant


def split(
    x: TensorValueLike, split_sizes: Sequence[DimLike], axis: int = 0
) -> list[TensorValue]:
    """Splits the input tensor into multiple tensors along a given dimension.

    Args:
        x: The input symbolic tensor to split.
        split_sizes: Sizes of each output tensor. Must add up to the split
            dimension `axis`.
        axis: Dimension to split the input tensor.

    Returns:
        A list of tensors with the same length as `split_sizes`, where each
        tensor has the same shape as the input except along the split dimension
        `axis`, where the size is given by the corresponding element in
        `split_sizes`.
    """
    v = TensorValue(x)
    if not split_sizes:
        return [v]

    if not (-v.rank <= axis < v.rank):
        raise ValueError(
            f"Split axis must be within the input rank ({v.rank}), got {axis}"
        )
    elif axis < 0:
        axis += v.rank

    if sum(int(size) for size in split_sizes) != v.shape[axis]:
        raise ValueError(
            f"The split_sizes values should sum to {v.shape[axis]} (input tensor's size at dimension {axis}), but got split_sizes={split_sizes}"
        )

    result_types = []
    for size in split_sizes:
        new_shape = v.shape.copy()
        new_shape[axis] = Dim(size)
        result_types.append(TensorType(v.dtype, new_shape, v.device).to_mlir())
    outputs = Graph.current._add_op(
        mo.split,
        result_types,
        v,
        constant(np.array(split_sizes), DType.int64, DeviceRef.CPU()),
        constant(axis, DType.int64, DeviceRef.CPU()),
    )
    return [out.tensor for out in outputs]
