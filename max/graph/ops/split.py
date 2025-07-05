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
"""Op implementation for split."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from max.dtype import DType
from max.mlir.dialects import mo

from ..dim import Dim, DimLike
from ..graph import Graph
from ..shape import Shape
from ..type import DeviceRef
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
        axis: Dimension to split the input tensor. Must have a statically
            known dimension size.

    Returns:
        A list of tensors with the same length as `split_sizes`, where each
        tensor has the same shape as the input except along the split dimension
        `axis`, where the size is given by the corresponding element in
        `split_sizes`.
    """
    x = TensorValue(x)
    split_sizes = [Dim(size) for size in split_sizes]

    if not (-x.rank <= axis < x.rank):
        raise IndexError(f"Axis out of range {axis=}, {x.rank=}")

    if axis < 0:
        axis += x.rank

    if sum(int(size) for size in split_sizes) != x.shape[axis]:
        raise ValueError(
            "Split sizes must sum to dimension value; "
            f"{x.shape[axis]=} != sum({split_sizes=})"
        )

    if any(int(size) < 0 for size in split_sizes):
        raise ValueError(f"Split sizes must be positive: {split_sizes=}")

    def split_type(dim: Dim):
        shape = Shape(x.shape)
        shape[axis] = dim
        return TensorType(x.dtype, shape, x.device)

    result_types = [split_type(Dim(size)) for size in split_sizes]

    if not split_sizes:
        return []  # op will assert on empty splits

    outputs = Graph.current._add_op(
        mo.split,
        result_types,
        x,
        constant(np.array(split_sizes), DType.int64, DeviceRef.CPU()),
        constant(axis, DType.int64, DeviceRef.CPU()),
    )
    return [out.tensor for out in outputs]
