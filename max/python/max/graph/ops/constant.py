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
"""Core graph primitives."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeAlias, cast

import numpy as np
from max.dtype import DType
from max.mlir.dialects import mo

from ..._core import graph as _graph
from ..._core.dialects import mo as _mo
from ...driver import CPU, Device, DLPackArray, Tensor
from ..graph import Graph
from ..type import DeviceRef, TensorType
from ..value import TensorValue

Number: TypeAlias = float | np.number[Any]
NestedArray: TypeAlias = Sequence["Number | NestedArray"]


def shape(literal: NestedArray | Number) -> tuple[int, ...]:
    if not isinstance(literal, Sequence):
        return ()
    outer = len(literal)
    inners: set[tuple[int, ...]] = {shape(inner) for inner in literal}
    if len(inners) > 1:
        raise ValueError(f"Array literals must be rectangular, got {literal=}")
    return outer, *next(iter(inners), ())


def index(literal: NestedArray | Number, idx: Sequence[int]) -> Number:
    if not idx:
        assert not isinstance(literal, Sequence)
        return cast(Number, literal)
    first, *rest = idx
    assert isinstance(literal, Sequence)
    return index(literal[first], rest)


def constant(
    value: DLPackArray | NestedArray | Number,
    dtype: DType | None = None,
    device: Device | DeviceRef | None = None,
) -> TensorValue:
    """Adds a node representing a constant operation.

    The value of this constant will have the type `TensorType` with the
    same shape as `value`. If `value` is a scalar type, it will create a `TensorType` with 0 dimensions.

    The constant will be loaded with the specified dtype.
    If the constant does not fit within the specified dtype, an error is raised.

    Warning: Loading the constant could result in precision loss.
    For example, loading `16777217` as a `float32` will result in `16777216.0`.

    Args:
        value: The constant's value.
        dtype: The constant tensor's element type.
        device: The device the constant lives on.

    Returns:
        A graph value containing the constant data as an attribute.
    """

    if not isinstance(value, DLPackArray):
        if dtype is None or device is None:
            raise TypeError(
                "Literal constants must explicitly set a dtype and device."
            )

        min, max = _DTYPE_MIN_AND_MAX[dtype]
        tensor = Tensor(dtype, shape(value), device=CPU())
        for idx in tensor._iterate_indices():
            v = index(value, idx)

            if not dtype.is_float() and not min <= int(v) <= max:
                raise ValueError(
                    "Unsafe cast: Refusing to implicitly promote external "
                    f"array with value {v} out of range for DType {dtype}."
                )

            tensor[idx] = v

        value = tensor
    elif isinstance(value, np.ndarray):
        value = np.ascontiguousarray(value)

    value = Tensor.from_dlpack(value)
    device = DeviceRef.from_device(device or value.device)
    dtype = dtype or value.dtype
    if dtype != value.dtype:
        raise ValueError(
            f"DType must match input dtype: {dtype=} != {value.dtype=}"
        )

    type = TensorType(dtype, value.shape, device=device)
    attr = _graph.array_attr(value, type.to_mlir())
    return Graph.current._add_op_generated(_mo.ConstantOp, type, attr)[0].tensor


def constant_external(name: str, type: TensorType) -> TensorValue:
    """Registers an external constant (weight) in the graph of a given type.

    Two external constants with the same name and type refer to the same weight.

    Two external constants with the same name and different types are
    incompatible and will fail compilation.

    Args:
        name: The name of the external constant.
            This should be the fully-qualified weight name and must be unique.
        type: The type of the constant value.
    Returns:
        A tensor value of the specified type, representing the weight value
        associated with the name at compile time.
    """
    return Graph.current._add_op(
        mo.constant_external,
        result=type,
        name=name,
        device=type.device.to_mlir(),
        align=type.dtype.align,
    )[0]


# For each DType, this is the full range of representable values.
# Since constant and scalar have explicit users dtypes, we trust that the specified dtype is wanted.
# We still error is a value does not fit in these ranges.
_DTYPE_MIN_AND_MAX = {
    DType.bool: (0, 1),
    DType.int8: (-(2**7), 2**7 - 1),
    DType.int16: (-(2**15), 2**15 - 1),
    DType.int32: (-(2**31), 2**31 - 1),
    DType.int64: (-(2**63), 2**63 - 1),
    DType.uint8: (0, 2**8 - 1),
    DType.uint16: (0, 2**16 - 1),
    DType.uint32: (0, 2**32 - 1),
    DType.uint64: (0, 2**64 - 1),
    DType.float8_e5m2: (float("-inf"), float("inf")),
    DType.float8_e5m2fnuz: (-57344, 57344),
    DType.float8_e4m3fn: (-448, 448),
    DType.float8_e4m3fnuz: (-240, 240),
    DType.bfloat16: (float("-inf"), float("inf")),
    DType.float16: (float("-inf"), float("inf")),
    DType.float32: (float("-inf"), float("inf")),
    DType.float64: (float("-inf"), float("inf")),
}
