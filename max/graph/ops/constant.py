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

from typing import Any

import numpy as np
import numpy.typing as npt
from max._core import graph as _graph
from max.dtype import DType
from max.mlir.dialects import mo

from ...driver import Device
from ..graph import Graph
from ..type import DeviceRef, TensorType
from ..value import TensorValue


def constant(
    value: npt.NDArray[np.number[Any]]
    | int
    | float
    | np.integer[Any]
    | np.floating[Any],
    dtype: DType,
    device: Device | DeviceRef,
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
    device = DeviceRef.from_device(device)
    if not isinstance(value, (np.ndarray, int, float, np.integer, np.floating)):
        raise TypeError(
            "ops.constant expects inputs to numpy array, int, float,"
            f" np.integer, or np.floating, but got '{type(value).__name__}'."
        )

    if isinstance(value, (int, float, np.integer, np.floating)):
        value = np.array(value)

    device = device or DeviceRef.CPU()

    if dtype == DType.bfloat16:
        # Numpy can't natively generate in bf16.
        # Generate in f32 and cast to bf16.
        return constant(value, DType.float32, device).cast(DType.bfloat16)
    elif dtype in (
        DType.float8_e4m3fn,
        DType.float8_e4m3fnuz,
        DType.float8_e5m2,
        DType.float8_e5m2fnuz,
    ):
        # Numpy can't natively generate in these types.
        # Generate in f32 and cast to these types.
        return constant(value, DType.float32, device).cast(dtype)

    if not dtype.is_float():
        min, max = _DTYPE_MIN_AND_MAX[dtype]
        if not (np.all(min <= value) and np.all(value <= max)):
            raise ValueError(
                f"Unsafe cast: Can't cast numpy array with value ({value}) to"
                f" dtype({dtype}). Values out of range of dtype."
            )
    tensor_type = TensorType(dtype, value.shape, device).to_mlir()
    array_attr = _graph.array_attr(
        "value",
        np.ascontiguousarray(value.astype(dtype.to_numpy())),
        tensor_type,
    )
    return Graph.current._add_op(
        mo.constant, result=tensor_type, value=array_attr
    )[0].tensor


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
