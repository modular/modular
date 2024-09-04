# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Core graph primitives."""
from typing import Union

import numpy as np
from max import _graph
from max.dtype import DType
from max.mlir.dialects import mo

from ..graph import Graph
from ..value import TensorValue
from ..type import TensorType


def constant(value: np.ndarray, dtype: DType) -> TensorValue:
    """Adds a node representing a constant operation.

    The value of this constant will have the type `TensorType` with the
    same shape and dtype as `value`.

    The constant will be loaded with the specified dtype.
    If the constant does not fit within the specified dtype, an error is raised.

    Warning: Loading the constant could result in precision loss.

    Args:
        value: The constant's value.
        dtype: The constant tensor's element type.

    Returns:
        A graph value containing the constant data as an attribute.
    """
    if isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(
            "ops.constant expects inputs to be numpy array, but got"
            f" '{type(value).__name__}'. Use ops.scalar for this instead."
        )
    if not isinstance(value, (np.ndarray)):
        raise TypeError(
            "ops.constant expects inputs to be numpy array, but got"
            f" '{type(value).__name__}'."
        )

    if dtype == DType.bfloat16:
        # Numpy can't natively generate in bf16.
        # Generate in f32 and cast to bf16.
        return constant(value, DType.float32).cast(DType.bfloat16)

    if not dtype.is_float():
        min, max = _DTYPE_MIN_AND_MAX[dtype]
        if not (np.all(min <= value) and np.all(value <= max)):
            raise ValueError(
                f"Unsafe cast: Can't cast numpy array with value ({value}) to"
                f" dtype({dtype}). Values out of range of dtype."
            )
    tensor_type = TensorType(dtype, value.shape).to_mlir()
    array_attr = _graph.array_attr(
        "value",
        np.ascontiguousarray(value.astype(dtype.to_numpy())),
        tensor_type,
    )
    return Graph.current._add_op(
        mo.constant, result=tensor_type, value=array_attr
    )[0].tensor


def scalar(
    value: Union[int, float, np.integer, np.floating],
    dtype: DType,
    rank: int = 0,
) -> TensorValue:
    """Creates a scalar in the graph and returns the value.

    A scalar is a tensor with a single element. Generally scalars are of rank 0, but that is configurable.

    The scalar will be loaded with the specified dtype.
    If the scalar does not fit within the specified dtype, an error is raised.

    Warning: Loading the scalar could result in precision loss.
    """
    if isinstance(value, (np.ndarray)):
        raise TypeError(
            "ops.scalar expects inputs to int, float, np.integer, or"
            " np.floating, but got numpy array. Use ops.constant for numpy"
            " arrays."
        )
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(
            "ops.scalar expects inputs to int, float, np.integer, or"
            f" np.floating, but got '{type(value).__name__}'."
        )

    if dtype == DType.bfloat16:
        # Numpy can't natively generate in bf16.
        # Generate in f32 and cast to bf16.
        return scalar(value, DType.float32, rank).cast(DType.bfloat16)

    if not dtype.is_float():
        min, max = _DTYPE_MIN_AND_MAX[dtype]
        if not (min <= value <= max):
            raise ValueError(
                f"Unsafe cast: Can't cast python scalar with value ({value}) to"
                f" dtype({dtype}). Value out of range of dtype."
            )

    tensor_type = TensorType(dtype, [1] * rank).to_mlir()
    array_attr = _graph.array_attr(
        "value", np.array([value]).astype(dtype.to_numpy()), tensor_type
    )
    return Graph.current._add_op(
        mo.constant, result=tensor_type, value=array_attr
    )[0].tensor


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
    DType.bfloat16: (float("-inf"), float("inf")),
    DType.float16: (float("-inf"), float("inf")),
    DType.float32: (float("-inf"), float("inf")),
    DType.float64: (float("-inf"), float("inf")),
}
