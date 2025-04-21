# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Handles DType promotion for TensorValues.

DType promotion in the graph api is used to decide the DType of the output of an operations with multiple inputs.
Not all operations have DType promotion, but most binary and mathematical ops have promotion.

DType promotion will always return one of the input dtypes.
This avoids accidentally over promoting and harming performance.

The target DType for promoting a value `x` and `y` is:
    `(max(category(x), category(y)), max(bitwidth(x), bitwidth(y))`

Where category is an ordered hierachy of: `bool < unsigned int < signed int < float`

If all input dtypes can be fully represented by the target dtype, the promotion is successful.
If an input can not be guaranteed representable (e.g. `uint8` -> `int8`), an error is raised.

DType promotion of a max object and a non-max object will only ever promote to the max object DType.
An error will be raised if the values in the non-max object are not precisely representable in the max object DType.
This means that `16777217` will raise an error if converted to float32 (where it would be represented as 16777216.0).
To convert a value while allowing for minor precision loss, `ops.constant` can be used.

If only non-max objects attempt promotion, it will always fail.
"""

import numpy as np
from max.dtype import DType

from . import ops
from .graph import DeviceRef
from .value import TensorValue, TensorValueLike, _strong_tensor_value_like


def _restrict_to_strong_dtypes(value: TensorValueLike) -> TensorValue:
    """Converts strong dtype values to TensorValue.

    Raise an error if the input dtype is weak.
    """
    if _is_strong(value):
        # Valid unary op with proper dtype.
        return TensorValue(value)
    else:
        # TODO: Maybe special case numpy array with non int64/float64 input.
        # Theoretically, that is an explicitly set dtype.

        # Non-max object with unary op.
        # This often is a bug and leads to overpromotion.
        raise TypeError(
            "Unary ops do not support non-max objects as input. Non-max"
            " objects tend to overpromote the dtype, leading to significant"
            " loss of performance. Please explicitly convert the input to a"
            " graph.Value. This can be done with ops.constant."
        )


def _promote_weak_dtypes(
    x: TensorValueLike, y: TensorValueLike
) -> tuple[TensorValue, TensorValue]:
    """Promote weak dtypes and handle device placement.

    Most of dtype promotion is dealt with in RMO.
    This function specifically deals with promotion of non-max objects.
    All non-max objects have a weak dtype and will promote to a max object dtype.
    That said, we will always scan the non-max object to ensure it is representable in the max object dtype.
    Finally, if a mix of weak and strong types are given, place weak types
    on the strong type's device.
    """

    x_was_strong = _is_strong(x)
    y_was_strong = _is_strong(y)

    if x_was_strong and y_was_strong:
        return (TensorValue(x), TensorValue(y))

    if not x_was_strong and not y_was_strong:
        raise TypeError(
            "Binary ops require at least one max object as input. Non-max"
            " objects tend to overpromote the dtype, leading to significant"
            " loss of performance. Please explicitly convert at least one"
            " input to a graph.Value. This can be done with ops.constant."
        )

    if x_was_strong:
        max_value = TensorValue(x)
        # TODO(GEX-2125): remove `or DeviceRef.CPU()` if once they are a non-optional field
        return (
            max_value,
            _promote_to_strong(
                y, max_value.dtype, max_value.device or DeviceRef.CPU()
            ),
        )
    else:
        max_value = TensorValue(y)
        return (
            _promote_to_strong(
                x, max_value.dtype, max_value.device or DeviceRef.CPU()
            ),
            max_value,
        )


def _promote_to_strong(
    value: TensorValueLike, strong_dtype: DType, device: DeviceRef
) -> TensorValue:
    """Promotes weak dtypes and handle device placement.

    If the the input value is already strong, its dtype will not be changed.
    Instead, strong dtype promotion will be handled by the individual ops in RMO.
    """
    if _is_strong(value):
        return TensorValue(value)
    elif isinstance(value, (int, np.integer)):
        min, max = _DTYPE_MIN_AND_MAX_FULL_PRECISION[strong_dtype]
        if min <= value <= max:
            return ops.constant(value, strong_dtype, device)

        raise ValueError(
            f"Unsafe cast: Can't promote python int with value ({value}) to"
            f" dtype({strong_dtype}). It would lose precision."
        )

    elif isinstance(value, (float, np.floating)):
        if strong_dtype.is_float():
            return ops.constant(value, strong_dtype, device)

        raise ValueError(
            f"Unsafe cast: Can't promote python float to dtype({strong_dtype})."
        )

    elif isinstance(value, (np.ndarray)):
        # In numpy 2.0, booleans compare as signed, which causes issues when doing
        # comparisons such as `np.array(False) < 2**64-1`, it will fail since it
        # treats the right value as signed, which fails to cast to a C long.
        # So we just cast to an unsigned type to make sure we can always do a comparison.
        if value.dtype == np.bool_:
            value = value.astype(np.uint8)
        elif DType.from_numpy(value.dtype).is_float():
            if strong_dtype.is_float():
                return ops.constant(value, strong_dtype, device)
            else:
                raise ValueError(
                    "Unsafe cast: Can't promote numpy float array to"
                    f" dtype({strong_dtype})."
                )

        min, max = _DTYPE_MIN_AND_MAX_FULL_PRECISION[strong_dtype]
        if np.all(min <= value) and np.all(value <= max):
            return ops.constant(value, strong_dtype, device)

        raise ValueError(
            "Unsafe cast: Can't promote numpy integer array with value"
            f" ({value}) to dtype({strong_dtype}). It would lose precision."
        )

    else:
        raise TypeError(
            "_promote_weak_dtypes() argument must be a TensorValueLike, not"
            f" '{type(value).__name__}'"
        )


def _is_strong(value: TensorValueLike) -> bool:
    return isinstance(value, _strong_tensor_value_like)


# For each DType, this is the range of values where a conversion would not lose precision.
# This is used for conversions from python/numpy int to said DType.
_DTYPE_MIN_AND_MAX_FULL_PRECISION = {
    DType.bool: (0, 1),
    DType.int8: (-(2**7), 2**7 - 1),
    DType.int16: (-(2**15), 2**15 - 1),
    DType.int32: (-(2**31), 2**31 - 1),
    DType.int64: (-(2**63), 2**63 - 1),
    DType.uint8: (0, 2**8 - 1),
    DType.uint16: (0, 2**16 - 1),
    DType.uint32: (0, 2**32 - 1),
    DType.uint64: (0, 2**64 - 1),
    DType.float8_e5m2: (-(1.10 * 2**16), 1.10 * 2**16),
    DType.float8_e5m2fnuz: (-(1.75 * 2**15), 1.75 * 2**15),
    DType.float8_e4m3fn: (-(1.75 * 2**8), 1.75 * 2**8),
    DType.float8_e4m3fnuz: (-240, 240),
    # This is two to the power of the number of significand bits plus one.
    DType.bfloat16: (-(2**8), 2**8),
    DType.float16: (-(2**11), 2**11),
    DType.float32: (-(2**24), 2**24),
    DType.float64: (-(2**53), 2**53),
}
