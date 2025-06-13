# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for hann_window."""

from __future__ import annotations

import numpy as np
from max.dtype import DType

from ..type import DeviceRef
from ..value import TensorValue
from .constant import constant
from .elementwise import cos
from .range import range


def hann_window(
    window_length: int,
    device: DeviceRef,
    periodic: bool = True,
    dtype: DType = DType.float32,
) -> TensorValue:
    """Calculate a Hann window for a given length.

    Hann window function:

    .. math::

        H[n] = 1/2 [1 - cos(2 * pi * n / (N - 1))]

    where N is window_length.

    Args:
        window_length: The length of the window.
        device: The device to run the operation on.
        periodic: bool
            flag determines whether the returned window trims off the last
            duplicate value from the symmetric window and is ready to be used
            as a periodic window with functions like stft().
            hann_window(L, periodic=True) == hann_window(L + 1, periodic=False)[:-1])
        dtype: The desired data type of the output tensor.

    Returns:
        A 1-D tensor of size (window_length,) containing the window.

    Raises:
        ValueError: If window_length is negative.
        TypeError: If window_length is not an integer.
    """
    if not isinstance(window_length, int):
        raise TypeError(
            f"window_length must be an integer, got {type(window_length).__name__}"
        )
    if window_length < 0:
        raise ValueError("window_length must be non-negative")
    if window_length == 0:
        return constant(np.array([], dtype=np.float32), dtype, device)
    elif window_length == 1:
        return constant(np.array([1.0], dtype=np.float32), dtype, device)

    if periodic:
        window_length += 1

    window = range(
        0, window_length, 1, out_dim=window_length, device=device, dtype=dtype
    )
    window = window * (2.0 * np.pi / np.float64(window_length - 1))
    window = cos(window) * (-0.5) + 0.5

    if periodic:
        window = window[:-1]  # Drop the last point for periodic windows

    return window
