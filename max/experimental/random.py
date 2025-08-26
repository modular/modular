# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from ..driver import Device
from ..dtype import DType
from ..graph import DeviceRef, ShapeLike, ops
from .functional import functional
from .tensor import TensorType, defaults

uniform_like = functional(ops.random.uniform)
gaussian_like = functional(ops.random.gaussian)
normal_like = gaussian_like


def uniform(
    shape: ShapeLike = (),
    range: tuple[float, float] = (0, 1),
    *,
    dtype: DType | None = None,
    device: Device | None = None,
):
    """Creates a tensor filled with random values from a uniform distribution.

    Args:
        shape: The shape of the output tensor. Defaults to scalar (empty tuple).
        range: A tuple specifying the (min, max) bounds of the uniform
            distribution. Defaults to (0, 1).
        dtype: The data type of the output tensor.
            If None, use the default dtype for the device.
        device: The device where the tensor will be allocated.
            If None, use the default device.

    Returns:
        A tensor with random values from the uniform distribution.

    Example:
        >>> from max.experimental import random
        >>> from max.dtype import DType
        >>> from max.driver import CPU
        >>>
        >>> # Generate 2x3 tensor with values between 0 and 1
        >>> tensor = random.uniform((2, 3), dtype=DType.float32, device=CPU())
    """
    dtype, device = defaults(dtype, device)
    type = TensorType(dtype, shape, device=DeviceRef.from_device(device))
    return uniform_like(type, range=range)


def gaussian(
    shape: ShapeLike = (),
    mean: float = 0.0,
    std: float = 1.0,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
):
    """Creates a tensor filled with random values from a Gaussian (normal) distribution.

    Args:
        shape: The shape of the output tensor. Defaults to scalar (empty tuple).
        mean: The mean (center) of the Gaussian distribution. Defaults to 0.0.
        std: The standard deviation (spread) of the Gaussian distribution.
            Must be positive. Defaults to 1.0.
        dtype: The data type of the output tensor.
            If None, use the default dtype for the device.
        device: The device where the tensor will be allocated.
            If None, use the default device.

    Returns:
        A tensor with random values from the Gaussian distribution.

    Example:
        >>> from max.experimental import random
        >>> from max.dtype import DType
        >>> from max.driver import CPU
        >>>
        >>> # Generate 2x3 tensor with standard normal distribution (mean=0, std=1)
        >>> tensor = random.gaussian((2, 3), dtype=DType.float32, device=CPU())
    """
    dtype, device = defaults(dtype, device)
    type = TensorType(dtype, shape, device=DeviceRef.from_device(device))
    return gaussian_like(type, mean=mean, std=std)


normal = gaussian
