# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

"""Provides random tensor generation utilities.

This module provides functions for generating random tensors with various
distributions. All functions support specifying data type and device,
with sensible defaults based on the target device.

You can generate random tensors using different distributions:

.. code-block:: python

    from max.experimental import random
    from max.dtype import DType
    from max.driver import CPU

    # Generate 2x3 tensor with values between 0 and 1
    tensor1 = random.uniform((2, 3), dtype=DType.float32, device=CPU())

    tensor2 = random.uniform((4, 4), range=(0, 1), dtype=DType.float32, device=CPU())
"""

from __future__ import annotations

from max.driver import Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.realization_context import seed, set_seed
from max.experimental.sharding import DeviceMapping, DeviceMesh, Partial
from max.experimental.tensor import Tensor, defaults
from max.graph import ShapeLike

__all__ = ["gaussian", "normal", "seed", "set_seed", "uniform"]


def _resolve_tensor_placement(
    dtype: DType | None, device: Device | DeviceMesh | DeviceMapping | None
) -> tuple[DType, DeviceMesh, DeviceMapping | None]:
    """Splits ``device`` into the mesh to generate on and the mapping to end in."""
    if not isinstance(device, DeviceMapping):
        dtype, mesh = defaults(dtype, device)
        return dtype, mesh, None
    if any(isinstance(p, Partial) for p in device.placements):
        raise ValueError(
            f"random values cannot be created with a Partial placement: {device}"
        )
    dtype, mesh = defaults(dtype, device.mesh)
    return dtype, mesh, device


def _place_tensor(tensor: Tensor, mapping: DeviceMapping | None) -> Tensor:
    return tensor if mapping is None else tensor.to(mapping)


def uniform(  # noqa: ANN201
    shape: ShapeLike = (),
    range: tuple[float, float] = (0, 1),
    *,
    dtype: DType | None = None,
    device: Device | DeviceMesh | DeviceMapping | None = None,
):
    """Creates a tensor filled with random values from a uniform distribution.

    Generates a tensor with values uniformly distributed between the specified
    minimum and maximum bounds. This is useful for initializing weights,
    generating random inputs, or creating noise.

    Create tensors with uniform random values::

        from max.experimental import random
        from max.dtype import DType
        from max.driver import CPU

        # Generate 2x3 tensor with values between 0 and 1
        tensor1 = random.uniform((2, 3), dtype=DType.float32, device=CPU())

        tensor2 = random.uniform((4, 4), range=(0, 1), dtype=DType.float32, device=CPU())

    Args:
        shape: The shape of the output tensor. Defaults to scalar (empty tuple).
        range: A tuple specifying the (min, max) bounds of the uniform
            distribution. The minimum value is inclusive, the maximum value
            is exclusive. Defaults to ``(0, 1)``.
        dtype: The data type of the output tensor. If ``None``, uses the
            default dtype for the specified device (float32 for CPU,
            bfloat16 for accelerators). Defaults to ``None``.
        device: The device where the tensor will be allocated. If ``None``,
            uses the default device (accelerator if available, otherwise CPU).
            Defaults to ``None``.

    Returns:
        A :class:`~max.experimental.tensor.Tensor` with random values sampled from
        the uniform distribution.

    Raises:
        ValueError: If the range tuple does not contain exactly two values
            or if min >= max, or if ``device`` is a ``DeviceMapping`` with a
            ``Partial`` placement.
    """
    dtype, mesh, mapping = _resolve_tensor_placement(dtype, device)
    return _place_tensor(
        F.uniform(shape, range, dtype=dtype, device=mesh), mapping
    )


def gaussian(  # noqa: ANN201
    shape: ShapeLike = (),
    mean: float = 0.0,
    std: float = 1.0,
    *,
    dtype: DType | None = None,
    device: Device | DeviceMesh | DeviceMapping | None = None,
):
    """Creates a tensor filled with random values from a Gaussian (normal) distribution.

    Generates a tensor with values sampled from a normal (Gaussian) distribution
    with the specified mean and standard deviation. This is commonly used for
    weight initialization using techniques like Xavier/Glorot or He initialization.

    Create tensors with random values from a Gaussian distribution::

        from max.experimental import random
        from max.driver import CPU
        from max.dtype import DType

        # Standard normal distribution
        tensor = random.gaussian((2, 3), dtype=DType.float32, device=CPU())

    Args:
        shape: The shape of the output tensor. Defaults to scalar (empty tuple).
        mean: The mean (center) of the Gaussian distribution. This determines
            where the distribution is centered. Defaults to ``0.0``.
        std: The standard deviation (spread) of the Gaussian distribution.
            Must be positive. Larger values create more spread in the distribution.
            Defaults to ``1.0``.
        dtype: The data type of the output tensor. If ``None``, uses the
            default dtype for the specified device (float32 for CPU,
            bfloat16 for accelerators). Defaults to ``None``.
        device: The device where the tensor will be allocated. If ``None``,
            uses the default device (accelerator if available, otherwise CPU).
            Defaults to ``None``.

    Returns:
        A :class:`~max.experimental.tensor.Tensor` with random values sampled from
        the Gaussian distribution.

    Raises:
        ValueError: If std <= 0, or if ``device`` is a ``DeviceMapping`` with
        ``Partial`` placement.
    """
    dtype, mesh, mapping = _resolve_tensor_placement(dtype, device)
    return _place_tensor(
        F.gaussian(shape, mean, std, dtype=dtype, device=mesh), mapping
    )


#: Alias for :func:`gaussian`.
#: Creates a tensor with values from a normal (Gaussian) distribution.
normal = gaussian
