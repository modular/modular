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

"""Random tensor creation ops.

All functions accept ``device`` as either a single
:class:`~max.driver.Device` or a
:class:`~max.experimental.sharding.DeviceMapping`.  For multi-device
mappings, a full-sized random tensor is generated on the first device
and then distributed according to the placements.  This is not
optimal but guarantees correct numerical values from a single seed.
"""

from __future__ import annotations

from max.driver import Device
from max.dtype import DType
from max.experimental.realization_context import seed, set_seed
from max.experimental.sharding import DeviceMapping
from max.experimental.tensor import Tensor, TensorType, defaults
from max.graph import DeviceRef, ShapeLike, ops

from ._context_provider import functional
from .collectives import shard
from .creation import _normalize_device

__all__ = ["gaussian", "normal", "seed", "set_seed", "uniform"]


@functional(linear=None)
def uniform(
    shape: ShapeLike = (),
    range: tuple[float, float] = (0, 1),
    *,
    dtype: DType | None = None,
    device: Device | DeviceMapping | None = None,
) -> Tensor:
    """Creates a tensor filled with random values from a uniform distribution.

    Args:
        shape: The shape of the output tensor.
        range: A tuple specifying the (min, max) bounds. Defaults to ``(0, 1)``.
        dtype: The data type of the output tensor.
        device: A :class:`~max.driver.Device` or
            :class:`~max.experimental.sharding.DeviceMapping`.
    """
    mapping = _normalize_device(device)
    resolved_dtype, first_device = defaults(dtype, mapping.mesh.devices[0])
    tt = TensorType(resolved_dtype, shape, DeviceRef.from_device(first_device))
    tv = ops.random.uniform(tt, range=range)
    t = Tensor.from_graph_value(tv)
    return shard(t, mapping)


@functional(linear=None)
def gaussian(
    shape: ShapeLike = (),
    mean: float = 0.0,
    std: float = 1.0,
    *,
    dtype: DType | None = None,
    device: Device | DeviceMapping | None = None,
) -> Tensor:
    """Creates a tensor filled with random values from a Gaussian distribution.

    Args:
        shape: The shape of the output tensor.
        mean: The mean of the distribution. Defaults to ``0.0``.
        std: The standard deviation. Defaults to ``1.0``.
        dtype: The data type of the output tensor.
        device: A :class:`~max.driver.Device` or
            :class:`~max.experimental.sharding.DeviceMapping`.
    """
    mapping = _normalize_device(device)
    resolved_dtype, first_device = defaults(dtype, mapping.mesh.devices[0])
    tt = TensorType(resolved_dtype, shape, DeviceRef.from_device(first_device))
    tv = ops.random.gaussian(tt, mean=mean, std=std)
    t = Tensor.from_graph_value(tv)
    return shard(t, mapping)


#: Alias for :func:`gaussian`.
normal = gaussian


def uniform_like(
    like: TensorType,
    range: tuple[float, float] = (0, 1),
) -> Tensor:
    """Creates a random uniform tensor matching the given type."""
    return uniform(
        like.shape,
        range=range,
        device=like.device.to_device(),
        dtype=like.dtype,
    )


def gaussian_like(
    like: TensorType,
    mean: float = 0.0,
    std: float = 1.0,
) -> Tensor:
    """Creates a random Gaussian tensor matching the given type."""
    return gaussian(
        like.shape,
        mean=mean,
        std=std,
        device=like.device.to_device(),
        dtype=like.dtype,
    )


#: Alias for :func:`gaussian_like`.
normal_like = gaussian_like
