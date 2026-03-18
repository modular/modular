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

"""Device mesh for organizing devices into an N-dimensional logical grid.

A device mesh enables combining different parallelism strategies. For example,
a 2x4 mesh with `(Replicated(), Sharded(axis=0))` placement gives DP=2, TP=4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from max.driver import Device


@dataclass(frozen=True)
class DeviceMesh:
    """An N-dimensional logical grid of devices.

    Args:
        devices: Flat tuple of devices in row-major order.
        mesh_shape: Shape of the logical grid, e.g. `(2, 4)` for DP=2, TP=4.
        axis_names: Human-readable names for each axis, e.g. `("dp", "tp")`.
    """

    devices: tuple[Device, ...]
    mesh_shape: tuple[int, ...]
    axis_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.devices) == 0:
            raise ValueError("DeviceMesh requires at least one device.")
        expected = math.prod(self.mesh_shape)
        if len(self.devices) != expected:
            raise ValueError(
                f"mesh_shape {self.mesh_shape} expects {expected} devices, "
                f"got {len(self.devices)}"
            )
        if len(self.axis_names) != len(self.mesh_shape):
            raise ValueError(
                f"axis_names length ({len(self.axis_names)}) must match "
                f"mesh_shape rank ({len(self.mesh_shape)})"
            )
        if len(set(self.axis_names)) != len(self.axis_names):
            raise ValueError(
                f"axis_names must be unique, got {self.axis_names}"
            )

    @property
    def ndim(self) -> int:
        """Returns the number of mesh dimensions."""
        return len(self.mesh_shape)

    @property
    def num_devices(self) -> int:
        """Returns the total number of devices."""
        return len(self.devices)

    def axis_size(self, axis: str | int) -> int:
        """Returns the size of a mesh axis by name or index."""
        idx = self._resolve_axis(axis)
        return self.mesh_shape[idx]

    def _resolve_axis(self, axis: str | int) -> int:
        """Converts an axis name or index to a validated integer index."""
        if isinstance(axis, str):
            if axis not in self.axis_names:
                raise ValueError(
                    f"Unknown axis name {axis!r}. Available: {self.axis_names}"
                )
            return self.axis_names.index(axis)
        if axis < 0 or axis >= self.ndim:
            raise IndexError(
                f"Axis index {axis} out of range for mesh with "
                f"{self.ndim} dimensions"
            )
        return axis

    def __repr__(self) -> str:
        shape_str = ", ".join(
            f"{name}={size}"
            for name, size in zip(
                self.axis_names, self.mesh_shape, strict=False
            )
        )
        return f"DeviceMesh({shape_str})"
