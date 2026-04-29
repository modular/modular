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

"""Device mesh for organizing devices into an N-dimensional logical grid."""

from __future__ import annotations

import math
from dataclasses import dataclass

from max.driver import CPU, Device


@dataclass(frozen=True)
class DeviceMesh:
    """An N-dimensional logical grid of devices.

    Args:
        devices: A flat tuple of devices in row-major order.
        mesh_shape: The shape of the logical grid (for example, ``(2, 4)``
            for DP=2, TP=4).
        axis_names: The human-readable names for each axis (for example,
            ``("dp", "tp")``).
    """

    devices: tuple[Device, ...]
    """The devices in the mesh, in row-major order."""

    mesh_shape: tuple[int, ...]
    """The shape of the logical grid."""

    axis_names: tuple[str, ...]
    """The human-readable name for each mesh axis."""

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
        # Device homogeneity: either all devices are the same physical
        # device (simulated multi-device on one CPU/GPU) or all devices
        # are distinct (real multi-device).  Mixed meshes (some repeated,
        # some different) are not supported.
        if len(self.devices) > 1:
            unique = set(self.devices)
            n_unique = len(unique)
            if n_unique != 1 and n_unique != len(self.devices):
                raise ValueError(
                    f"DeviceMesh requires either all-same devices "
                    f"(simulated) or all-distinct devices (multi-device), "
                    f"but got {n_unique} unique devices for "
                    f"{len(self.devices)} slots."
                )

    @property
    def ndim(self) -> int:
        """The number of mesh dimensions."""
        return len(self.mesh_shape)

    @property
    def num_devices(self) -> int:
        """The total number of devices."""
        return len(self.devices)

    def axis_size(self, axis: str | int) -> int:
        """Returns the size of a mesh axis by name or index.

        Args:
            axis: The mesh axis to look up, either by name or by integer
                index.

        Returns:
            The number of devices along the requested axis.

        Raises:
            ValueError: If ``axis`` is a name that doesn't exist on the mesh.
            IndexError: If ``axis`` is an integer outside ``[0, ndim)``.
        """
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

    @staticmethod
    def single(device: Device) -> DeviceMesh:
        """Creates a trivial single-device mesh.

        Args:
            device: The single device the mesh wraps.
        """
        return DeviceMesh(devices=(device,), mesh_shape=(1,), axis_names=("_",))

    @staticmethod
    def default() -> DeviceMesh:
        """Returns a single-device mesh on the default device (CPU)."""
        return DeviceMesh.single(CPU())

    @property
    def is_single(self) -> bool:
        """Returns ``True`` if this mesh contains exactly one device."""
        return self.num_devices == 1

    @property
    def is_simulated(self) -> bool:
        """Returns ``True`` if all mesh slots reference the same device.

        A simulated mesh uses graph-level ops (add, concat, split) to
        emulate multi-device collectives on a single CPU or GPU.
        """
        return self.num_devices > 1 and len(set(self.devices)) == 1
