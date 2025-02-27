# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import Any, Sequence, Union, overload

import numpy as np
from max._core.dtype import DType

_IdxElType = Union[int, slice]
IndexType = Union[Sequence[_IdxElType], _IdxElType]
ShapeType = Sequence[int]

class Device:
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    @property
    def stats(self) -> dict[str, Any]:
        """Returns utilization data for the device.

        .. code-block:: python

            from max import driver

            device = driver.CPU()
            device.stats
        """
    @property
    def label(self) -> str:
        """Returns device label.

        Possible values are:
        - "cpu" for host devices
        - "gpu" for accelerators

        .. code-block:: python

            from max import driver

            device = driver.CPU()
            device.label
        """
    @property
    def api(self) -> str:
        """Returns the API used to program the device.

        Possible values are:
        - "cpu" for host devices
        - "cuda" for NVIDIA GPUs
        - "hip" for AMD GPUs

        .. code-block:: python

            from max import driver

            device = driver.CPU()
            device.api
        """
    @property
    def id(self) -> int:
        """Returns a zero-based device id. For a CPU device this is the numa id.
        For GPU accelerators this is the id of the device relative to this host.
        Along with the `label`, an id can uniquely identify a device,
        e.g. "gpu:0", "gpu:1".

        .. code-block:: python

            from max import driver

            device = driver.CPU()
            device.id
        """
    @property
    def is_host(self) -> bool:
        """Whether this device is the CPU (host) device.

        .. code-block:: python

            from max import driver

            device = driver.CPU()
            device.is_host
        """
    @property
    def is_compatible(self) -> bool:
        """Returns whether this device is compatible with MAX."""
    def synchronize(self) -> None:
        """Ensures all operations on this device complete before returning.

        Raises:
            ValueError: If any enqueue'd operations had an internal error.
        """
    @staticmethod
    def cpu(id: int = -1) -> Device:
        """Creates a CPU device with the provided numa id."""
    @staticmethod
    def accelerator(id: int = -1) -> Device:
        """Creates an accelerator (e.g. GPU) device with the provided id."""

class Tensor:
    @property
    def shape(self) -> ShapeType: ...
    @property
    def rank(self) -> int: ...
    @property
    def is_contiguous(self) -> bool: ...
    @property
    def is_host(self) -> bool: ...
    @property
    def num_elements(self) -> int: ...
    @overload
    def __init__(
        self,
        shape: ShapeType,
        dtype: DType,
        device: Device,
    ) -> None: ...
    @overload
    def __init__(self, shape: np.ndarray, device: Device) -> None: ...
    def view(self, shape: ShapeType, dtype: DType) -> Tensor: ...
    def set(self, index: IndexType, value: Any) -> None: ...
    def get(self, index: IndexType) -> Tensor: ...
    def item(self) -> Any: ...
    def copy_to(self, device: Device) -> Tensor: ...
    def zeros(self) -> None: ...
    @staticmethod
    def from_dlpack(arg0: Any) -> Tensor: ...
    @property
    def device(self) -> Device: ...
    @property
    def dtype(self) -> DType: ...
    def __dlpack__(
        self, *, stream: Any | None = None, _mmap: Any | None = None
    ) -> Any: ...
    def __dlpack_device__(self) -> tuple[Any, Any]: ...

def cpu_device(device_id: int = -1) -> Device: ...
def accelerator(device_id: int = -1) -> Device: ...
def accelerator_count() -> int:
    """Returns number of accelerator devices available."""

__version__: str = ...
