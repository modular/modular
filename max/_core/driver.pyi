# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, overload

import max._core
import typing_extensions
from numpy.typing import ArrayLike

class Accelerator(Device):
    def __init__(self, id: int = -1) -> None:
        """
        Creates an accelerator device with the specified ID.

        Provides access to GPU or other hardware accelerators in the system.

        .. code-block:: python

            from max import driver
            # Create default accelerator (usually first available GPU)
            device = driver.Accelerator()
            # Or specify GPU id
            device = driver.Accelerator(id=0)  # First GPU
            device = driver.Accelerator(id=1)  # Second GPU
            # Get device id
            device_id = device.id
        """

class CPU(Device):
    def __init__(self, id: int = -1) -> None:
        """
        Creates a CPU device.

        .. code-block:: python

            from max import driver
            # Create default CPU device
            device = driver.CPU()
            # Device id is always 0 for CPU devices
            device_id = device.id
        """

class Device:
    def can_access(self, other: Device) -> bool:
        """
        Checks if this device can directly access memory of another device.

        Args:
            other (Device): The other device to check peer access against.

        Returns:
            bool: True if peer access is possible, False otherwise.

        Example:
            .. code-block:: python

                from max import driver

                gpu0 = driver.Accelerator(id=0)
                gpu1 = driver.Accelerator(id=1)

                if gpu0.can_access(gpu1):
                    print("GPU0 can directly access GPU1 memory.")
        """

    def synchronize(self) -> None:
        """
        Ensures all operations on this device complete before returning.

        Raises:
            ValueError: If any enqueue'd operations had an internal error.
        """

    @property
    def is_host(self) -> bool:
        """
        Whether this device is the CPU (host) device.

        .. code-block:: python

            from max import driver

            device = driver.CPU()
            device.is_host
        """

    @property
    def stats(self) -> Mapping[str, Any]:
        """
        Returns utilization data for the device.

        .. code-block:: python

            from max import driver

            device = driver.CPU()
            device.stats
        """

    @property
    def label(self) -> str:
        """
        Returns device label.

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
        """
        Returns the API used to program the device.

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
        """
        Returns a zero-based device id. For a CPU device this is always 0.
        For GPU accelerators this is the id of the device relative to this host.
        Along with the `label`, an id can uniquely identify a device,
        e.g. "gpu:0", "gpu:1".

        .. code-block:: python

            from max import driver

            device = driver.Accelerator()
            device.id
        """

    @property
    def is_compatible(self) -> bool:
        """Returns whether this device is compatible with MAX."""

    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __eq__(self, arg: object, /) -> bool: ...
    @staticmethod
    def cpu(id: int = -1) -> CPU:
        """Creates a CPU device. The id is ignored currently."""

class Tensor:
    @overload
    def __init__(
        self, shape: Sequence[int], dtype: max._core.dtype.DType, device: Device
    ) -> None: ...
    @overload
    def __init__(
        self, shape: Annotated[ArrayLike, dict(writable=False)], device: Device
    ) -> None: ...
    @property
    def dtype(self) -> max._core.dtype.DType: ...
    @property
    def shape(self) -> tuple: ...
    @property
    def rank(self) -> int: ...
    @property
    def num_elements(self) -> int: ...
    @property
    def device(self) -> Device: ...
    @property
    def is_contiguous(self) -> bool: ...
    @property
    def is_host(self) -> bool: ...
    def view(
        self, shape: Sequence[int], dtype: max._core.dtype.DType
    ) -> Tensor: ...
    def set(self, index: object, value: object) -> None: ...
    def get(self, index: object) -> Tensor: ...
    def item(self) -> object: ...
    def copy_to(self, device: Device) -> Tensor: ...
    def zeros(self) -> None: ...
    def _aligned(self, alignment: int) -> bool: ...
    def __dlpack_device__(self) -> tuple: ...
    def __dlpack__(
        self, *, stream: object | None = None, _mmap: object | None = None
    ) -> typing_extensions.CapsuleType: ...
    @staticmethod
    def from_dlpack(arg: object, /) -> Tensor: ...

def accelerator_count() -> int:
    """Returns number of accelerator devices available."""
