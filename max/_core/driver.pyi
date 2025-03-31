# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

from collections.abc import Generator, Mapping, Sequence
from typing import Annotated, Any, overload

import max._core
import numpy
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
    """
    Device-resident tensor representation. Allocates memory onto a given device
    with the provided shape and dtype. Tensors can be sliced to provide strided
    views of the underlying memory, but any tensors input into model execution
    must be contiguous. Does not currently support setting items across multiple
    indices, but does support numpy-style slicing.

    :param dtype: DType of tensor
    :param shape: Tuple of positive, non-zero integers denoting the tensor shape.
    :param device: Device to allocate tensor onto.
    """

    @overload
    def __init__(
        self,
        shape: Sequence[int],
        dtype: max._core.dtype.DType,
        device: Device | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self, shape: Annotated[ArrayLike, dict(writable=False)], device: Device
    ) -> None: ...
    @overload
    def __init__(self, other: Tensor) -> None:
        """
        Moves the internals from an existing Tensor object into a new Tensor object.

        Primarily used for initializing subclasses with existing Tensors.
        """

    @property
    def device(self) -> Device:
        """Device on which tensor is resident."""

    @property
    def dtype(self) -> max._core.dtype.DType:
        """DType of constituent elements in tensor."""

    @property
    def element_size(self) -> int:
        """Return the size of the element type in bytes."""

    @property
    def is_contiguous(self) -> bool:
        """
        Whether or not tensor is contiguously allocated in memory. Returns
        false if the tensor is a non-contiguous slice.

        Currently, we consider certain situations that are contiguous as
        non-contiguous for the purposes of our engine, such as when a tensor
        has negative steps.
        """

    @property
    def is_host(self) -> bool:
        """
        Whether or not tensor is host-resident. Returns false for GPU tensors,
        true for CPU tensors.

        .. code-block:: python

            from max import driver
            from max.dtype import DType

            cpu_tensor = driver.Tensor([2, 3], dtype=DType.bfloat16, device=driver.CPU())

            print(cpu_tensor.is_host)
        """

    @property
    def num_elements(self) -> int:
        """
        Returns the number of elements in this tensor.

        Rank-0 tensors have 1 element by convention.
        """

    @property
    def rank(self) -> int:
        """Tensor rank."""

    @property
    def shape(self) -> tuple:
        """Shape of tensor."""

    def contiguous(self) -> Tensor:
        """Creates a contiguous copy of the tensor."""

    def copy(self, device: Device | None = None) -> Tensor:
        """
        Create a deep copy on an optionally given device.

        If a device is None (default), a copy is created on the same device.

        .. code-block:: python

            from max import driver
            from max.dtype import DType

            cpu_tensor = driver.Tensor([2, 3], dtype=DType.bfloat16, device=driver.CPU())

            cpu_copy = cpu_tensor.copy()
        """

    def inplace_copy_from(self, src: Tensor) -> None:
        """
        Copy the contents of another tensor into this one. These tensors may
        be on different devices.
        Requires that both tensors are contiguous and have same size.
        """

    @staticmethod
    def from_dlpack(array: Any, *, copy: bool | None = None) -> Tensor:
        """
        Create a tensor from an object implementing the dlpack protocol.
        This usually does not result in a copy, and the producer of the object
        retains ownership of the underlying memory.
        """

    @staticmethod
    def from_numpy(arr: numpy.ndarray) -> Tensor:
        """
        Creates a tensor from a provided numpy array on the host device.
        The underlying data is not copied unless the array is noncontiguous. If
        it is, a contiguous copy will be returned.
        """

    def item(self) -> Any:
        """
        Returns the scalar value at a given location. Currently
        implemented only for zero-rank tensors. The return type is
        converted to a Python built-in type.
        """

    @staticmethod
    def scalar(
        value: Any, dtype: max._core.dtype.DType, device: Device | None = None
    ) -> Tensor:
        """
        Create a scalar value of a given dtype and value.

        If device is None (default), the tensor will be allocated on the CPU.
        """

    def to(self, device: Device) -> Tensor:
        """
        Return a tensor that's guaranteed to be on the given device.

        The tensor is only copied if the input device is different from the
        device upon which the tensor is already resident.
        """

    def to_numpy(self) -> numpy.ndarray:
        """
        Converts the tensor to a numpy array.

        If the tensor is on the host (CPU), the numpy array aliases the existing memory.
        Otherwise, it is copied to the host device.
        """

    def view(
        self, dtype: max._core.dtype.DType, shape: Sequence[int] | None = None
    ) -> Tensor:
        """
        Return a new tensor with the given type and shape that shares the
        underlying memory.
        If the shape is not given, it will be deduced if possible, or a
        ValueError is raised.
        """

    @staticmethod
    def zeros(
        shape: Sequence[int],
        dtype: max._core.dtype.DType,
        device: Device | None = None,
    ) -> Tensor:
        """Allocates a tensor with all elements initialized to zero."""

    def __dlpack__(
        self, *, stream: object | None = None, _mmap: object | None = None
    ) -> typing_extensions.CapsuleType:
        """Implements part of the dlpack contract."""

    def __dlpack_device__(self) -> tuple:
        """Implements part of the dlpack contract."""

    def __getitem__(self, idx: int | slice | Sequence[int | slice]) -> Tensor:
        """
        Gets a tensor slice. Supports full numpy-style slicing. Invocations
        using only integer-based indexes will return zero-rank tensors.
        """

    def __setitem__(
        self, idx: int | slice | Sequence[int | slice], value: Any
    ) -> None:
        """Sets an item in the tensor."""

    def _aligned(self, alignment: int | None = None) -> bool: ...
    def __aligned(self, alignment: int | None = None) -> bool: ...
    @staticmethod
    def _from_dlpack(arg: object, /) -> Tensor: ...
    def _iterate_indices(self) -> Generator[Sequence[int]]: ...
    def _view(
        self, dtype: max._core.dtype.DType, shape: Sequence[int]
    ) -> Tensor: ...
    def _inplace_copy_from(self, src: Tensor) -> None: ...

def accelerator_count() -> int:
    """Returns number of accelerator devices available."""
