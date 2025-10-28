# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

import os
from collections.abc import Generator, Mapping, Sequence
from typing import Annotated, Any, overload

import max._core.dtype
import numpy
import typing_extensions
from numpy.typing import ArrayLike

class Device:
    def can_access(self, other: Device) -> bool:
        """
        Checks if this device can directly access memory of another device.

        .. code-block:: python

             from max import driver

             gpu0 = driver.Accelerator(id=0)
             gpu1 = driver.Accelerator(id=1)

             if gpu0.can_access(gpu1):
                 print("GPU0 can directly access GPU1 memory.")

        Args:
            other (Device): The other device to check peer access against.

        Returns:
            bool: True if peer access is possible, False otherwise.
        """

    def synchronize(self) -> None:
        """
        Ensures all operations on this device complete before returning.

        Raises:
            ValueError: If any enqueued operations had an internal error.
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
            stats = device.stats

        Returns:
            dict: A dictionary containing device utilization statistics.
        """

    @property
    def label(self) -> str:
        """
        Returns device label.

        Possible values are:

        - ``cpu`` for host devices.
        - ``gpu`` for accelerators.

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

        - ``cpu`` for host devices.
        - ``cuda`` for NVIDIA GPUs.
        - ``hip`` for AMD GPUs.

        .. code-block:: python

            from max import driver

            device = driver.CPU()
            device.api
        """

    @property
    def architecture_name(self) -> str:
        """
        Returns the architecture name of the device.

        Examples of possible values:

        - ``gfx90a``, ``gfx942`` for AMD GPUs.
        - ``sm_80``, ``sm_86`` for NVIDIA GPUs.
        - CPU devices raise an exception.

        .. code-block:: python

            from max import driver

            device = driver.Accelerator()
            device.architecture_name
        """

    @property
    def id(self) -> int:
        """
        Returns a zero-based device id. For a CPU device this is always 0.
        For GPU accelerators this is the id of the device relative to this host.
        Along with the ``label``, an id can uniquely identify a device,
        e.g. ``gpu:0``, ``gpu:1``.

        .. code-block:: python

            from max import driver

            device = driver.Accelerator()
            device_id = device.id

        Returns:
            int: The device ID.
        """

    @property
    def default_stream(self) -> DeviceStream:
        """
        Returns the default stream for this device.

        The default stream is initialized when the device object is created.

        Returns:
            DeviceStream: The default execution stream for this device.
        """

    @property
    def is_compatible(self) -> bool:
        """
        Returns whether this device is compatible with MAX.

        Returns:
            bool: True if the device is compatible with MAX, False otherwise.
        """

    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __eq__(self, arg: object, /) -> bool: ...
    def __hash__(self) -> int: ...
    @staticmethod
    def cpu(id: int = -1) -> CPU:
        """Creates a CPU device. The id is ignored currently."""

class Accelerator(Device):
    def __init__(self, id: int = -1, device_memory_limit: int = -1) -> None:
        """
        Creates an accelerator device with the specified ID and memory limit.

                Provides access to GPU or other hardware accelerators in the system.

                Repeated instantiations with a previously-used device-id will still
                refer to the first such instance that was created. This is especially
                important when providing a different memory limit: only the value
                (implicitly or explicitly) provided in the first such instantiation
                is effective.

                .. code-block:: python

                  from max import driver
                  # Create default accelerator (usually first available GPU)
                  device = driver.Accelerator()
                  # Or specify GPU id
                  device = driver.Accelerator(id=0)  # First GPU
                  device = driver.Accelerator(id=1)  # Second GPU
                  # Get device id
                  device_id = device.id
                  # Optionally specify memory limit
                  device = driver.Accelerator(id=0, device_memory_limit=256*MB)
                  device2 = driver.Accelerator(id=0, device_memory_limit=512*MB)
                  # ... device2 will use the memory limit of 256*MB

                Args:
                    id (int, optional): The device ID to use. Defaults to -1, which selects
                        the first available accelerator.
                    device_memory_limit (int, optional): The maximum amount of memory
                        in bytes that can be allocated on the device. Defaults to 99%
                        of free memory.

                Returns:
                    Accelerator: A new Accelerator device object.
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

        Args:
            id (int, optional): The device ID to use.
                Defaults to -1.

        Returns:
            CPU: A new CPU device object.
        """

class DeviceStream:
    """
    Provides access to a stream of execution on a device.

    A stream represents a sequence of operations that will be executed in order.
    Multiple streams on the same device can execute concurrently.

    .. code-block:: python

        from max import driver
        # Create a default accelerator device
        device = driver.Accelerator()
        # Get the default stream for the device
        stream = device.default_stream
        # Create a new stream of execution on the device
        new_stream = driver.DeviceStream(device)
    """

    def __init__(self, device: Device) -> None:
        """
        Creates a new stream of execution associated with the device.

        Args:
            device (Device): The device to create the stream on.

        Returns:
            DeviceStream: A new stream of execution.
        """

    def synchronize(self) -> None:
        """
        Ensures all operations on this stream complete before returning.

        Raises:
            ValueError: If any enqueued operations had an internal error.
        """

    @overload
    def wait_for(self, stream: DeviceStream) -> None:
        """
        Ensures all operations on the other stream complete before future work
        submitted to this stream is scheduled.

        Args:
            stream (DeviceStream): The stream to wait for.
        """

    @overload
    def wait_for(self, device: Device) -> None:
        """
        Ensures all operations on device's default stream complete before
        future work submitted to this stream is scheduled.

        Args:
            device (Device): The device whose default stream to wait for.
        """

    @property
    def device(self) -> Device:
        """The device this stream is executing on."""

    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __eq__(self, arg: object, /) -> bool: ...

def accelerator_count() -> int:
    """Returns number of accelerator devices available."""

class Tensor:
    """
    Device-resident tensor representation.

    Allocates memory onto a given device with the provided shape and dtype.
    Tensors can be sliced to provide strided views of the underlying memory,
    but any tensors input into model execution must be contiguous.

    Supports numpy-style slicing but does not currently support setting
    items across multiple indices.

    .. code-block:: python

        from max import driver
        from max.dtype import DType

        # Create a tensor on CPU
        cpu_tensor = driver.Tensor(shape=[2, 3], dtype=DType.float32)

        # Create a tensor on GPU
        gpu = driver.Accelerator()
        gpu_tensor = driver.Tensor(shape=[2, 3], dtype=DType.float32, device=gpu)

    Args:
        dtype (DType): Data type of tensor elements.
        shape (Sequence[int]): Tuple of positive, non-zero integers denoting the tensor shape.
        device (Device, optional): Device to allocate tensor onto. Defaults to the CPU.
        pinned (bool, optional): If True, memory is page-locked (pinned). Defaults to False.
        stream (DeviceStream, optional): Stream to associate the tensor with.
    """

    @overload
    def __init__(
        self,
        dtype: max._core.dtype.DType,
        shape: Sequence[int],
        device: Device | None = None,
        pinned: bool = False,
    ) -> None: ...
    @overload
    def __init__(
        self,
        dtype: max._core.dtype.DType,
        shape: Sequence[int],
        stream: DeviceStream,
        pinned: bool = False,
    ) -> None: ...
    @overload
    def __init__(
        self, shape: Annotated[ArrayLike, dict(writable=False)], device: Device
    ) -> None: ...
    @property
    def device(self) -> Device:
        """Device on which tensor is resident."""

    @property
    def stream(self) -> DeviceStream:
        """Stream to which tensor is bound."""

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

            cpu_tensor = driver.Tensor(shape=[2, 3], dtype=DType.bfloat16, device=driver.CPU())

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

    @overload
    def copy(self, stream: DeviceStream) -> Tensor:
        """
        Creates a deep copy on the device associated with the stream.

        Args:
            stream (DeviceStream): The stream to associate the new tensor with.

        Returns:
            Tensor: A new tensor that is a copy of this tensor.
        """

    @overload
    def copy(self, device: Device | None = None) -> Tensor:
        """
        Creates a deep copy on an optionally given device.

        If device is None (default), a copy is created on the same device.

        .. code-block:: python

            from max import driver
            from max.dtype import DType

            cpu_tensor = driver.Tensor(shape=[2, 3], dtype=DType.bfloat16, device=driver.CPU())
            cpu_copy = cpu_tensor.copy()

            # Copy to GPU
            gpu = driver.Accelerator()
            gpu_copy = cpu_tensor.copy(device=gpu)

        Args:
            device (Device, optional): The device to create the copy on.
                Defaults to None (same device).

        Returns:
            Tensor: A new tensor that is a copy of this tensor.
        """

    @staticmethod
    def mmap(
        filename: os.PathLike,
        dtype: max._core.dtype.DType,
        shape: Sequence[int],
        mode: numpy._MemMapModeKind = "copyonwrite",
        offset: int = 0,
    ):
        """
        Create a memory-mapped tensor from a binary file on disk.
                  The constructor argument semantics follow that of np.memmap.
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
        Creates a tensor from an object implementing the dlpack protocol.

        This usually does not result in a copy, and the producer of the object
        retains ownership of the underlying memory.

        Args:
            array (Any): An object that implements the dlpack protocol.
            copy (bool, optional): Whether to create a copy of the data.
                Defaults to None.

        Returns:
            Tensor: A new tensor that views or copies the dlpack data.
        """

    @staticmethod
    def from_numpy(arr: numpy.ndarray) -> Tensor:
        """
        Creates a tensor from a provided numpy array on the host device.

        The underlying data is not copied unless the array is noncontiguous. If
        it is, a contiguous copy will be returned.

        Args:
            arr (numpy.ndarray): The numpy array to convert.

        Returns:
            Tensor: A new tensor that views or copies the numpy array data.
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

    @overload
    def to(self, device: Device) -> Tensor:
        """
        Return a tensor that's guaranteed to be on the given device.

        The tensor is only copied if the requested device is different from the
        device upon which the tensor is already resident.
        """

    @overload
    def to(self, stream: DeviceStream) -> Tensor:
        """
        Return a tensor that's guaranteed to be on the given device and associated
        with the given stream.

        The tensor is only copied if the requested device is different from the
        device upon which the tensor is already resident.
        """

    @overload
    def to(self, devices: Sequence[Device]) -> list[Tensor]:
        """
        Return a list of tensors that are guaranteed to be on the given devices.

        The tensors are only copied if the requested devices are different from the
        device upon which the tensor is already resident.
        """

    @overload
    def to(self, streams: Sequence[DeviceStream]) -> list[Tensor]:
        """
        Return a list of tensors that are guaranteed to be on the given streams.

        The tensors are only copied if the requested streams are different from the
        stream upon which the tensor is already resident.
        """

    def to_numpy(self) -> numpy.ndarray:
        """
        Converts the tensor to a numpy array.

        If the tensor is on the host (CPU), the numpy array aliases the existing memory.
        Otherwise, it is copied to the host device.

        Returns:
            numpy.ndarray: A numpy array containing the tensor data.
        """

    @property
    def pinned(self) -> bool:
        """Whether or not the underlying memory is pinned (page-locked)."""

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
        """
        Allocates a tensor with all elements initialized to zero.

        Args:
            shape (Sequence[int]): The shape of the tensor.
            dtype (DType): The data type of the tensor.
            device (Device, optional): The device to allocate the tensor on.
                Defaults to None (CPU).

        Returns:
            Tensor: A new tensor filled with zeros.
        """

    def __dlpack__(
        self, *, stream: object | None = None
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

    def _aligned(self, alignment: int | None = None) -> bool:
        """Returns whether the tensor is aligned to the desired alignment."""

    @overload
    @staticmethod
    def _from_dlpack(arg: object, /) -> Tensor: ...
    @overload
    @staticmethod
    def _from_dlpack(
        arg0: typing_extensions.CapsuleType, arg1: Device, arg2: int, /
    ) -> Tensor: ...
    def _iterate_indices(self) -> Generator[Sequence[int]]: ...
    def _view(
        self, dtype: max._core.dtype.DType, shape: Sequence[int]
    ) -> Tensor: ...
    def _inplace_copy_from(self, src: Tensor) -> None: ...
    def _data_ptr(self) -> int:
        """Gets the memory address of the tensor data. Internal use only."""
