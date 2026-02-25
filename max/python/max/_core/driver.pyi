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
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

import os
import types
from collections.abc import Generator, Mapping, Sequence
from typing import Annotated, Any, overload

import max._core.dtype
import numpy
from numpy.typing import NDArray

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
    def _device_context_ptr(self) -> int:
        """Gets the device context pointer. Returns 0 for host devices."""

    @staticmethod
    def cpu(id: int = -1) -> CPU:
        """Creates a CPU device. The id is ignored currently."""

class Accelerator(Device):
    def __init__(self, id: int = -1) -> None:
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

        Args:
            id (int, optional): The device ID to use. Defaults to -1, which selects
                the first available accelerator.

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

class DeviceEvent:
    """
    Provides access to an event object.

    An event can be used to wait for the GPU execution to reach a certain
    point on the given stream.

    .. code-block:: python

        from max import driver
        # Create a default accelerator device
        device = driver.Accelerator()
        # Create an event on the device
        event = driver.DeviceEvent(device)
        # Record an event on the device (default stream)
        device.default_stream.record_event(event)
        # Wait for execution on the default stream to reach the event
        event.synchronize()
    """

    def __init__(self, device: Device) -> None:
        """
        Creates an event for synchronization on the specified device.

        Args:
            device (Device): The device on which to create the event.

        Raises:
            ValueError: If event creation failed.

        .. code-block:: python

            from max import driver

            device = driver.Accelerator()
            event = driver.DeviceEvent(device)
        """

    def synchronize(self) -> None:
        """
        Ensures all operations on this stream complete before returning.

        Raises:
            ValueError: If any enqueued operations had an internal error.
        """

    def is_ready(self) -> bool:
        """
        Returns whether this event is ready.

        Returns:
            bool: True if the event is complete, otherwise false.
        Raises:
          ValueError: If querying the event status returned an error
        """

    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __eq__(self, arg: object, /) -> bool: ...

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
    def record_event(self) -> DeviceEvent:
        """
        Records an event on this stream.

        Returns:
            DeviceEvent: A new event that will be signaled when all operations
                submitted to this stream before this call have completed.

        Raises:
            ValueError: If recording the event failed.
        """

    @overload
    def record_event(self, event: DeviceEvent) -> None:
        """
        Records an existing event on this stream.

        Args:
            event (DeviceEvent): The event to record on this stream.

        Raises:
            ValueError: If recording the event failed.
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

def set_virtual_device_count(count: int) -> None:
    """
    Set the number of virtual devices for device creation.

    When count is greater than 0, Device::create() will return VirtualDevice
    instances instead of real hardware devices for GPU APIs, and
    Device::numberOfDevices() will return this count. This allows creating
    devices for GPU configurations that don't match the current hardware.

    Args:
        count (int): The number of virtual devices. Set to 0 to disable
            virtual device mode.
    """

def get_virtual_device_count() -> int:
    """
    Get the current virtual device count.

    Returns:
        int: The number of virtual devices, or 0 if virtual device mode
            is disabled.
    """

def is_virtual_device_mode() -> bool:
    """
    Check if virtual device mode is currently enabled.

    Returns:
        bool: True if virtual device mode is enabled (count > 0), False otherwise.
    """

def set_virtual_device_api(api: str) -> None:
    """
    Set the target API for virtual devices in compile-only mode.

    This specifies which GPU API (e.g., "cuda", "hip", "metal") virtual
    devices will use for compilation. Must be called before creating
    virtual devices via set_virtual_device_count().

    Args:
        api (str): The target API string (e.g., "cuda" for NVIDIA,
            "hip" for AMD, "metal" for Apple).
    """

def get_virtual_device_api() -> str:
    """
    Get the current target API for virtual devices.

    Returns:
        str: The target API string, or empty string if not set.
    """

def set_virtual_device_target_arch(arch: str) -> None:
    """
    Set the target GPU architecture for virtual devices in compile-only mode.

    This specifies the GPU architecture (e.g., "sm_80", "sm_90") that virtual
    devices will target when compiling code. Must be called before creating
    virtual devices via set_virtual_device_count().

    Args:
        arch (str): The target GPU architecture string (e.g., "sm_80" for
            Ampere/A100, "sm_90" for Hopper/H100).
    """

def get_virtual_device_target_arch() -> str:
    """
    Get the current target GPU architecture for virtual devices.

    Returns:
        str: The target GPU architecture string, or empty string if not set.
    """

class Buffer:
    """
    Device-resident buffer representation.

    Allocates memory onto a given device with the provided shape and dtype.
    Buffers can be sliced to provide strided views of the underlying memory,
    but any buffers input into model execution must be contiguous.

    Supports numpy-style slicing but does not currently support setting
    items across multiple indices.

    .. code-block:: python

        from max import driver
        from max.dtype import DType

        # Create a buffer on CPU
        cpu_buffer = driver.Buffer(shape=[2, 3], dtype=DType.float32)

        # Create a buffer on GPU
        gpu = driver.Accelerator()
        gpu_buffer = driver.Buffer(shape=[2, 3], dtype=DType.float32, device=gpu)

    Args:
        dtype (DType): Data type of buffer elements.
        shape (Sequence[int]): Tuple of positive, non-zero integers denoting the buffer shape.
        device (Device, optional): Device to allocate buffer onto. Defaults to the CPU.
        pinned (bool, optional): If True, memory is page-locked (pinned). Defaults to False.
        stream (DeviceStream, optional): Stream to associate the buffer with.
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
        self, shape: Annotated[NDArray, dict(writable=False)], device: Device
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
        Whether or not buffer is contiguously allocated in memory. Returns
        false if the buffer is a non-contiguous slice.

        Currently, we consider certain situations that are contiguous as
        non-contiguous for the purposes of our engine, such as when a buffer
        has negative steps.
        """

    @property
    def is_host(self) -> bool:
        """
        Whether or not buffer is host-resident. Returns false for GPU buffers,
        true for CPU buffers.

        .. code-block:: python

            from max import driver
            from max.dtype import DType

            cpu_buffer = driver.Buffer(shape=[2, 3], dtype=DType.bfloat16, device=driver.CPU())

            print(cpu_buffer.is_host)
        """

    @property
    def num_elements(self) -> int:
        """
        Returns the number of elements in this buffer.

        Rank-0 buffers have 1 element by convention.
        """

    @property
    def rank(self) -> int:
        """Buffer rank."""

    @property
    def shape(self) -> tuple:
        """Shape of buffer."""

    def contiguous(self) -> Buffer:
        """Creates a contiguous copy of the buffer."""

    def disable_auto_sync(self) -> None:
        """
        Disables automatic synchronization for asynchronous operations on this buffer.

        .. caution::
          This is an experimental feature that may be unstable. It also
          requires special care from the user to ensure proper synchronization.

        By default, certain operations on buffers cause synchronization, such
        as when trying to access a buffer on the host through `to_numpy`.
        However the default synchronization is quite conservative and often
        ends up waiting on more than what is strictly needed.

        This function disables the default synchronization method and enables
        `mark_as_ready()`, which allows for a finer control of what is waited on
        when a buffer needs to be synchronized.

        .. code-block:: python

            # Assuming we have 3 buffers of the same sizes, a, b and c

            # Default case with auto-synchronization
            a.to(b) # 1
            a.to(c) # 2

            # Will wait on 1 and 2
            b.to_numpy()

            # Disabled synchronization
            a.disable_auto_sync()
            a.to(b) # 1
            a.to(c) # 2

            # Doesn't wait on 1 or 2, data in b could be invalid
            b.to_numpy()

            # Disabled synchronization with mark_as_ready
            a.disable_auto_sync()
            a.to(b) # 1
            b.mark_as_ready()
            a.to(c) # 2

            # Wait on 1 but not on 2
            b.to_numpy()
        """

    def mark_as_ready(self) -> None:
        """
        Establishes a synchronization point for buffers with disabled auto-sync.

        .. caution::
          This is an experimental feature that may be unstable. It also
          requires special care from the user to ensure proper synchronization.

        This method can only be called on buffers with disabled synchronization
        through `disable_auto_sync()`.

        It instructs max that whenever it needs to wait on this buffer it
        should only wait to the point where this was called.

        It can be called multiple times, but it will override a previous
        synchronization point with the new one.

        Refer to the `disable_auto_sync()` documentation for more details and examples.
        """

    @overload
    def copy(self, stream: DeviceStream) -> Buffer:
        """
        Creates a deep copy on the device associated with the stream.

        Args:
            stream (DeviceStream): The stream to associate the new buffer with.

        Returns:
            Buffer: A new buffer that is a copy of this buffer.
        """

    @overload
    def copy(self, device: Device | None = None) -> Buffer:
        """
        Creates a deep copy on an optionally given device.

        If device is None (default), a copy is created on the same device.

        .. code-block:: python

            from max import driver
            from max.dtype import DType

            cpu_buffer = driver.Buffer(shape=[2, 3], dtype=DType.bfloat16, device=driver.CPU())
            cpu_copy = cpu_buffer.copy()

            # Copy to GPU
            gpu = driver.Accelerator()
            gpu_copy = cpu_buffer.copy(device=gpu)

        Args:
            device (Device, optional): The device to create the copy on.
                Defaults to None (same device).

        Returns:
            Buffer: A new buffer that is a copy of this buffer.
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
        Create a memory-mapped buffer from a binary file on disk.
        The constructor argument semantics follow that of np.memmap.
        """

    def inplace_copy_from(self, src: Buffer) -> None:
        """
        Copy the contents of another buffer into this one. These buffers may
        be on different devices.
        Requires that both buffers are contiguous and have same size.
        """

    @staticmethod
    def from_dlpack(array: Any, *, copy: bool | None = None) -> Buffer:
        """
        Creates a buffer from an object implementing the dlpack protocol.

        This usually does not result in a copy, and the producer of the object
        retains ownership of the underlying memory.

        Args:
            array (Any): An object that implements the dlpack protocol.
            copy (bool, optional): Whether to create a copy of the data.
                Defaults to None.

        Returns:
            Buffer: A new buffer that views or copies the dlpack data.
        """

    @staticmethod
    def from_numpy(arr: numpy.ndarray) -> Buffer:
        """
        Creates a buffer from a provided numpy array on the host device.

        The underlying data is not copied unless the array is noncontiguous. If
        it is, a contiguous copy will be returned.

        Args:
            arr (numpy.ndarray): The numpy array to convert.

        Returns:
            Buffer: A new buffer that views or copies the numpy array data.
        """

    def item(self) -> Any:
        """
        Returns the scalar value at a given location. Currently
        implemented only for zero-rank buffers. The return type is
        converted to a Python built-in type.
        """

    @staticmethod
    def scalar(
        value: Any, dtype: max._core.dtype.DType, device: Device | None = None
    ) -> Buffer:
        """
        Create a scalar value of a given dtype and value.

        If device is None (default), the buffer will be allocated on the CPU.
        """

    @overload
    def to(self, device: Device) -> Buffer:
        """
        Return a buffer that's guaranteed to be on the given device.

        The buffer is only copied if the requested device is different from the
        device upon which the buffer is already resident.
        """

    @overload
    def to(self, stream: DeviceStream) -> Buffer:
        """
        Return a buffer that's guaranteed to be on the given device and associated
        with the given stream.

        The buffer is only copied if the requested device is different from the
        device upon which the buffer is already resident. If the destination
        stream is on the same device, then a new reference to the same buffer is
        returned.
        """

    @overload
    def to(self, devices: Sequence[Device]) -> list[Buffer]:
        """
        Return a list of buffers that are guaranteed to be on the given devices.

        The buffers are only copied if the requested devices are different from the
        device upon which the buffer is already resident.
        """

    @overload
    def to(self, streams: Sequence[DeviceStream]) -> list[Buffer]:
        """
        Return a list of buffers that are guaranteed to be on the given streams.

        The buffers are only copied if the requested streams are different from the
        stream upon which the buffer is already resident.
        """

    def to_numpy(self) -> numpy.ndarray:
        """
        Converts the buffer to a numpy array.

        If the buffer is on the host (CPU), the numpy array aliases the existing memory.
        Otherwise, it is copied to the host device.

        Returns:
            numpy.ndarray: A numpy array containing the buffer data.
        """

    @property
    def pinned(self) -> bool:
        """Whether or not the underlying memory is pinned (page-locked)."""

    def view(
        self, dtype: max._core.dtype.DType, shape: Sequence[int] | None = None
    ) -> Buffer:
        """
        Return a new buffer with the given type and shape that shares the
        underlying memory.
        If the shape is not given, it will be deduced if possible, or a
        ValueError is raised.
        """

    @staticmethod
    def zeros(
        shape: Sequence[int],
        dtype: max._core.dtype.DType,
        device: Device | None = None,
        pinned: bool = False,
    ) -> Buffer:
        """
        Allocates a buffer with all elements initialized to zero.

        Args:
            shape (Sequence[int]): The shape of the buffer.
            dtype (DType): The data type of the buffer.
            device (Device, optional): The device to allocate the buffer on.
                Defaults to None (CPU).
            pinned (bool, optional): If True, allocate pinned host memory for
                non-CPU devices. Defaults to False.

        Returns:
            Buffer: A new buffer filled with zeros.
        """

    def __dlpack__(
        self, *, stream: int | None = None, **kwargs
    ) -> types.CapsuleType:
        """Implements part of the dlpack contract."""

    def __dlpack_device__(self) -> tuple:
        """Implements part of the dlpack contract."""

    def __getitem__(self, idx: int | slice | Sequence[int | slice]) -> Buffer:
        """
        Gets a buffer slice. Supports full numpy-style slicing. Invocations
        using only integer-based indexes will return zero-rank buffers.
        """

    def __setitem__(
        self, idx: int | slice | Sequence[int | slice], value: Any
    ) -> None:
        """Sets an item in the buffer."""

    def _aligned(self, alignment: int | None = None) -> bool:
        """Returns whether the buffer is aligned to the desired alignment."""

    @overload
    @staticmethod
    def _from_dlpack(arg: object, /) -> Buffer: ...
    @overload
    @staticmethod
    def _from_dlpack(
        arg0: types.CapsuleType, arg1: Device, arg2: int, /
    ) -> Buffer: ...
    def _iterate_indices(self) -> Generator[Sequence[int]]: ...
    def _view(
        self, dtype: max._core.dtype.DType, shape: Sequence[int]
    ) -> Buffer: ...
    def _inplace_copy_from(self, src: Buffer) -> None: ...
    def _data_ptr(self) -> int:
        """Gets the memory address of the buffer data. Internal use only."""
