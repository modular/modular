# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import Optional

from max.tensor import TensorShape, TensorSpec
from memory import UnsafePointer

from ._driver_library import DriverLibrary
from ._status import Status
from .anytensor import AnyTensor
from .device import Device, _CDevice
from .tensor import Tensor


trait DeviceBuffer:
    fn copy_to(self, dev: Device, name: Optional[String]) raises -> Self:
        """Copies the contents of self into DeviceBuffer allocated on dev.
        Note: this function allocates memory on dev.

        Args:
            dev: The Device on which to allocate the new DeviceBuffer.
            name: Optional name of the new DeviceBuffer.

        Returns:
            Newly allocated DeviceBuffer containing a copy of self's contents.

        Raises:
            If the DeviceBuffer is backed by the same Device object as dev.
        """
        ...

    fn copy_into(self, mut dst_memory: Self) raises:
        """Copies the contents of self into a preallocated DeviceBuffer.

        Args:
            dst_memory: The destination DeviceBuffer of the copy.
        """
        ...

    fn move_to(owned self, dev: Device) raises -> Self:
        """Returns self if already allocated on dev, otherwise copy the contents
        of self to dev.

        Args:
            dev: The Device of the returned buffer.

        Returns:
            A DeviceBuffer located in dev's address space.
        """
        ...

    fn unsafe_ptr(self) -> UnsafePointer[UInt8]:
        """Returns a pointer to the DeviceBuffer's storage in device memory."""
        ...

    fn device(self) -> Device:
        """Returns the Device on which the DeviceBuffer was allocated."""
        ...

    fn bytecount(self) -> Int:
        """Returns the size of the DeviceBuffer in bytes."""
        ...


struct DeviceMemory(DeviceBuffer, StringableRaising, CollectionElement):
    """DeviceMemory is an owning buffer allocated on a (possibly non-CPU) Device.
    """

    var _impl_ptr: UnsafePointer[NoneType]
    var _device: Device
    var name: Optional[String]
    var num_bytes: Int

    fn __init__(out self):
        """Constructs a DeviceMemory object in a state that is only valid for deletion.
        Can be used to represent a `moved from` state.
        """
        self = Self(
            UnsafePointer[NoneType](),
            0,
            Device(),
        )

    fn __init__(
        out self,
        num_bytes: Int,
        device: Device,
        name: Optional[String] = None,
    ) raises:
        """Allocates DeviceMemory from the Device's address space.

        Args:
            num_bytes: Size of the DeviceMemory buffer to allocate in bytes.
            device: Device on which to perform the allocation.
            name: Optional name for the DeviceMemory.

        """
        self._device = device
        var tmp_spec = TensorSpec(DType.uint8, num_bytes)
        var status = Status(device._lib.value())
        # CAUTION: this assumes that TensorSpec is bitwise identical in mojo and cpp
        self._impl_ptr = device._lib.value().create_device_memory_fn(
            UnsafePointer[TensorSpec](to=tmp_spec),
            self._device._cdev._ptr,
            status.impl,
        )
        if status:
            raise String(status)
        self.name = name
        self.num_bytes = num_bytes

    @doc_private
    fn __init__(
        out self,
        owned_impl_ptr: UnsafePointer[NoneType],
        num_bytes: Int,
        device: Device,
        name: Optional[String] = None,
    ):
        self._device = device
        self._impl_ptr = owned_impl_ptr
        self.name = name
        self.num_bytes = num_bytes

    fn __copyinit__(out self, existing: Self):
        # This temporarily exists so that we can store DeviceMemory in a List
        # TODO(MSTDL-467): Once Copyable requirement on List is removed, this
        # can be removed
        constrained[
            False, "__copyinit__ not supported on DeviceMemory, MSTDL-467"
        ]()
        self._impl_ptr = existing._impl_ptr
        self._device = existing._device
        self.name = existing.name
        self.num_bytes = existing.num_bytes

    @always_inline
    fn copy(self) -> Self:
        """Explicitly construct a copy of self.

        Returns:
            A copy of this value.
        """
        return self

    fn __init__[
        type: DType, rank: Int
    ](out self, owned tensor: Tensor[type, rank]) raises:
        """Creates a DeviceMemory from the existing `tensor` storage.

        Args:
            tensor: Tensor whose storage to use.
        """

        self._device = tensor._get_device()
        self.name = tensor.name
        self.num_bytes = tensor.spec().bytecount()
        self._impl_ptr = tensor._device_memory_impl_ptr
        tensor._device_memory_impl_ptr = UnsafePointer[NoneType]()

    fn __init__(out self, owned anytensor: AnyTensor) raises:
        """Creates a device tensor the existing `anytensor` storage.

        Args:
            anytensor: AnyTensor whose storage to use.

        """

        self._device = anytensor._device
        self.name = anytensor._name
        self._impl_ptr = anytensor._device_memory_impl_ptr
        self.num_bytes = anytensor._spec.bytecount()
        anytensor._device_memory_impl_ptr = UnsafePointer[NoneType]()

    fn __del__(owned self):
        """De-allocate and destroy the DeviceMemory.

        Note: this will also decrement the refcount on the Device used to allocate
        the DeviceMemory.
        """
        if not self._impl_ptr:
            return
        self._device._lib.value().destroy_device_memory_fn(self._impl_ptr)

    fn bytecount(self) -> Int:
        """Returns the number of bytes in the DeviceMemory."""
        return self.num_bytes

    fn get_device(self) -> Device:
        """Returns the device on which the DeviceMemory was allocated."""

        return self._device

    fn device(self) -> Device:
        """Returns the device on which the DeviceMemory was allocated."""

        return self._device

    fn __moveinit__(out self, owned existing: Self):
        self._impl_ptr = existing._impl_ptr
        self._device = existing._device^
        self.name = existing.name^
        self.num_bytes = existing.num_bytes

    fn __str__(self) raises -> String:
        """Returns a description of the DeviceMemory."""
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        """
        Formats a description of the DeviceMemory to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """
        writer.write(
            "DeviceMemory(",
            self.name.value() if self.name else "",
            "," if self.name else "",
            self.get_device(),
            ",Bytecount(",
            self.bytecount(),
            "))",
        )

    fn _steal_impl_ptr(owned self) -> UnsafePointer[NoneType]:
        var tmp = self._impl_ptr
        self._impl_ptr = UnsafePointer[NoneType]()
        return tmp

    fn _steal_ptr(owned self) -> UnsafePointer[UInt8]:
        alias func_name_take_data = "M_takeDataFromDeviceMemory"
        var take_data_func = self._device._lib.value().get_handle().get_function[
            fn (UnsafePointer[NoneType]) -> UnsafePointer[UInt8]
        ](
            func_name_take_data
        )
        var data = take_data_func(self._impl_ptr)
        # Extend lifetime of self to avoid the C funtion working on invalid pointer.
        _ = self^
        return data

    fn copy_into(self, mut dst_memory: DeviceMemory) raises:
        """Copies the contents of self into preallocated DeviceMemory.

        Args:
            dst_memory: The destination DeviceMemory of the copy.
        """
        if self.bytecount() != dst_memory.bytecount():
            raise String(
                "source bytecount({}) does not match destination bytecount({})"
            ).format(self.bytecount(), dst_memory.bytecount())

        var status = Status(self._device._lib.value())

        self._device._lib.value().copy_device_memory_fn(
            dst_memory._impl_ptr, self._impl_ptr, status.impl
        )
        if status:
            raise String(status)

    fn copy_to(
        self, dev: Device, name: Optional[String] = None
    ) raises -> DeviceMemory:
        """Copies the contents of self into DeviceMemory allocated on dev.
        Note: this function allocates memory on dev.

        Args:
            dev: The Device on which to allocate the new DeviceMemory.
            name: Optional name of the new DeviceMemory.

        Returns:
            Newly allocated DeviceMemory containing a copy of self's contents.

        Raises:
            If the DeviceMemory is backed by the same Device object as dev.
        """
        if dev == self._device:
            raise Error(self, "is already allocated on ", dev)

        var dst = dev.allocate(self.bytecount(), name)
        self.copy_into(dst)
        return dst^

    fn move_to(owned self, dev: Device) raises -> Self:
        """Returns self if already allocated on dev, otherwise copy the contents
        of self to dev.

        Args:
            dev: The Device on which the returned buffer is allocated.

        Returns:
            DeviceMemory located in dev's address space.
        """
        if dev == self._device:
            return self^
        else:
            return self.copy_to(dev)

    fn unsafe_ptr(self) -> UnsafePointer[UInt8]:
        """Returns a pointer to the underlying device memory.

        Note: The caller is responsible for ensuring that the returned pointer
        is not used after its owner is last used.
        """

        return self._device._lib.value().get_data_fn(self._impl_ptr)

    fn take(mut self) raises -> Self:
        """Takes and returns the contents of `self`, leaving `self` in an empty but destructible state.
        """
        var tmp = Self()
        swap(tmp, self)
        return tmp


struct DeviceTensor(DeviceBuffer, StringableRaising, CollectionElement):
    var _storage: DeviceMemory
    var spec: TensorSpec

    fn __init__(
        out self, spec: TensorSpec, device: Device, name: Optional[String]
    ) raises:
        """Allocates a DeviceTensor in the Device's address space.

        Args:
            spec: TensorSpec describing the dtype and shape of the DeviceTensor.
            device: Device on which to perform the allocation.
            name: Optional name for the DeviceMemory.

        """
        self.spec = spec
        self._storage = DeviceMemory(
            spec.bytecount(),
            device,
            name,
        )

    fn __init__(out self):
        """Constructs a DeviceTensor in a state that is only valid for deletion.
        Can be used to represent a `moved from` state.
        """
        self.spec = TensorSpec()
        self._storage = DeviceMemory()

    fn __init__(out self, owned storage: DeviceMemory, spec: TensorSpec) raises:
        """Constructs a DeviceTensor from an existing storage buffer and spec.

        Args:
            storage: The storage backing the DeviceTensor.
            spec: TensorSpec describing the type and shape of the DeviceTensor.
        """
        self._storage = storage^
        self.spec = spec

        if self.bytecount() != self.bytecount():
            raise "DeviceMemory size does not match DeviceTensor requirements"

    fn copy_to(self, dev: Device, name: Optional[String] = None) raises -> Self:
        """Copies the contents of self into a DeviceTensor allocated on dev.
        Note: this function allocates memory on dev.

        Args:
            dev: The Device on which to allocate the new DeviceTensor.
            name: Optional name of the new DeviceTensor.

        Returns:
            Newly allocated DeviceTensor containing a copy of self's contents.

        Raises:
            If the DeviceTensor is backed by the same Device object as dev.
        """
        var t = Self(self._storage.copy_to(dev, name), self.spec)
        return t

    fn copy_into(self, mut dst_tensor: Self) raises:
        """Copies the contents of self into a preallocated DeviceTensor.

        Args:
            dst_tensor: The destination DeviceTensor of the copy.
        """
        if dst_tensor.spec != self.spec:
            raise String(
                "source({}) and destination({}) specs do not match"
            ).format(self.spec, dst_tensor.spec)
        self._storage.copy_into(dst_tensor._storage)

    fn move_to(owned self, dev: Device) raises -> Self:
        """Returns self if already allocated on dev, otherwise copy the contents
        of self to dev.

        Args:
            dev: The Device on which the returned buffer is allocated.

        Returns:
            A DeviceTensor located in dev's address space.
        """
        if dev == self._storage._device:
            return self^
        else:
            return self.copy_to(dev)

    fn unsafe_ptr(self) -> UnsafePointer[UInt8]:
        """Returns a pointer to the DeviceTensor's storage in device memory."""
        return self._storage.unsafe_ptr()

    fn to_tensor[
        type: DType, rank: Int
    ](owned self) raises -> Tensor[type, rank]:
        """Returns a Tensor created using the DeviceTensor's shape and storage.
        """
        if rank != self.spec.rank():
            raise "requested rank does not match existing rank"

        if type != self.spec.dtype():
            raise "requested dtype does not match existing type."

        return Tensor[type, rank](device_tensor=self^)

    fn device(self) -> Device:
        """Returns the Device on which the DeviceTensor was allocated."""
        return self._storage.device()

    fn name(self) -> Optional[String]:
        """Returns the name of the DeviceTensor."""
        return self._storage.name

    fn __copyinit__(out self, existing: Self):
        # This temporarily exists so that we can store DeviceMemory in a List
        # TODO(MSTDL-467): Once Copyable requirement on List is removed, this
        # can be removed
        constrained[
            False, "__copyinit__ not supported on DeviceTensor, MSTDL-467"
        ]()
        self._storage = existing._storage
        self.spec = existing.spec

    @always_inline
    fn copy(self) -> Self:
        """Explicitly construct a copy of self.

        Returns:
            A copy of this value.
        """
        return self

    fn __moveinit__(out self, owned existing: Self):
        self._storage = existing._storage^
        self.spec = existing.spec^

    fn __str__(self) raises -> String:
        """Returns a descriptor for the DeviceTensor."""
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        """
        Formats a description of the DeviceTensor to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """
        writer.write(
            "DeviceTensor(",
            self._storage.name.value() if self._storage.name else "",
            "," if self._storage.name else "",
            self.device(),
            ",Spec(",
            self.spec,
            "))",
        )

    fn bytecount(self) -> Int:
        """Returns the number of bytes in the DeviceTensor."""
        return self.spec.bytecount()

    fn take(mut self) -> Self:
        """Takes and returns the contents of `self`, leaving `self` in an empty but destructible state.
        """
        var tmp = Self()
        swap(tmp, self)
        return tmp
