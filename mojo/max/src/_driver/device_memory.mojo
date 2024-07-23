# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from max.tensor import TensorSpec, TensorShape
from ._driver_library import DriverLibrary
from max._utils import call_dylib_func
from .device import Device, _CDevice
from .tensor import Tensor
from .anytensor import AnyTensor


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

    fn copy_into(self, inout dst_memory: Self) raises:
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
        """
        ...

    fn unsafe_ptr(self) -> UnsafePointer[UInt8]:
        ...

    fn device(self) -> Device:
        ...

    fn bytecount(self) -> Int:
        ...


struct DeviceMemory(DeviceBuffer, StringableRaising, CollectionElement):
    """DeviceMemory is an owning buffer allocated on a (possibly non-CPU) Device.
    """

    var _impl_ptr: UnsafePointer[NoneType]
    var _device: Device
    var lib: DriverLibrary
    var name: Optional[String]
    var num_bytes: Int

    fn __init__(inout self):
        self = Self(
            UnsafePointer[NoneType](),
            0,
            Device(),
        )

    fn __init__(
        inout self,
        num_bytes: Int,
        device: Device,
        name: Optional[String] = None,
    ) raises:
        """Allocates a DeviceMemory from the Device's address space."""
        self._device = device
        self.lib = self._device.lib
        alias func_name_create_tensor = "M_createDeviceMemory"
        var tmp_spec = TensorSpec(DType.uint8, num_bytes)
        # CAUTION: this assumes that TensorSpec is bitwise identical in mojo and cpp
        self._impl_ptr = call_dylib_func[UnsafePointer[NoneType]](
            self.lib.get_handle(),
            func_name_create_tensor,
            UnsafePointer[TensorSpec].address_of(tmp_spec),
            self._device._cdev,
        )
        self.name = name
        self.num_bytes = num_bytes
        _ = tmp_spec^

    fn __init__(
        inout self,
        owned_impl_ptr: UnsafePointer[NoneType],
        num_bytes: Int,
        device: Device,
        name: Optional[String] = None,
    ):
        """Creates DeviceMemory from a C Ptr."""
        self._device = device
        self.lib = self._device.lib
        self._impl_ptr = owned_impl_ptr
        self.name = name
        self.num_bytes = num_bytes

    fn __copyinit__(inout self, existing: Self):
        # This temporarily exists so that we can store DeviceMemory in a List
        # TODO(MSTDL-467): Once Copyable requirement on List is removed, this
        # can be removed
        constrained[
            False, "__copyinit__ not supported on DeviceMemory, MSTDL-467"
        ]()
        self._impl_ptr = existing._impl_ptr
        self._device = existing._device
        self.lib = existing.lib
        self.name = existing.name
        self.num_bytes = existing.num_bytes

    fn __init__[
        type: DType, rank: Int
    ](inout self, owned tensor: Tensor[type, rank]) raises:
        """Creates a DeviceMemory from given tensor."""

        self._device = tensor._get_device()
        self.lib = self._device.lib
        self.name = tensor.name
        self.num_bytes = tensor.spec().bytecount()
        self._impl_ptr = tensor._device_memory_impl_ptr
        tensor._device_memory_impl_ptr = UnsafePointer[NoneType]()

    fn __init__(inout self, owned anytensor: AnyTensor) raises:
        """Creates a device tensor from given anytensor."""

        self._device = anytensor._device
        self.lib = self._device.lib
        self.name = anytensor._name
        self._impl_ptr = anytensor._device_memory_impl_ptr
        self.num_bytes = anytensor._spec.bytecount()
        anytensor._device_memory_impl_ptr = UnsafePointer[NoneType]()

    fn __del__(owned self):
        alias func_name_destroy = "M_destroyDeviceMemory"
        call_dylib_func[NoneType](
            self.lib.get_handle(), func_name_destroy, self._impl_ptr
        )
        # Extend lifetime of library until C function returns.
        _ = self.lib^

    fn bytecount(self) -> Int:
        return self.num_bytes

    fn get_device(self) -> Device:
        """Returns the device on which the DeviceMemory was allocated."""

        return self._device

    fn device(self) -> Device:
        """Returns the device on which the DeviceMemory was allocated."""

        return self._device

    fn __moveinit__(inout self, owned existing: Self):
        self._impl_ptr = existing._impl_ptr
        self._device = existing._device^
        self.lib = existing.lib^
        self.name = existing.name^
        self.num_bytes = existing.num_bytes

    fn __str__(self) raises -> String:
        """Returns a description of the DeviceMemory."""

        return (
            "DeviceMemory("
            + (self.name.value() + "," if self.name else "")
            + str(self.get_device())
            + ",Bytecount("
            + str(self.bytecount())
            + "))"
        )

    fn _steal_impl_ptr(owned self) -> UnsafePointer[NoneType]:
        var tmp = self._impl_ptr
        self._impl_ptr = UnsafePointer[NoneType]()
        return tmp

    fn _steal_ptr(owned self) -> UnsafePointer[UInt8]:
        alias func_name_take_data = "M_takeDataFromDeviceMemory"
        var take_data_func = self.lib.get_handle().get_function[
            fn (UnsafePointer[NoneType]) -> UnsafePointer[UInt8]
        ](func_name_take_data)
        var data = take_data_func(self._impl_ptr)
        # Extend lifetime of self to avoid the C funtion working on invalid pointer.
        _ = self^
        return data

    fn copy_into(self, inout dst_memory: DeviceMemory) raises:
        if self.bytecount() != dst_memory.bytecount():
            raise "src and dst bytcount mismatch in copy_into()"

        alias func_name_steal_data = "M_copyDeviceMemory"
        call_dylib_func(
            self.lib.get_handle(),
            func_name_steal_data,
            dst_memory._impl_ptr,
            self._impl_ptr,
        )

    fn copy_to(
        self, dev: Device, name: Optional[String] = None
    ) raises -> DeviceMemory:
        if dev == self._device:
            raise str(self) + "is already allocated on " + str(dev)

        var dst = dev.allocate(self.bytecount(), name)
        self.copy_into(dst)
        return dst^

    fn move_to(owned self, dev: Device) raises -> Self:
        if dev == self._device:
            return self^
        else:
            return self.copy_to(dev)

    fn unsafe_ptr(self) -> UnsafePointer[UInt8]:
        """Returns a pointer to the underlying device memory.

        Note: The caller is responsible for ensuring that the returned pointer
        is not used after it's owner is last used.
        """

        alias func_name_get_data = "M_getData"
        return call_dylib_func[UnsafePointer[UInt8], UnsafePointer[NoneType],](
            self.lib.get_handle(),
            func_name_get_data,
            self._impl_ptr,
        )

    fn take(inout self) raises -> Self:
        var tmp = Self()
        swap(tmp, self)
        return tmp


struct DeviceTensor(DeviceBuffer, StringableRaising, CollectionElement):
    var _storage: DeviceMemory
    var spec: TensorSpec

    fn __init__(
        inout self, spec: TensorSpec, device: Device, name: Optional[String]
    ) raises:
        self.spec = spec
        self._storage = DeviceMemory(
            spec.bytecount(),
            device,
            name,
        )

    fn __init__(inout self):
        self.spec = TensorSpec()
        self._storage = DeviceMemory()

    fn __init__(
        inout self, owned storage: DeviceMemory, spec: TensorSpec
    ) raises:
        self._storage = storage^
        self.spec = spec

        if self.bytecount() != self.bytecount():
            raise "DeviceMemory size does not match DeviceTensor requirements"

    fn copy_to(self, dev: Device, name: Optional[String] = None) raises -> Self:
        var t = Self(self._storage.copy_to(dev, name), self.spec)
        return t

    fn copy_into(self, inout dst_tensor: Self) raises:
        if dst_tensor.spec != self.spec:
            raise "src and dst tensor specs do not match"
        self._storage.copy_into(dst_tensor._storage)

    fn move_to(owned self, dev: Device) raises -> Self:
        if dev == self._storage._device:
            return self^
        else:
            return self.copy_to(dev)

    fn unsafe_ptr(self) -> UnsafePointer[UInt8]:
        return self._storage.unsafe_ptr()

    fn to_tensor[
        type: DType, rank: Int
    ](owned self) raises -> Tensor[type, rank]:
        if rank != self.spec.rank():
            raise "requested rank does not match existing rank"

        if type != self.spec.dtype():
            raise "requested dtype does not match existing type."

        return Tensor[type, rank](self^)

    fn device(self) -> Device:
        return self._storage.device()

    fn name(self) -> Optional[String]:
        return self._storage.name

    fn __copyinit__(inout self, existing: Self):
        # This temporarily exists so that we can store DeviceMemory in a List
        # TODO(MSTDL-467): Once Copyable requirement on List is removed, this
        # can be removed
        constrained[
            False, "__copyinit__ not supported on DeviceTensor, MSTDL-467"
        ]()
        self._storage = existing._storage
        self.spec = existing.spec

    fn __moveinit__(inout self, owned existing: Self):
        self._storage = existing._storage^
        self.spec = existing.spec^

    fn __str__(self) raises -> String:
        """Returns a description of the DeviceMemory."""

        return (
            "DeviceTensor("
            + (self._storage.name.value() + "," if self._storage.name else "")
            + str(self.device())
            + ",Spec("
            + str(self.spec)
            + "))"
        )

    fn bytecount(self) -> Int:
        return self.spec.bytecount()

    fn take(inout self) -> Self:
        """The returned value takes self's resources and replaces them with default
        initialized values."""
        var tmp = Self()
        swap(tmp, self)
        return tmp
