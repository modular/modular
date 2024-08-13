# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from collections import Optional
from sys.ffi import DLHandle, _get_global_or_null
from max._utils import call_dylib_func, get_lib_path_from_cfg
from pathlib import Path
from max.tensor import TensorSpec
from ._driver_library import DriverLibrary
from .device_memory import DeviceMemory, DeviceTensor
from .graph import CompiledGraph, _CCompiledGraph, _CExecutableGraph
from max.graph import Graph
from ._status import Status, _CStatus
from .utils import _steal_device_memory_impl_ptr


struct CPUDescriptor:
    var numa_id: Int

    fn __init__(inout self, *, numa_id: Optional[Int] = None):
        self.numa_id = numa_id.value() if numa_id else -1


fn _get_driver_path() raises -> String:
    return get_lib_path_from_cfg(".driver_lib", "MAX Driver")


@value
@register_passable("trivial")
struct _CDevice:
    var _ptr: UnsafePointer[NoneType]

    fn copy(self, lib: Optional[DriverLibrary]) -> Self:
        if not lib:
            return self
        return lib.value().copy_device_fn(self._ptr)

    fn free_data(self, lib: DriverLibrary, data: UnsafePointer[UInt8]):
        lib.free_device_data_fn(self._ptr, data)

    fn __eq__(self, other: Self) -> Bool:
        return self._ptr == other._ptr


struct Device(Stringable):
    var lib: Optional[DriverLibrary]
    var _cdev: _CDevice

    fn __init__(inout self):
        """Constructs a default initialized Device in a state that is only valid
        for deletion. Can be used to represent a 'moved from' state.


        Use cpu_device() or cuda_device() to create a CPU or GPU Device.
        """

        self.lib = None
        self._cdev = _CDevice(UnsafePointer[NoneType]())

    @doc_private
    fn __init__(
        inout self, lib: DriverLibrary, *, owned owned_ptr: _CDevice
    ) raises:
        self.lib = lib
        self._cdev = owned_ptr

    fn __copyinit__(inout self, existing: Self):
        """Create a copy of the Device (bumping a refcount on the underlying Device).

        Args:
            existing: Instance from which to copy.
        """

        self.lib = existing.lib
        self._cdev = existing._cdev.copy(existing.lib)

    fn __moveinit__(inout self, owned existing: Self):
        """Create a new Device and consume `existing`.

        Args:
            existing: Instance from which to move from.
        """
        self.lib = existing.lib^
        self._cdev = existing._cdev

    fn allocate(
        self, spec: TensorSpec, name: Optional[String] = None
    ) raises -> DeviceTensor:
        """Returns a DeviceTensor allocated in the Device's address space.

        Args:
            spec: TensorSpec descripting the shape and type of the tensor to allocate.
            name: An optional name for the DeviceTensor.
        """

        return DeviceTensor(spec, self, name)

    fn allocate(
        self, bytecount: Int, name: Optional[String] = None
    ) raises -> DeviceMemory:
        """Allocates a DeviceMemory object in the Device's address space.

        Args:
            bytecount: The size of the memory to allocate in bytes.
            name: An optional name for the DeviceMemory.

        Returns:
            A DeviceMemory object allocated in the Device's address space.
        """

        return DeviceMemory(bytecount, self, name)

    fn _free(self, data: UnsafePointer[UInt8]):
        self._cdev.free_data(self.lib.value(), data)

    fn __str__(self) -> String:
        """Returns a descriptor of the device."""

        return StringRef(self.lib.value().get_device_desc_fn(self._cdev._ptr))

    fn __del__(owned self):
        """Destroys the device.

        Note that any DeviceBuffer allocated on the Device will contain a reference
        to the Device, and the Device will only be de-allocated when all of its
        DeviceBuffers have also been destroyed.
        """

        if not self._cdev._ptr:
            return
        self.lib.value().destroy_device_fn(self._cdev._ptr)

    fn __eq__(self, other: Self) -> Bool:
        """Check if `self` and `other` point to the same underlying Device."""
        return self._cdev == other._cdev

    @doc_private
    fn compile(self, graph: Graph) raises -> CompiledGraph:
        """TODO (MSDK-731): Removing in favour of engine APIs."""

        # contains AsyncRT Runtime, Telemetery etc.
        var max_context = _get_global_or_null["MaxContext"]().address
        var status = Status(self.lib.value())
        var compiled_ptr = call_dylib_func[_CCompiledGraph](
            self.lib.value().get_handle(),
            "M_compileGraph",
            graph._module().c.ptr,
            self._cdev,
            max_context,
            status.impl,
        )
        if status:
            raise str(status)
        return CompiledGraph(compiled_ptr, self)

    @doc_private
    fn load(self, compiled_graph: CompiledGraph) raises -> ExecutableGraph:
        """TODO (MSDK-731): Removing in favour of engine APIs."""

        var status = Status(self.lib.value())
        var executable_ptr = call_dylib_func[_CExecutableGraph](
            self.lib.value().get_handle(),
            "M_loadGraph",
            compiled_graph._impl,
            self._cdev,
            status.impl,
        )
        if status:
            raise str(status)
        return ExecutableGraph(executable_ptr, self)


fn cpu_device(descriptor: CPUDescriptor = CPUDescriptor()) raises -> Device:
    """Creates a CPU Device from a CPUDescriptor."""
    var lib = DriverLibrary()
    return Device(lib, owned_ptr=lib.create_cpu_device_fn(descriptor.numa_id))
