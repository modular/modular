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
from ._driver_library import DriverLibrary, ManagedDLHandle
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

    fn copy(self, lib: DriverLibrary) -> Self:
        alias func_name_copy = "M_copyDevice"
        return call_dylib_func[UnsafePointer[NoneType]](
            lib.get_handle(), func_name_copy, self
        )

    fn free_data(self, lib: DriverLibrary, data: UnsafePointer[UInt8]):
        alias func_free_data = "M_freeDeviceData"
        call_dylib_func(lib.get_handle(), func_free_data, self, data)

    fn __eq__(self, other: Self) -> Bool:
        return self._ptr == other._ptr


struct Device(Stringable):
    var lib: DriverLibrary
    var _cdev: _CDevice

    fn __init__(inout self, descriptor: CPUDescriptor = CPUDescriptor()) raises:
        """Creates a CPU Device from a CPUDescriptor."""

        alias func_name_create = "M_createCPUDevice"
        self.lib = ManagedDLHandle(DLHandle(_get_driver_path()))
        self._cdev = _CDevice(
            call_dylib_func[UnsafePointer[NoneType]](
                self.lib.get_handle(),
                func_name_create,
                Int32(descriptor.numa_id),
            )
        )

    fn __init__(
        inout self, lib: DriverLibrary, *, owned owned_ptr: _CDevice
    ) raises:
        """Creates a CPU Device from an opaque _CDevice object. Not intended for
        external use."""

        self.lib = lib
        self._cdev = owned_ptr

    fn __copyinit__(inout self, existing: Self):
        """Copy constructor for device.
        Args:
            existing(Device): Instance from which to copy.
        """

        self.lib = existing.lib
        self._cdev = existing._cdev.copy(self.lib)

    fn __moveinit__(inout self, owned existing: Self):
        self.lib = existing.lib^
        self._cdev = existing._cdev

    fn allocate(
        self, spec: TensorSpec, name: Optional[String] = None
    ) raises -> DeviceTensor:
        """Returns a DeviceMemory allocated in the Device's memory space.
        DeviceMemory holds a reference count to Device.
        """

        return DeviceTensor(spec, self, name)

    fn allocate(
        self, bytecount: Int, name: Optional[String] = None
    ) raises -> DeviceMemory:
        """Returns a DeviceMemory allocated in the Device's memory space.
        DeviceMemory holds a reference count to Device.
        """

        return DeviceMemory(bytecount, self, name)

    fn _free(self, data: UnsafePointer[UInt8]):
        self._cdev.free_data(self.lib, data)

    fn __str__(self) -> String:
        """Returns a descriptor of the device."""

        alias func_name_desc = "M_getDeviceDesc"
        return StringRef(
            call_dylib_func[UnsafePointer[UInt8]](
                self.lib.get_handle(), func_name_desc, self._cdev
            )
        )

    fn __del__(owned self):
        """Decrements the refcount to Device and destroys it if this object holds
        the only reference."""

        alias func_name_destroy = "M_destroyDevice"
        call_dylib_func[NoneType](
            self.lib.get_handle(), func_name_destroy, self._cdev
        )
        # Extend lifetime of library until C function returns.
        _ = self.lib^

    fn __eq__(self, other: Self) -> Bool:
        return self._cdev == other._cdev

    fn compile(self, graph: Graph) raises -> CompiledGraph:
        """Compiles graph to the given device.

        Args:
            graph: Graph to be compiled.

        Returns:
            Compiled graph ready to be loaded and executed.
        """
        # Graph compiler shares the context with Mojo. This context
        # contains AsyncRT Runtime, Telemetery etc.
        var max_context = _get_global_or_null["MaxContext"]().address
        var status = Status(self.lib)
        var compiled_ptr = call_dylib_func[_CCompiledGraph](
            self.lib.get_handle(),
            "M_compileGraph",
            graph._module().c.ptr,
            self._cdev,
            max_context,
            status.impl,
        )
        if status:
            raise str(status)
        return CompiledGraph(compiled_ptr, self)

    fn load(self, compiled_graph: CompiledGraph) raises -> ExecutableGraph:
        """Load and initialize compiled graph. This will run the setup function
        of graph which includes initializing constants, loading constants into
        device of choice etc.

        Arguments:
            compiled_graph: Compiled graph returned by Device.compile()

        Returns:
            Model ready for execution.
        """
        var status = Status(self.lib)
        var executable_ptr = call_dylib_func[_CExecutableGraph](
            self.lib.get_handle(),
            "M_loadGraph",
            compiled_graph._impl,
            self._cdev,
            status.impl,
        )
        if status:
            raise str(status)
        return ExecutableGraph(executable_ptr, self)


fn cpu_device(descriptor: CPUDescriptor = CPUDescriptor()) raises -> Device:
    return Device(descriptor)
