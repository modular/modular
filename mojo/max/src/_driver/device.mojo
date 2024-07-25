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
        """Creates a default-initialized Device."""

        self.lib = None
        self._cdev = _CDevice(UnsafePointer[NoneType]())

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
        self._cdev = existing._cdev.copy(existing.lib)

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
        self._cdev.free_data(self.lib.value(), data)

    fn __str__(self) -> String:
        """Returns a descriptor of the device."""

        return StringRef(self.lib.value().get_device_desc_fn(self._cdev._ptr))

    fn __del__(owned self):
        """Decrements the refcount to Device and destroys it if this object holds
        the only reference."""

        if not self._cdev._ptr:
            return
        self.lib.value().destroy_device_fn(self._cdev._ptr)

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

    fn load(self, compiled_graph: CompiledGraph) raises -> ExecutableGraph:
        """Load and initialize compiled graph. This will run the setup function
        of graph which includes initializing constants, loading constants into
        device of choice etc.

        Arguments:
            compiled_graph: Compiled graph returned by Device.compile()

        Returns:
            Model ready for execution.
        """
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
