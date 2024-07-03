# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from max.graph import Graph
from max._utils import call_dylib_func
from sys.ffi import _get_global_or_null
from max.tensor import TensorSpec

from .anytensor import AnyTensor
from .device import Device
from .device_memory import DeviceMemory, DeviceTensor
from ._status import Status, _CStatus


@value
@register_passable("trivial")
struct _CCompiledGraph:
    var ptr: UnsafePointer[NoneType]


@value
@register_passable("trivial")
struct _CExecutableGraph:
    var ptr: UnsafePointer[NoneType]


struct ExecutableGraph:
    """Represents a compiled graph loaded to device and ready for execution."""

    var _impl: _CExecutableGraph
    var _device: Device

    fn __init__(inout self, impl: _CExecutableGraph, device: Device):
        self._impl = impl
        self._device = device

    fn __moveinit__(inout self, owned existing: Self):
        self._impl = existing._impl
        self._device = existing._device^

    fn _steal_device_memory_impl_ptr(
        self, inout memory: AnyTensor
    ) raises -> UnsafePointer[NoneType]:
        """This takes `memory` as inout and not owned because it is called on
        References owned by a List (returned by List.__getitem__()).
        """
        var taken_memory = memory.take()

        var tmp_device_tensor = taken_memory^.to_device_tensor()
        var taken_device_memory = tmp_device_tensor._storage.take()

        var ptr = taken_device_memory^._steal_impl_ptr()
        return ptr

    fn _add_to_output_list(
        self,
        list: UnsafePointer[NoneType],
        device_memory_ptr: UnsafePointer[NoneType],
        spec_ptr: UnsafePointer[NoneType],
    ) raises:
        var spec = spec_ptr.bitcast[TensorSpec]()[]
        var device_memory = DeviceTensor(
            DeviceMemory(device_memory_ptr, spec.bytecount(), self._device),
            spec,
        )
        var typed_list = list.bitcast[List[AnyTensor]]()
        typed_list[].append(device_memory^)

    fn __del__(owned self):
        var lib = self._device.lib
        call_dylib_func(lib.get_handle(), "M_destroyGraph", self._impl)
        _ = self._device^


struct CompiledGraph:
    """Represents graph that is compiled to a device architecture."""

    var _impl: _CCompiledGraph
    var _device: Device

    fn __init__(inout self, impl: _CCompiledGraph, owned device: Device):
        self._impl = impl
        self._device = device^

    fn __del__(owned self):
        var lib = self._device.lib
        call_dylib_func(lib.get_handle(), "M_destroyCompiledGraph", self._impl)
        _ = self._device^
