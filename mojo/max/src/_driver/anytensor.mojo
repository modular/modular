# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from .device_memory import DeviceMemory
from .tensor import Tensor
from .device import Device
from max.tensor import TensorSpec


struct AnyTensor:
    """A type erased tensor representation that is useful
    for situations where we need variadics of tensors."""

    var _data: DTypePointer[DType.uint8]
    var _spec: TensorSpec
    var _device: Device
    var _name: Optional[String]
    var _device_memory_impl_ptr: UnsafePointer[NoneType]

    fn __init__(inout self) raises:
        self._device = Device()
        self._spec = TensorSpec(DType.uint8, 0)
        self._name = None
        self._data = DTypePointer[DType.uint8]()
        self._device_memory_impl_ptr = UnsafePointer[NoneType]()

    fn __init__(inout self, owned device_tensor: DeviceTensor):
        self._device = device_tensor.device()
        self._spec = device_tensor.spec
        self._name = device_tensor.name()
        self._data = device_tensor.unsafe_ptr()
        var tmp = device_tensor^
        var tmp_dm = tmp._storage^
        tmp._storage = DeviceMemory()
        self._device_memory_impl_ptr = tmp_dm^._steal_impl_ptr()

    fn __copyinit__(inout self, existing: Self):
        constrained[False, "AnyTensor is non-copyable"]()
        self._device = existing._device
        self._spec = existing._spec
        self._name = existing._name
        self._data = existing._data
        self._device_memory_impl_ptr = existing._device_memory_impl_ptr

    fn __moveinit__(inout self, owned existing: Self):
        self._device = existing._device^
        self._spec = existing._spec^
        self._name = existing._name^
        self._data = existing._data
        self._device_memory_impl_ptr = existing._device_memory_impl_ptr

    fn __init__[
        type: DType, rank: Int
    ](inout self, owned tensor: Tensor[type, rank]) raises:
        self = Self(tensor^.to_device_tensor())

    fn get_rank(self) -> Int:
        """Gets rank of the tensor.

        Returns:
            Rank of the tensor.
        """
        return self._spec.rank()

    fn _steal_ptr(owned self) -> DTypePointer[DType.uint8]:
        var ptr = self._data
        self._data = DTypePointer[DType.uint8]()
        return ptr

    fn to_device_tensor(owned self) raises -> DeviceTensor:
        var spec = self._spec
        return DeviceTensor(DeviceMemory(self^), spec)

    fn take(inout self) raises -> Self:
        """The returned value takes self's resources and replaces them with default
        initialized values."""
        var tmp = Self()
        swap(self, tmp)
        return tmp

    fn __del__(owned self):
        _ = DeviceMemory(
            self._device_memory_impl_ptr, self._spec.bytecount(), self._device
        )
