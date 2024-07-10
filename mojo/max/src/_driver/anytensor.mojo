# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from .device_memory import DeviceMemory
from .tensor import Tensor
from .device import Device
from max.tensor import TensorSpec
from max._utils import exchange


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

    fn spec(self) -> TensorSpec:
        return self._spec

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


@value
@register_passable("trivial")
struct _CMojoValue:
    var _ptr: UnsafePointer[NoneType]

    alias _destroy_func_type = fn (UnsafePointer[NoneType]) -> None
    var _destroy_func: Self._destroy_func_type

    fn __init__(inout self):
        self._ptr = UnsafePointer[NoneType]()
        self._destroy_func = Self._destroy_pointee_wrapper[NoneType]

    fn __init__[T: Movable](inout self, ptr: UnsafePointer[T]):
        self._ptr = ptr.bitcast[NoneType]()
        self._destroy_func = Self._destroy_pointee_wrapper[T]

    @staticmethod
    fn _destroy_pointee_wrapper[T: AnyType](ptr: UnsafePointer[NoneType]):
        ptr.bitcast[T]().destroy_pointee()

    @staticmethod
    fn _free(ptr: UnsafePointer[NoneType]):
        external_call["KGEN_CompilerRT_MojoValueFreeBuffer", NoneType](ptr)

    fn destroy(self):
        if self._ptr:
            self._destroy_func(self._ptr)
            self._free(self._ptr)


struct AnyMojoValue:
    """Type erased representation of a mojo object. This is useful for passing
    opaque type as input for graph executution."""

    alias c_type = _CMojoValue

    var _impl: Self.c_type

    fn __init__(inout self):
        self._impl = _CMojoValue()

    fn __init__[T: Movable](inout self, owned val: T):
        var ptr = external_call[
            "KGEN_CompilerRT_MojoValueAllocateBuffer", UnsafePointer[T]
        ](sizeof[T](), alignof[T]())
        ptr.init_pointee_move(val^)
        self._impl = _CMojoValue(ptr)

    fn __copyinit__(inout self, existing: Self):
        constrained[False, "AnyMojoValue is not copyable"]()
        self._impl = existing._impl

    fn __moveinit__(inout self, owned existing: Self):
        self._impl = existing._impl

    fn take(inout self) -> Self:
        """Returns the current value and initializes this object to default state.
        """
        var tmp = Self()
        swap(tmp, self)
        return tmp^

    fn release(owned self) -> Self.c_type:
        """Release the underlying Mojo Value pointer. Caller is responsible for
        destroying the object."""
        var impl = exchange(self._impl, _CMojoValue())
        return impl

    fn __del__(owned self):
        self._impl.destroy()


struct AnyMemory:
    """A generic representation which can either be a Driver Tensor or Mojo object.
    """

    var _value: Variant[AnyTensor, AnyMojoValue]

    fn __init__(inout self, owned device_tensor: DeviceTensor):
        self._value = AnyTensor(device_tensor^)

    fn __init__[
        type: DType, rank: Int
    ](inout self, owned tensor: Tensor[type, rank]) raises:
        self._value = AnyTensor(tensor^)

    fn __init__(inout self, owned tensor: AnyTensor):
        self._value = tensor^

    fn __init__(inout self, owned value: AnyMojoValue):
        self._value = value^

    fn is_tensor(self) -> Bool:
        """Check whether this contains a tensor."""
        return self._value.isa[AnyTensor]()

    fn take_tensor(inout self) raises -> AnyTensor:
        """Take tensor from object. Further access to this object is undefined behavior.
        """
        return self._value[AnyTensor].take()

    fn take_value(inout self) -> AnyMojoValue:
        """Take value from object. Further access to this object is undefined behavior.
        """
        return self._value[AnyMojoValue].take()
