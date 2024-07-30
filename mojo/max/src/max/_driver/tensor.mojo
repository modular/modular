# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from .device import Device, DeviceMemory, DeviceTensor
from .tensor_slice import TensorSlice
from max.tensor import TensorSpec, TensorShape
from max._tensor_utils import UnsafeTensorSlice, TensorLike
from tensor_utils.indexing import (
    _dot_prod,
    _row_major_strides,
)
from collections import Optional
from utils import InlineArray, StaticIntTuple
from utils._serialize import _serialize


struct Tensor[type: DType, rank: Int](CollectionElement, TensorLike):
    var _ptr: UnsafePointer[Scalar[type]]
    var _spec: StaticTensorSpec[type, rank]
    var _strides: StaticIntTuple[rank]
    var _device: Device
    var name: Optional[String]

    # TODO: We should be able to hold DeviceMemory here. Revisit
    # after DeviceMemory/DeviceTensor work.
    # this is needed because DeviceMemory may have a custom free
    # function set on the cpp side.
    var _device_memory_impl_ptr: UnsafePointer[NoneType]

    fn __init__(inout self) raises:
        self._ptr = UnsafePointer[Scalar[type]]()
        self._spec = StaticTensorSpec[type, rank](StaticIntTuple[rank]())
        self._strides = StaticIntTuple[rank]()
        self._device = Device()
        self.name = None
        self._device_memory_impl_ptr = UnsafePointer[NoneType]()

    fn __init__(inout self, owned device_tensor: DeviceTensor) raises:
        self._device = device_tensor.device()
        self.name = device_tensor.name()
        self._spec = device_tensor.spec
        self._strides = _row_major_strides(self._spec)
        self._ptr = device_tensor.unsafe_ptr().bitcast[type]()
        var tmp = device_tensor._storage^
        device_tensor._storage = DeviceMemory()
        self._device_memory_impl_ptr = tmp^._steal_impl_ptr()

    fn __init__(inout self, shape: Tuple) raises:
        var device = cpu_device()
        var spec = TensorSpec(type, shape)
        var dt = device.allocate(spec)
        self = Self(dt)

    fn __init__(inout self, shape: Tuple, device: Device) raises:
        var spec = TensorSpec(type, shape)
        var dt = device.allocate(spec)
        self = Self(dt)

    fn __init__(inout self, shape: TensorShape) raises:
        var device = cpu_device()
        var spec = TensorSpec(type, shape)
        var dt = device.allocate(spec)
        self = Self(dt)

    fn __init__(inout self, shape: TensorShape, device: Device) raises:
        var spec = TensorSpec(type, shape)
        var dt = device.allocate(spec)
        self = Self(dt)

    fn __moveinit__(inout self, owned existing: Self):
        self._ptr = existing._ptr
        self._spec = existing._spec^
        self._strides = existing._strides
        self._device = existing._device^
        self.name = existing.name^
        self._device_memory_impl_ptr = existing._device_memory_impl_ptr

    fn __copyinit__(inout self, existing: Self):
        # This temporarily exists so that we can store Tensor in a List
        # TODO(MSTDL-467): Once Copyable requirement on List is removed, this
        # can be removed
        constrained[False, "__copyinit__ not supported on Tensor, MSTDL-467"]()
        self._ptr = existing._ptr
        self._spec = existing._spec
        self._strides = existing._strides
        self._device = existing._device
        self.name = existing.name
        self._device_memory_impl_ptr = existing._device_memory_impl_ptr

    fn spec(self) -> TensorSpec:
        return self._spec.get_tensor_spec()

    @always_inline
    fn __getitem__(
        inout self, *indices: Int
    ) -> ref [__lifetime_of(self)] Scalar[type]:
        """Gets the value at the specified indices.

        Args:
          indices: The indices of the value to retrieve.

        Returns:
          The value at the specified indices.
        """
        debug_assert(
            len(indices) == rank, "mismatch between requested index and rank"
        )

        @always_inline
        @parameter
        fn _is_cpu() -> Bool:
            return "CPU" in str(self._device)

        debug_assert[_is_cpu](
            "Cannot index into non-CPU Tensor from host",
        )

        var offset = _dot_prod(indices, self._strides)
        return self._ptr[offset]

    fn _canonicalize_slices(
        self, slices: VariadicListMem[Slice, _, _]
    ) -> InlineArray[Slice, rank]:
        var slice_array = InlineArray[Slice, rank](unsafe_uninitialized=True)
        for i in range(len(slices)):
            slice_array[i] = slices[i]
            slice_array[i].start = (slice_array[i].start or 0).value()
            slice_array[i].end = (slice_array[i].end or self._spec[i]).value()
        # pads any unspecified Slices with default values
        for i in range(len(slices), rank):
            slice_array[i].start = 0
            slice_array[i].end = self._spec[i]
            slice_array[i].step = 1

        return slice_array

    @always_inline
    fn __getitem__(
        ref [_]self: Self, *slices: Slice
    ) raises -> TensorSlice[type, rank, __lifetime_of(self)]:
        if len(slices) > rank:
            raise "len(slices) exceeds rank"
        return TensorSlice(self, self._canonicalize_slices(slices))

    @always_inline
    fn unsafe_slice(
        self,
        *slices: Slice,
    ) raises -> UnsafeTensorSlice[type, rank]:
        if len(slices) > rank:
            raise "len(slices) exceeds rank"
        return UnsafeTensorSlice[type, rank](
            self.unsafe_ptr(),
            self._canonicalize_slices(slices),
            self._spec,
        )

    @always_inline
    fn load[*, width: Int = 1](self, *indices: Int) -> SIMD[type, width]:
        """Gets the SIMD value at the specified indices.

        Parameters:
          width: The SIMD width of the vector.

        Args:
          indices: The indices of the value to retrieve.

        Returns:
          The SIMD value at the specified indices.
        """
        debug_assert(
            len(indices) == rank, "mismatch between requested index and rank"
        )

        @always_inline
        @parameter
        fn _is_cpu() -> Bool:
            return "CPU" in str(self._device)

        debug_assert[_is_cpu](
            "Cannot index into non-CPU Tensor from host",
        )
        return SIMD[size=width].load(
            self._ptr, _dot_prod(indices, self._strides)
        )

    @always_inline
    fn load[
        *, width: Int = 1
    ](self, indices: StaticIntTuple[rank]) -> SIMD[type, width]:
        """Gets the SIMD value at the specified indices.

        Parameters:
          width: The SIMD width of the vector.

        Args:
          indices: The indices of the value to retrieve.

        Returns:
          The SIMD value at the specified indices.
        """

        @always_inline
        @parameter
        fn _is_cpu() -> Bool:
            return "CPU" in str(self._device)

        debug_assert[_is_cpu](
            "Cannot index into non-CPU Tensor from host",
        )
        return SIMD[size=width].load(
            self._ptr, _dot_prod(indices, self._strides)
        )

    @always_inline
    fn store[
        *, width: Int = 1
    ](inout self, indices: StaticIntTuple[rank], val: SIMD[type, width]):
        """Sets the SIMD value at the specified indices.

        Parameters:
          width: The SIMD width of the vector.

        Args:
          indices: The indices of the value to set.
          val: The SIMD value to store.
        """

        @always_inline
        @parameter
        fn _is_cpu() -> Bool:
            return "CPU" in str(self._device)

        debug_assert[_is_cpu](
            "Cannot index into non-CPU Tensor from host",
        )
        SIMD[size=width].store(
            self._ptr, _dot_prod(indices, self._strides), val
        )

    fn _steal_ptr(owned self) -> UnsafePointer[Scalar[type]]:
        var tmp = self._ptr
        self._ptr = UnsafePointer[Scalar[type]]()
        return tmp

    fn _get_device(self) -> Device:
        return self._device

    fn to_device_tensor(owned self) raises -> DeviceTensor:
        var spec = self._spec.get_tensor_spec()
        return DeviceTensor(DeviceMemory(self^), spec)

    fn __del__(owned self):
        _ = DeviceMemory(
            self._device_memory_impl_ptr,
            self._spec.get_tensor_spec().bytecount(),
            self._device,
        )

    fn unsafe_ptr[__type: DType = type](self) -> UnsafePointer[Scalar[__type]]:
        """Returns a pointer to the underlying memory.

        Note: The caller is responsible for ensuring that the returned pointer
        is not used after it's owner is last used.
        """
        return rebind[UnsafePointer[Scalar[__type]]](self._ptr)

    fn take(inout self) raises -> Self:
        """The returned value takes self's resources and replaces them with default
        initialized values."""
        var tmp = Self()
        swap(tmp, self)
        return tmp

    @no_inline
    fn __str__(self) -> String:
        """Gets the tensor as a string.

        Returns:
          A compact string of the tensor.
        """

        return String.format_sequence(self)

    fn format_to(self, inout writer: Formatter):
        """
        Formats this Tensor to the provided formatter.

        Args:
            writer: The formatter to write to.
        """

        writer.write("Tensor(")

        @parameter
        fn write_dtype_and_shape():
            writer.write("dtype=")
            writer.write(type)
            writer.write(", ")
            writer.write("shape=")
            for i in range(rank):
                if i > 0:
                    writer.write("x")
                writer.write(self._spec.shape[i])

        var device_str = str(self._device)
        if "CPU" not in device_str:
            writer.write("<Unable to print device tensor>, ")
            writer.write(device_str)
            writer.write(", ")
            write_dtype_and_shape()
            writer.write(")")
            return

        @parameter
        fn serialize[T: Formattable](val: T):
            writer.write(val)

        var shape = List[Int]()
        for i in range(self._spec.rank):
            shape.append(self._spec.shape[i])

        _serialize[serialize_fn=serialize, serialize_end_line=False](
            self._ptr, shape
        )
        writer.write(")")
