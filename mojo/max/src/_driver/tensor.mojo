# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from .device import Device, DeviceMemory, DeviceTensor
from .tensor_slice import TensorSlice, UnsafeTensorSlice
from utils import InlineArray
from max.tensor import TensorSpec, TensorShape


@always_inline
fn _dot_prod[
    rank: Int
](x: StaticIntTuple[rank], y: StaticIntTuple[rank]) -> Int:
    var offset = 0

    @parameter
    for i in range(rank):
        offset += x[i] * y[i]
    return offset


@always_inline
fn _slice_to_tuple[
    func: fn (Slice) capturing -> Int, rank: Int
](slices: InlineArray[Slice, rank]) -> StaticIntTuple[rank]:
    """Takes a tuple of `Slice`s and returns a tuple of Ints.
    `func` is used to extract the appropriate field (i.e. start, stop or end)
    of the Slice.
    """
    var tuple = StaticIntTuple[rank]()

    @parameter
    for i in range(rank):
        tuple[i] = func(slices[i])
    return tuple


@always_inline
fn _row_major_strides[
    type: DType, rank: Int
](spec: StaticTensorSpec[type, rank]) -> StaticIntTuple[rank]:
    var offset = 1
    var strides = StaticIntTuple[rank]()

    @parameter
    for i in range(rank - 1, -1, -1):
        strides[i] = offset
        offset *= spec.shape[i]
    return strides


@value
@register_passable
struct StaticTensorSpec[type: DType, rank: Int]:
    var shape: StaticIntTuple[rank]

    fn __init__(inout self, spec: TensorSpec):
        """Construct from TensorSpec.

        Args:
            spec: TensorSpec of given rank and type.
        """
        debug_assert(spec.rank() == rank, "rank mismatch")
        debug_assert(spec.dtype() == type, "dtype mismatch")
        self.shape = StaticIntTuple[rank]()

        for i in range(rank):
            self.shape[i] = spec[i]

    fn get_tensor_spec(self) -> TensorSpec:
        var shapes = List[Int]()
        shapes.reserve(rank)
        for i in range(rank):
            shapes.append(self[i])
        return TensorSpec(type, shapes)

    fn __getitem__(self, idx: Int) -> Int:
        return self.shape[idx]


trait TensorLike:
    fn spec(self) -> TensorSpec:
        ...

    fn unsafe_ptr[type: DType](self) -> DTypePointer[type]:
        pass


struct Tensor[type: DType, rank: Int](CollectionElement, TensorLike):
    var _ptr: DTypePointer[type]
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
        self._ptr = DTypePointer[type]()
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

    fn __init__(inout self, shape: TensorShape) raises:
        var device = cpu_device()
        var spec = TensorSpec(type, shape)
        var dt = device.allocate(spec)
        self = Self(dt)

    fn __init__(inout self, shape: Tuple, device: Device) raises:
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
        debug_assert(
            "CPU" in str(self._device),
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
        debug_assert(
            "CPU" in str(self._device),
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
        debug_assert(
            "CPU" in str(self._device),
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
        debug_assert(
            "CPU" in str(self._device),
            "Cannot index into non-CPU Tensor from host",
        )
        SIMD[size=width].store(
            self._ptr, _dot_prod(indices, self._strides), val
        )

    fn _steal_ptr(owned self) -> DTypePointer[type]:
        var tmp = self._ptr
        self._ptr = DTypePointer[type]()
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

    fn unsafe_ptr[__type: DType = type](self) -> DTypePointer[__type]:
        """Returns a pointer to the underlying memory.

        Note: The caller is responsible for ensuring that the returned pointer
        is not used after it's owner is last used.
        """
        return rebind[DTypePointer[__type]](self._ptr)

    fn take(inout self) raises -> Self:
        """The returned value takes self's resources and replaces them with default
        initialized values."""
        var tmp = Self()
        swap(tmp, self)
        return tmp
