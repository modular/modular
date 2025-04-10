# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Defines tensor type, which is an owned, indexible buffer allocated on a given
device.

For example, a tensor can be created and used like this:

```mojo
from max.driver import Tensor
from max.tensor import TensorShape

def main():
    tensor = Tensor[DType.float32, rank=2](TensorShape(1, 2))
    tensor[0, 0] = 1.0
```

"""

from buffer.dimlist import DimList
from collections import InlineArray, Optional

from layout import IntTuple, Layout, LayoutTensor, RuntimeLayout
from max._tensor_utils import _indexing
from max.tensor import Tensor as OldTensor
from max.tensor import TensorShape, TensorSpec
from memory import UnsafePointer

from utils import IndexList
from utils._serialize import _serialize

from testing import Testable

from ._utils import _convert_from
from .device import Device, DeviceMemory, DeviceTensor


struct Tensor[type: DType, rank: Int](CollectionElement, Testable):
    """An owned, indexible buffer type."""

    var _ptr: UnsafePointer[Scalar[type]]
    var _spec: RuntimeTensorSpec[type, rank]
    var _strides: IndexList[rank]
    var _device: Device
    var name: Optional[String]

    # TODO: We should be able to hold DeviceMemory here. Revisit
    # after DeviceMemory/DeviceTensor work.
    # this is needed because DeviceMemory may have a custom free
    # function set on the cpp side.
    var _device_memory_impl_ptr: UnsafePointer[NoneType]

    alias layout_tensor = LayoutTensor[
        type,
        Layout(
            IntTuple(DimList.create_unknown[rank]()),
            IntTuple(DimList.create_unknown[rank]()),
        ),
        MutableAnyOrigin,
    ]
    """The corresponding layout tensor type which acts as a structured view
    into the underlying data."""

    fn __init__(out self) raises:
        """Default constructor for Tensor. Accessing the elements of default
        constructed tensor is undefined behavior.
        """
        self._ptr = UnsafePointer[Scalar[type]]()
        self._spec = RuntimeTensorSpec[type, rank](IndexList[rank]())
        self._strides = IndexList[rank]()
        self._device = Device()
        self.name = None
        self._device_memory_impl_ptr = UnsafePointer[NoneType]()

    fn __init__(out self, *, owned device_tensor: DeviceTensor) raises:
        """Creates a tensor from DeviceTensor.

        Args:
            device_tensor: DeviceTensor to create tensor from.
        """
        self._device = device_tensor.device()
        self.name = device_tensor.name()
        self._spec = device_tensor.spec
        self._strides = _indexing._row_major_strides(self._spec)
        self._ptr = device_tensor.unsafe_ptr().bitcast[Scalar[type]]()
        var tmp = device_tensor._storage^
        device_tensor._storage = DeviceMemory()
        self._device_memory_impl_ptr = tmp^._steal_impl_ptr()

    fn __init__(
        out self, shape: TensorShape, device: Optional[Device] = None
    ) raises:
        """Creates tensor with given shape on the given device. If device is
        not given tensor will be created on cpu.

        Args:
            shape: Shape of the tensor.
            device: Device on which tensor is to be allocated.
        """
        var spec = TensorSpec(type, shape)
        var dev = device.value() if device else cpu()
        var dt = dev.allocate(spec)
        self = Self(device_tensor=dt)

    fn __init__(out self, tensor: OldTensor[type]) raises:
        """Converts max.tensor to max.driver.Tensor. This creates tensor on
        the CPU.

        Args:
            tensor: Tensor to copy from.
        """
        self = _convert_from[rank=rank](tensor)

    fn __moveinit__(out self, owned existing: Self):
        """Move constructor for Tensor.

        Args:
            existing: Instance to move from.
        """
        self._ptr = existing._ptr
        self._spec = existing._spec
        self._strides = existing._strides
        self._device = existing._device^
        self.name = existing.name^
        self._device_memory_impl_ptr = existing._device_memory_impl_ptr

    @doc_private
    fn __copyinit__(out self, existing: Self):
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

    @always_inline
    fn copy(self) -> Self:
        """Explicitly construct a copy of self.

        Returns:
            A copy of this value.
        """
        return self

    fn spec(self) -> RuntimeTensorSpec[type, rank]:
        """Gets the spec of tensor.

        Returns
            Spec of the tensor.
        """
        return self._spec

    @always_inline
    fn __getitem__(mut self, *indices: Int) -> ref [self] Scalar[type]:
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
            return "cpu" in String(self._device)

        debug_assert[_is_cpu](
            "Cannot index into non-CPU Tensor from host",
        )

        var offset = _indexing._dot_prod(indices, self._strides)
        return self._ptr[offset]

    @always_inline
    fn to_layout_tensor(
        self,
        out result: Self.layout_tensor,
    ) raises:
        """Returns a view of the tensor conforming to given slices. If given
        a single slice `:` the view would point to the entire tensor. The caller
        is responsible to make sure tensor outlives the returned slice.

        Args:

        Returns:
            View of the tensor according to given slices.
        """
        return __type_of(result)(
            self.unsafe_ptr(),
            __type_of(result.runtime_layout)(self._spec.shape, self._strides),
        )

    fn _steal_ptr(owned self) -> UnsafePointer[Scalar[type]]:
        var tmp = self._ptr
        self._ptr = UnsafePointer[Scalar[type]]()
        return tmp

    fn _get_device(self) -> Device:
        return self._device

    fn to_device_tensor(owned self) raises -> DeviceTensor:
        """Converts the tensor to a DeviceTensor.

        Returns:
            DeviceTensor pointing to the memory owned by tensor.
        """
        var spec = self.spec()
        return DeviceTensor(DeviceMemory(self^), TensorSpec(spec))

    fn __del__(owned self):
        """Destructor for the tensor."""
        _ = DeviceMemory(
            self._device_memory_impl_ptr,
            self.spec().bytecount(),
            self._device,
        )

    fn unsafe_ptr[__type: DType = type](self) -> UnsafePointer[Scalar[__type]]:
        """Gets a pointer to the underlying memory.

        Note: The caller is responsible for ensuring that the returned pointer
        is not used after it's owner is last used.

        Parameters:
            __type: If given the pointer will be rebound to this type. Defaulted
                    to type of tensor.

        Returns:
           Pointer to the beginning of tensor data.
        """
        return rebind[UnsafePointer[Scalar[__type]]](self._ptr)

    fn take(mut self) raises -> Self:
        """Takes self's resources and replaces them with default
        initialized values.

        Returns:
            An instance of tensor.
        """
        var tmp = Self()
        swap(tmp, self)
        return tmp

    @no_inline
    fn __str__(self) -> String:
        """Gets the tensor as a string.

        Returns:
          A compact string of the tensor.
        """

        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        """
        Formats this Tensor to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
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

        var device_str = String(self._device)
        if "cpu" not in device_str:
            writer.write("<Unable to print device tensor>, ")
            writer.write(device_str)
            writer.write(", ")
            write_dtype_and_shape()
            writer.write(")")
            return

        @parameter
        fn serialize[T: Writable](val: T):
            writer.write(val)

        var shape = List[Int, hint_trivial_type=True]()
        for i in range(self._spec.rank):
            shape.append(self._spec.shape[i])

        _serialize[serialize_fn=serialize, serialize_end_line=False](
            self._ptr, shape
        )
        writer.write(")")

    fn move_to(owned self, device: Device) raises -> Self:
        """Returns self if already allocated on device, otherwise copy the contents
        of self to device.

        Args:
            device: The Device of the returned buffer.

        Returns:
            Instance of Tensor allocated on given device.
        """
        return self^.to_device_tensor().move_to(device).to_tensor[type, rank]()

    fn __eq__(self, other: Self) -> Bool:
        """Check if two tensors are equal. Note that only host tensors can be
        compared. If either tensor is on an accelerator device, the result is False.

        Args:
            other: The tensor to compare with.

        Returns:
            True if the tensors have the same shape and all elements are equal, False otherwise.
        """

        # First, check that both tensors on on the host device. We can't compare
        # tensors on accelerator devices.
        if not (
            "cpu" in String(self._device) and "cpu" in String(other._device)
        ):
            return False

        # Next check if they point to the same memory
        if self._ptr == other._ptr:
            return True

        # Check if shapes match
        if self._spec.shape != other._spec.shape:
            return False

        # Now compare element by element
        var self_ptr = self.unsafe_ptr()
        var other_ptr = other.unsafe_ptr()
        var size = self._spec.shape[0] * self._spec.shape[1]
        for i in range(size):
            if self_ptr[i] != other_ptr[i]:
                return False
        return True

    fn __ne__(self, other: Self) -> Bool:
        """Check if two tensors are not equal. Note that only host tensors can be
        compared. If either tensor is on an accelerator device, the result is True.

        Args:
            other: The tensor to compare with.

        Returns:
            True if the tensors are not equal, False otherwise.
        """
        return not self == other
