# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""
Defines different data formats you can use to pass inputs to MAX Engine
when executing a model.

You can pass each of the types shown here to
[`Model.execute()`](/max/api/mojo/engine/model/Model#execute).
"""
from collections import List

from max.tensor import Tensor
from memory import ArcPointer, UnsafePointer
from memory.unsafe import bitcast
from python import Python, PythonObject

from ._tensor_impl import CTensor, _Numpy
from .tensor_spec import TensorSpec


struct _OwningPointer(Movable):
    """A type that deallocates the specified pointer when it is destroyed."""

    var ptr: UnsafePointer[NoneType]

    @implicit
    fn __init__(out self, ptr: UnsafePointer[NoneType]):
        self.ptr = ptr

    fn __moveinit__(out self, owned existing: Self):
        self.ptr = existing.ptr

    fn __del__(owned self):
        self.ptr.free()


@value
struct NamedTensor:
    """A named input tensor."""

    var name: String
    """Name of the tensor."""
    var _tensor_data: ArcPointer[_OwningPointer]
    """Reference-counted pointer keeping the tensor data alive."""
    var _view: EngineTensorView

    fn __init__[
        dtype: DType
    ](out self, owned name: String, owned tensor: Tensor[dtype]):
        """Creates a `NamedTensor` owning the tensor with a reference count.

        Parameters:
            dtype: Data type of the tensor to own.

        Args:
            name: Name of the tensor.
            tensor: Tensor to take ownership of.
        """
        self.name = name^

        # The view takes a pointer to the element data in the tensor, but
        # doesn't extend the lifetime of the tensor buffer data.
        self._view = EngineTensorView(tensor)

        # We want NamedTensor to be copyable but it needs to keep the underlying
        # buffer alive.  Use an ArcPointer[OwningPointer] to keep the underlying data
        # alive and us copyable.  We don't care what `dtype` is, and don't want
        # NamedTensor to have to be generic on `dtype`.
        self._tensor_data = ArcPointer(
            _OwningPointer(tensor._take_data_ptr().bitcast[NoneType]())
        )

        # FIXME(MSDK-230): This is leaking tensors.
        self._tensor_data._inner[].add_ref()


@value
struct EngineTensorView:
    """A non-owning register_passable view of a tensor
    that does runtime type checking.

    CAUTION: Make sure the source tensor outlives the view.
    """

    var _spec: TensorSpec
    var _data_ptr: UnsafePointer[NoneType]
    var _dtype: DType

    @implicit
    fn __init__[type: DType](out self, tensor: Tensor[type]):
        """Creates a non-owning view of given Tensor.

        Parameters:
            type: DType of the tensor.

        Args:
            tensor: Tensor backing the view.
        """
        self._spec = tensor._spec
        self._data_ptr = tensor.unsafe_ptr().bitcast[NoneType]()
        self._dtype = type

    fn data[type: DType](self) raises -> UnsafePointer[Scalar[type]]:
        """Returns pointer to the start of tensor.

        Parameters:
            type: Expected type of tensor.

        Returns:
            UnsafePointer of given type.

        Raises:
            If the given type does not match the type of tensor.
        """
        if type != self._dtype:
            raise String("Expected type: ") + self._dtype.__str__()
        return self._data_ptr.bitcast[Scalar[type]]()

    fn unsafe_ptr(self) -> UnsafePointer[NoneType]:
        """Returns type erased pointer to the start of tensor.

        Returns:
            UnsafePointer of invalid type.
        """
        return self._data_ptr

    fn spec(self) -> TensorSpec:
        """Returns the spec of tensor backing the view.

        Returns:
            Stdlib TensorSpec of the tensor.
        """

        return self._spec


@value
@register_passable
struct EngineNumpyView:
    """A register_passable view of a numpy array.

    Keeps its own reference to the NumPy PythonObject, so there is no need to
    manually keep the Python object alive after construction.
    """

    var _np: _Numpy
    var _obj: PythonObject

    fn __init__(out self, tensor: PythonObject) raises:
        """Creates a non-owning view of given numpy array.

        Args:
            tensor: Numpy Array backing the view.
        """
        self._np = _Numpy()
        self._obj = tensor

    fn unsafe_ptr(self) raises -> UnsafePointer[NoneType]:
        """Returns type erased pointer to the start of numpy array.

        Returns:
            UnsafePointer of given type.
        """
        return rebind[UnsafePointer[NoneType]](
            self._obj.ctypes.data.unsafe_get_as_pointer[DType.invalid]()
        )

    fn dtype(self) raises -> DType:
        """Get DataType of the array backing the view.

        Returns:
            DataType of the array backing the view.
        """
        var self_type = self._obj.dtype
        if self_type == self._np.int8:
            return DType.int8
        if self_type == self._np.int16:
            return DType.int16
        if self_type == self._np.int32:
            return DType.int32
        if self_type == self._np.int64:
            return DType.int64

        if self_type == self._np.uint8:
            return DType.uint8
        if self_type == self._np.uint16:
            return DType.uint16
        if self_type == self._np.uint32:
            return DType.uint32
        if self_type == self._np.uint64:
            return DType.uint64

        if self_type == self._np.float16:
            return DType.float16
        if self_type == self._np.float32:
            return DType.float32
        if self_type == self._np.float64:
            return DType.float64

        raise "Unknown datatype"

    fn spec(self) raises -> TensorSpec:
        """Returns the spec of numpy array backing the view.

        Returns:
            Numpy array spec in format of Stdlib TensorSpec.
        """

        @always_inline
        @parameter
        fn get_spec[ty: DType]() raises -> TensorSpec:
            var shape = List[Int, hint_trivial_type=True]()
            var array_shape = self._obj.shape
            for dim in array_shape:
                shape.append(Int(dim))
            return TensorSpec(ty, shape)

        if self.dtype() is DType.int8:
            return get_spec[DType.int8]()
        if self.dtype() is DType.uint16:
            return get_spec[DType.int16]()
        if self.dtype() is DType.int32:
            return get_spec[DType.int32]()
        if self.dtype() is DType.int64:
            return get_spec[DType.int64]()

        if self.dtype() is DType.uint8:
            return get_spec[DType.uint8]()
        if self.dtype() is DType.uint16:
            return get_spec[DType.uint16]()
        if self.dtype() is DType.uint32:
            return get_spec[DType.uint32]()
        if self.dtype() is DType.uint64:
            return get_spec[DType.uint64]()

        if self.dtype() is DType.float16:
            return get_spec[DType.float16]()
        if self.dtype() is DType.float32:
            return get_spec[DType.float32]()
        if self.dtype() is DType.float64:
            return get_spec[DType.float64]()
        if self.dtype() is DType.bool:
            return get_spec[DType.bool]()

        raise String("Expected type: ") + self.dtype().__str__()
