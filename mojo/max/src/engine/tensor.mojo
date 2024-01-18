# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory.unsafe import DTypePointer
from sys.ffi import DLHandle
from ._utils import *
from ._dtypes import *
from ._tensor_spec_impl import *
from memory.unsafe import bitcast, Pointer
from tensor import Tensor
from .session import InferenceSession
from python import PythonObject
from python import Python
from collections.vector import DynamicVector
from utils.list import Dim
from ._tensor_impl import _Numpy, CTensor


@value
@register_passable
struct EngineTensorView:
    """A non-owning register_passable view of a tensor
    that does runtime type checking.

    CAUTION: Make sure the source tensor outlives the view.
    """

    var ptr: Pointer[Tensor[DType.invalid]]
    var data_ptr: DTypePointer[DType.invalid]
    var dtype: DType

    fn __init__[type: DType](inout tensor: Tensor[type]) -> Self:
        return Self {
            ptr: bitcast[Tensor[DType.invalid]](
                Pointer[Tensor[type]].address_of(tensor)
            ),
            data_ptr: bitcast[DType.invalid](tensor.data()),
            dtype: type,
        }

    fn data[type: DType](self) raises -> DTypePointer[type]:
        if type != self.dtype:
            raise String("Expected type: ") + self.dtype.__str__()
        return bitcast[type](self.data_ptr)

    fn data(self) -> DTypePointer[DType.invalid]:
        return self.data_ptr

    fn spec(self) raises -> TensorSpec:
        @always_inline
        @parameter
        fn get_spec[ty: DType]() -> TensorSpec:
            return self._get_value[ty]().spec()

        if self.dtype.is_int8():
            return get_spec[DType.int8]()
        if self.dtype.is_int16():
            return get_spec[DType.int16]()
        if self.dtype.is_int32():
            return get_spec[DType.int32]()
        if self.dtype.is_int64():
            return get_spec[DType.int64]()

        if self.dtype.is_uint8():
            return get_spec[DType.uint8]()
        if self.dtype.is_uint16():
            return get_spec[DType.uint16]()
        if self.dtype.is_uint32():
            return get_spec[DType.uint32]()
        if self.dtype.is_uint64():
            return get_spec[DType.uint64]()

        if self.dtype.is_float16():
            return get_spec[DType.float16]()
        if self.dtype.is_float32():
            return get_spec[DType.float32]()
        if self.dtype.is_float64():
            return get_spec[DType.float64]()
        if self.dtype.is_bool():
            return get_spec[DType.bool]()

        raise String("Expected type: ") + self.dtype.__str__()

    @always_inline("nodebug")
    fn _get_value[type: DType](self) -> Tensor[type]:
        return __get_address_as_lvalue(bitcast[Tensor[type]](self.ptr).address)


@value
@register_passable
struct EngineNumpyView:
    """A non-owning register_passable view of a numpy array.

    CAUTION: Make sure the source array outlives the view.
    """

    var np: _Numpy
    var ptr: Pointer[PythonObject]

    fn __init__(inout tensor: PythonObject) raises -> Self:
        return Self {
            np: _Numpy(), ptr: Pointer[PythonObject].address_of(tensor)
        }

    fn data(self) raises -> DTypePointer[DType.invalid]:
        let data_ptr = __get_address_as_lvalue(
            self.ptr.address
        ).ctypes.data.__index__()
        return bitcast[DType.invalid](data_ptr)

    fn dtype(self) raises -> DType:
        let self_type = __get_address_as_lvalue(self.ptr.address).dtype
        if self_type == self.np.int8:
            return DType.int8
        if self_type == self.np.int16:
            return DType.int16
        if self_type == self.np.int32:
            return DType.int32
        if self_type == self.np.int64:
            return DType.int64

        if self_type == self.np.uint8:
            return DType.uint8
        if self_type == self.np.uint16:
            return DType.uint16
        if self_type == self.np.uint32:
            return DType.uint32
        if self_type == self.np.uint64:
            return DType.uint64

        if self_type == self.np.float16:
            return DType.float16
        if self_type == self.np.float32:
            return DType.float32
        if self_type == self.np.float64:
            return DType.float64

        raise "Unknown datatype"

    fn spec(self) raises -> TensorSpec:
        @always_inline
        @parameter
        fn get_spec[ty: DType]() raises -> TensorSpec:
            var shape = DynamicVector[Int]()
            var array_shape = __get_address_as_lvalue(self.ptr.address).shape
            for dim in array_shape:
                shape.push_back(dim.__index__())
            return TensorSpec(ty, shape)

        if self.dtype().is_int8():
            return get_spec[DType.int8]()
        if self.dtype().is_int16():
            return get_spec[DType.int16]()
        if self.dtype().is_int32():
            return get_spec[DType.int32]()
        if self.dtype().is_int64():
            return get_spec[DType.int64]()

        if self.dtype().is_uint8():
            return get_spec[DType.uint8]()
        if self.dtype().is_uint16():
            return get_spec[DType.uint16]()
        if self.dtype().is_uint32():
            return get_spec[DType.uint32]()
        if self.dtype().is_uint64():
            return get_spec[DType.uint64]()

        if self.dtype().is_float16():
            return get_spec[DType.float16]()
        if self.dtype().is_float32():
            return get_spec[DType.float32]()
        if self.dtype().is_float64():
            return get_spec[DType.float64]()
        if self.dtype().is_bool():
            return get_spec[DType.bool]()

        raise String("Expected type: ") + self.dtype().__str__()
