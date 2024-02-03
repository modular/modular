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
from memory.unsafe import bitcast
from memory.anypointer import AnyPointer
from tensor import Tensor
from .session import InferenceSession
from python import PythonObject
from python import Python
from collections.vector import DynamicVector
from utils.list import Dim


@value
@register_passable("trivial")
struct CTensor:
    """Represents AsyncTensor ptr from Engine."""

    var ptr: DTypePointer[DType.invalid]

    alias GetTensorNumElementsFnName = "M_getTensorNumElements"
    alias GetTensorDTypeFnName = "M_getTensorType"
    alias GetTensorDataFnName = "M_getTensorData"
    alias GetTensorSpecFnName = "M_getTensorSpec"
    alias FreeTensorFnName = "M_freeTensor"

    fn size(self, borrowed lib: DLHandle) -> Int:
        return call_dylib_func[Int](lib, Self.GetTensorNumElementsFnName, self)

    fn dtype(self, borrowed lib: DLHandle) -> EngineDType:
        return call_dylib_func[EngineDType](
            lib, Self.GetTensorDTypeFnName, self
        )

    fn data(self, borrowed lib: DLHandle) -> DTypePointer[DType.invalid]:
        return call_dylib_func[DTypePointer[DType.invalid]](
            lib, Self.GetTensorDataFnName, self
        )

    fn get_tensor_spec(
        self, borrowed lib: DLHandle, owned session: InferenceSession
    ) -> EngineTensorSpec:
        let spec = call_dylib_func[CTensorSpec](
            lib, Self.GetTensorSpecFnName, self
        )
        return EngineTensorSpec(spec, lib, session ^)

    fn free(self, borrowed lib: DLHandle):
        """
        Free the status ptr.
        """
        call_dylib_func(lib, Self.FreeTensorFnName, self)


struct EngineTensor(Sized):
    var ptr: CTensor
    var lib: DLHandle
    var session: InferenceSession

    fn __init__(
        inout self,
        ptr: CTensor,
        lib: DLHandle,
        owned session: InferenceSession,
    ):
        self.ptr = ptr
        self.lib = lib
        self.session = session ^

    fn __moveinit__(inout self, owned existing: Self):
        self.ptr = exchange[CTensor](
            existing.ptr, DTypePointer[DType.invalid]()
        )
        self.lib = existing.lib
        self.session = existing.session ^

    fn __len__(self) -> Int:
        return self.ptr.size(self.lib)

    fn data(self) -> DTypePointer[DType.invalid]:
        return self.ptr.data(self.lib)

    fn data[type: DType](self) raises -> DTypePointer[type]:
        let ptr = self.data()
        return bitcast[type](ptr)

    fn dtype(self) -> DType:
        return self.ptr.dtype(self.lib).to_dtype()

    fn spec(self) raises -> TensorSpec:
        return self.ptr.get_tensor_spec(
            self.lib, self.session.copy()
        ).get_as_tensor_spec()

    fn buffer[type: DType](self) raises -> Buffer[type, Dim()]:
        return Buffer[type, Dim()](self.data[type](), len(self))

    fn buffer(self) -> Buffer[DType.invalid, Dim()]:
        return Buffer[DType.invalid, Dim()](
            self.data(), len(self) * self.dtype().sizeof()
        )

    fn tensor[type: DType](self) raises -> Tensor[type]:
        let tensor = Tensor[type](self.spec())
        memcpy(
            tensor.data(),
            self.data[type](),
            len(self),
        )
        return tensor ^

    fn __del__(owned self):
        self.ptr.free(self.lib)
        _ = self.session ^


@value
@register_passable
struct _Numpy:
    var np: AnyPointer[PythonObject]

    fn __init__() raises -> Self:
        let np_ptr = AnyPointer[PythonObject].alloc(1)
        __get_address_as_uninit_lvalue(np_ptr.value) = Python.import_module(
            "numpy"
        )

        return Self {np: np_ptr}

    fn __getattr__(self, attr: StringLiteral) raises -> PythonObject:
        return __get_address_as_lvalue(self.np.value).__getattr__(attr)

    fn __del__(owned self):
        _ = __get_address_as_owned_value(self.np.value)
        self.np.free()
