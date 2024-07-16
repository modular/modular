# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from buffer import Buffer
from memory.unsafe import bitcast, DTypePointer
from python import Python, PythonObject
from sys.ffi import DLHandle
from max._utils import call_dylib_func, exchange

from .session import InferenceSession
from .tensor_spec import TensorSpec
from ._dtypes import EngineDType
from ._tensor_spec_impl import CTensorSpec

from max.tensor import Tensor


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

    fn size(self, lib: DLHandle) -> Int:
        return call_dylib_func[Int](lib, Self.GetTensorNumElementsFnName, self)

    fn dtype(self, lib: DLHandle) -> EngineDType:
        return call_dylib_func[EngineDType](
            lib, Self.GetTensorDTypeFnName, self
        )

    fn unsafe_ptr(self, lib: DLHandle) -> DTypePointer[DType.invalid]:
        return call_dylib_func[DTypePointer[DType.invalid]](
            lib, Self.GetTensorDataFnName, self
        )

    fn get_tensor_spec(
        self, lib: DLHandle, owned session: InferenceSession
    ) -> EngineTensorSpec:
        var spec = call_dylib_func[CTensorSpec](
            lib, Self.GetTensorSpecFnName, self
        )
        return EngineTensorSpec(spec, lib, session^)

    fn free(self, lib: DLHandle):
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
        self.session = session^

    fn __moveinit__(inout self, owned existing: Self):
        self.ptr = exchange[CTensor](
            existing.ptr, DTypePointer[DType.invalid]()
        )
        self.lib = existing.lib
        self.session = existing.session^

    fn __len__(self) -> Int:
        return self.ptr.size(self.lib)

    fn unsafe_ptr(self) -> DTypePointer[DType.invalid]:
        return self.ptr.unsafe_ptr(self.lib)

    fn data[type: DType](self) raises -> DTypePointer[type]:
        var ptr = self.unsafe_ptr()
        return ptr.bitcast[type]()

    fn dtype(self) -> DType:
        return self.ptr.dtype(self.lib).to_dtype()

    fn spec(self) raises -> TensorSpec:
        return self.ptr.get_tensor_spec(
            self.lib, self.session
        ).get_as_tensor_spec()

    fn buffer[type: DType](self) raises -> Buffer[type]:
        return Buffer[type](self.data[type](), len(self))

    fn buffer(self) -> Buffer[DType.invalid]:
        return Buffer[DType.invalid](
            self.unsafe_ptr(), len(self) * self.dtype().sizeof()
        )

    fn tensor[type: DType](self) raises -> Tensor[type]:
        var tensor = Tensor[type](self.spec())
        memcpy(
            tensor.unsafe_ptr().address,
            self.data[type]().address,
            len(self),
        )
        return tensor^

    fn __del__(owned self):
        self.ptr.free(self.lib)
        _ = self.session^


@value
@register_passable
struct _Numpy:
    var np: PythonObject

    fn __init__(inout self) raises:
        self.np = Python.import_module("numpy")

    fn __getattr__(self, attr: StringLiteral) raises -> PythonObject:
        return self.np.__getattr__(attr)
