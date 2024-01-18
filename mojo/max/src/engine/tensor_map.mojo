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
from tensor import Tensor, TensorShape
from ._tensor_impl import *
from ._tensor_spec_impl import *
from ._context import *
from .session import InferenceSession
from ._tensor_map_impl import CTensorMap


struct TensorMap(SizedRaising):
    var ptr: CTensorMap
    var lib: DLHandle
    var session: InferenceSession

    alias NewTensorMapFnName = "M_newAsyncTensorMap"

    fn __init__(
        inout self,
        ctx: CRuntimeContext,
        lib: DLHandle,
        owned session: InferenceSession,
    ):
        self.ptr = call_dylib_func[CTensorMap](
            lib, Self.NewTensorMapFnName, ctx
        )
        self.lib = lib
        self.session = session ^

    fn __init__(
        inout self,
        ptr: CTensorMap,
        lib: DLHandle,
        owned session: InferenceSession,
    ):
        self.ptr = ptr
        self.lib = lib
        self.session = session ^

    fn __moveinit__(inout self, owned existing: Self):
        self.ptr = exchange[CTensorMap](
            existing.ptr, DTypePointer[DType.invalid]()
        )
        self.lib = existing.lib
        self.session = existing.session ^

    fn borrow[type: DType](self, key: StringRef, value: Tensor[type]) raises:
        let spec = EngineTensorSpec(
            key, value.spec(), self.lib, self.session.copy()
        )
        self.ptr.borrow_tensor_by_name(
            bitcast[DType.invalid](value.data()), spec, self.lib
        )

    fn borrow(self, key: StringRef, value: EngineTensorView) raises:
        let spec = EngineTensorSpec(
            key, value.spec(), self.lib, self.session.copy()
        )
        self.ptr.borrow_tensor_by_name(value.data(), spec, self.lib)

    fn borrow(self, key: StringRef, value: EngineNumpyView) raises:
        let spec = EngineTensorSpec(
            key, value.spec(), self.lib, self.session.copy()
        )
        self.ptr.borrow_tensor_by_name(value.data(), spec, self.lib)

    fn get[type: DType](self, key: StringRef) raises -> Tensor[type]:
        let tensor_ptr = self.ptr.get_tensor_by_name(key.data, self.lib)
        let mof_tensor = EngineTensor(tensor_ptr, self.lib, self.session.copy())
        let tensor = mof_tensor.tensor[type]()
        return tensor ^

    fn buffer[type: DType](self, key: StringRef) raises -> Buffer[Dim(), type]:
        let tensor_ptr = self.ptr.get_tensor_by_name(key.data, self.lib)
        return EngineTensor(tensor_ptr, self.lib, self.session.copy()).buffer[
            type
        ]()

    fn __len__(self) raises -> Int:
        return self.ptr.size(self.lib)

    fn borrow_ptr(self) -> CTensorMap:
        return self.ptr

    fn __del__(owned self):
        self.ptr.free(self.lib)
        _ = self.session ^
