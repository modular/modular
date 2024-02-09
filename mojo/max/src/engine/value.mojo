# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from sys.ffi import DLHandle
from ._context import CRuntimeContext
from ._tensor_impl import EngineTensor
from ._utils import call_dylib_func, exchange
from ._value_impl import CValue


struct Value:
    var ptr: CValue
    var lib: DLHandle
    var session: InferenceSession

    alias _NewBorrowedTensorFnName = "M_createBorrowedTensor"

    fn __init__(
        inout self,
        ptr: CValue,
        lib: DLHandle,
        owned session: InferenceSession,
    ):
        self.ptr = ptr
        self.lib = lib
        self.session = session ^

    fn __moveinit__(inout self, owned existing: Self):
        self.ptr = exchange[CValue](existing.ptr, DTypePointer[DType.invalid]())
        self.lib = existing.lib
        self.session = existing.session ^

    fn __del__(owned self):
        self.ptr.free(self.lib)
        _ = self.session ^

    @staticmethod
    fn _new_borrowed_tensor[
        type: DType
    ](
        ctx: CRuntimeContext,
        lib: DLHandle,
        owned session: InferenceSession,
        tensor: Tensor[type],
    ) raises -> Self:
        let spec = EngineTensorSpec("", tensor.spec(), lib, session.copy())
        let ptr = call_dylib_func[CValue](
            lib,
            Self._NewBorrowedTensorFnName,
            tensor.data(),
            spec._borrow_ptr(),
            ctx,
        )
        _ = spec ^
        return Self(ptr, lib, session ^)

    fn _as_engine_tensor(self) raises -> EngineTensor:
        let ptr = self.ptr.get_c_tensor(self.lib)
        if not ptr.ptr:
            raise "value is not a tensor"
        return EngineTensor(ptr, self.lib, self.session.copy())

    fn as_tensor_copy[type: DType](self) raises -> Tensor[type]:
        """Return a copy of the tensor contained in this value."""
        return self._as_engine_tensor().tensor[type]()
