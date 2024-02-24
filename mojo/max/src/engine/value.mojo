# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from sys.ffi import DLHandle
from ._context import CRuntimeContext
from ._tensor_impl import EngineTensor
from ._utils import call_dylib_func, exchange
from ._value_impl import CValue, CList


struct Value:
    var ptr: CValue
    var lib: DLHandle
    var session: InferenceSession

    alias _NewBorrowedTensorFnName = "M_createBorrowedTensor"
    alias _NewBoolFnName = "M_createBoolAsyncValue"
    alias _NewListFnName = "M_createListAsyncValue"

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
        var spec = EngineTensorSpec("", tensor.spec(), lib, session.copy())
        var ptr = call_dylib_func[CValue](
            lib,
            Self._NewBorrowedTensorFnName,
            tensor.data(),
            spec._borrow_ptr(),
            ctx,
        )
        _ = spec ^
        return Self(ptr, lib, session ^)

    fn _as_engine_tensor(self) raises -> EngineTensor:
        var ptr = self.ptr.get_c_tensor(self.lib)
        if not ptr.ptr:
            raise "value is not a tensor"
        return EngineTensor(ptr, self.lib, self.session.copy())

    fn as_tensor_copy[type: DType](self) raises -> Tensor[type]:
        """Return a copy of the tensor contained in this value."""
        return self._as_engine_tensor().tensor[type]()

    @staticmethod
    fn _new_bool(
        ctx: CRuntimeContext,
        lib: DLHandle,
        owned session: InferenceSession,
        value: Bool,
    ) raises -> Self:
        var ptr = call_dylib_func[CValue](lib, Self._NewBoolFnName, value, ctx)
        return Self(ptr, lib, session ^)

    fn as_bool(self) -> Bool:
        """Get the boolean contained in this value."""
        return self.ptr.get_bool(self.lib)

    @staticmethod
    fn _new_list(
        ctx: CRuntimeContext, lib: DLHandle, owned session: InferenceSession
    ) raises -> Self:
        var ptr = call_dylib_func[CValue](lib, Self._NewListFnName, ctx)
        return Self(ptr, lib, session ^)

    fn as_list(self) raises -> List:
        """Borrow the list contained in this value.

        Ownership of the list is not transferred.  User must ensure the value
        outlives the list.
        """
        var ptr = self.ptr.get_list(self.lib)
        if not ptr.ptr:
            raise "value is not a list"
        return List(ptr, self.lib, self.session.copy())


struct List(Sized):
    var ptr: CList
    var lib: DLHandle
    var session: InferenceSession

    fn __init__(
        inout self,
        ptr: CList,
        lib: DLHandle,
        owned session: InferenceSession,
    ):
        self.ptr = ptr
        self.lib = lib
        self.session = session ^

    fn __moveinit__(inout self, owned existing: Self):
        self.ptr = exchange[CList](existing.ptr, DTypePointer[DType.invalid]())
        self.lib = existing.lib
        self.session = existing.session ^

    fn __del__(owned self):
        self.ptr.free(self.lib)
        _ = self.session ^

    fn __len__(self) -> Int:
        return self.ptr.get_size(self.lib)

    fn __getitem__(self, index: Int) raises -> Value:
        var c_value = self.ptr.get_value(self.lib, index)
        if not c_value.ptr:
            raise "list index out of range"
        return Value(c_value, self.lib, self.session.copy())

    fn append(self, value: Value):
        self.ptr.append(self.lib, value.ptr)
