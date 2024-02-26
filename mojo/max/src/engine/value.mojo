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
    var _ptr: CValue
    var _lib: DLHandle
    var _session: InferenceSession

    alias _NewBorrowedTensorFnName = "M_createBorrowedTensor"
    alias _NewBoolFnName = "M_createBoolAsyncValue"
    alias _NewListFnName = "M_createListAsyncValue"

    fn __init__(
        inout self,
        ptr: CValue,
        lib: DLHandle,
        owned session: InferenceSession,
    ):
        self._ptr = ptr
        self._lib = lib
        self._session = session ^

    fn __moveinit__(inout self, owned existing: Self):
        self._ptr = exchange[CValue](
            existing._ptr, DTypePointer[DType.invalid]()
        )
        self._lib = existing._lib
        self._session = existing._session ^

    fn __del__(owned self):
        self._ptr.free(self._lib)
        _ = self._session ^

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
        var ptr = self._ptr.get_c_tensor(self._lib)
        if not ptr.ptr:
            raise "value is not a tensor"
        return EngineTensor(ptr, self._lib, self._session.copy())

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
        return self._ptr.get_bool(self._lib)

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
        var ptr = self._ptr.get_list(self._lib)
        if not ptr.ptr:
            raise "value is not a list"
        return List(ptr, self._lib, self._session.copy())


struct List(Sized):
    var _ptr: CList
    var _lib: DLHandle
    var _session: InferenceSession

    fn __init__(
        inout self,
        ptr: CList,
        lib: DLHandle,
        owned session: InferenceSession,
    ):
        self._ptr = ptr
        self._lib = lib
        self._session = session ^

    fn __moveinit__(inout self, owned existing: Self):
        self._ptr = exchange[CList](
            existing._ptr, DTypePointer[DType.invalid]()
        )
        self._lib = existing._lib
        self._session = existing._session ^

    fn __del__(owned self):
        self._ptr.free(self._lib)
        _ = self._session ^

    fn __len__(self) -> Int:
        return self._ptr.get_size(self._lib)

    fn __getitem__(self, index: Int) raises -> Value:
        var c_value = self._ptr.get_value(self._lib, index)
        if not c_value.ptr:
            raise "list index out of range"
        return Value(c_value, self._lib, self._session.copy())

    fn append(self, value: Value):
        self._ptr.append(self._lib, value._ptr)
