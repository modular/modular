# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Provides C bindings to KServe types and data structures."""

from builtin.coroutine import _coro_resume_fn, _suspend_async
from memory.unsafe import DTypePointer
from sys.ffi import DLHandle
from runtime.llcl import _get_current_runtime

from max._utils import call_dylib_func, exchange


# ===----------------------------------------------------------------------=== #
# CBatch
# ===----------------------------------------------------------------------=== #


trait CRequest:
    fn __init__(inout self, lib: DLHandle, ptr: DTypePointer[DType.invalid]):
        ...


trait CResponse:
    fn __init__(inout self, lib: DLHandle, ptr: DTypePointer[DType.invalid]):
        ...


struct CBatch:
    """Corresponds to the batch C type."""

    var _lib: DLHandle
    var _ptr: DTypePointer[DType.invalid]

    alias _NewFnName = "M_newBatch"
    alias _FreeFnName = "M_freeBatch"
    alias _SizeFnName = "M_batchSize"
    alias _RequestAtFn = "M_batchRequestAt"
    alias _ResponseAtFn = "M_batchResponseAt"
    alias _PushCompleteFnName = "M_batchPushComplete"
    alias _PushFailedFnName = "M_batchPushFailed"

    alias _AsyncAndThenFnName = "M_asyncBatchAndThen"
    alias _GetFnName = "M_asyncBatchGet"
    alias _FreeValueFnName = "M_freeAsyncBatch"

    fn __init__(inout self, lib: DLHandle):
        self._lib = lib
        self._ptr = DTypePointer[DType.invalid]()
        call_dylib_func(
            self._lib, Self._NewFnName, UnsafePointer.address_of(self._ptr)
        )

    fn __init__(inout self, lib: DLHandle, ptr: DTypePointer[DType.invalid]):
        self._lib = lib
        self._ptr = ptr

    fn __moveinit__(inout self, owned existing: Self):
        self._lib = existing._lib
        self._ptr = exchange[DTypePointer[DType.invalid]](
            existing._ptr, DTypePointer[DType.invalid]()
        )

    fn __del__(owned self):
        call_dylib_func(self._lib, Self._FreeFnName, self._ptr)

    fn size(self) -> Int64:
        return call_dylib_func[Int64](self._lib, Self._SizeFnName, self._ptr)

    fn request_at[T: CRequest](self, index: Int64) -> T:
        var ptr = DTypePointer[DType.invalid]()
        call_dylib_func(
            self._lib,
            Self._RequestAtFn,
            self._ptr,
            index,
            UnsafePointer.address_of(ptr),
        )
        return T(self._lib, ptr)

    fn response_at[T: CResponse](self, index: Int64) -> T:
        var ptr = DTypePointer[DType.invalid]()
        call_dylib_func(
            self._lib,
            Self._ResponseAtFn,
            self._ptr,
            index,
            UnsafePointer.address_of(ptr),
        )
        return T(self._lib, ptr)

    fn push_complete(inout self, index: Int64):
        call_dylib_func(
            self._lib,
            self._PushCompleteFnName,
            self._ptr,
            index,
        )

    fn push_failed(inout self, index: Int64, message: String):
        call_dylib_func(
            self._lib,
            self._PushFailedFnName,
            self._ptr,
            index,
            message._strref_dangerous(),
        )

    async fn load[
        pop_function: StringLiteral
    ](inout self, server_ptr: DTypePointer[DType.invalid]):
        """Waits for a batch to become available."""
        var batch_ptr = DTypePointer[DType.invalid]()
        call_dylib_func(
            self._lib,
            pop_function,
            server_ptr,
            _get_current_runtime(),
            UnsafePointer.address_of(batch_ptr),
        )

        @always_inline
        @parameter
        fn await_body(cur_hdl: AnyCoroutine):
            call_dylib_func(
                self._lib,
                Self._AsyncAndThenFnName,
                _coro_resume_fn,
                batch_ptr,
                cur_hdl,
            )

        _suspend_async[await_body]()

        call_dylib_func(
            self._lib,
            Self._GetFnName,
            batch_ptr,
            self._ptr,
        )
        call_dylib_func(self._lib, Self._FreeValueFnName, batch_ptr)
