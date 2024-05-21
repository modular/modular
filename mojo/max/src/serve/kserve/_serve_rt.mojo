# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Provides C bindings to ServeRT."""

from sys.ffi import DLHandle
from memory.unsafe import DTypePointer, Pointer
from builtin.coroutine import _coro_resume_fn, _suspend_async
from runtime.llcl import Runtime, Chain, ChainPromise
from utils.variant import Variant

from max.engine import InferenceSession, Model
from max.engine._compilation import CCompiledModel
from max.engine._model_impl import CModel
from max.engine._utils import call_dylib_func, exchange, CString

from ._kserve_impl import (
    ModelInferRequest,
    ModelInferResponse,
    CModelInferRequest,
    CModelInferResponse,
)


@value
@register_passable
struct TensorView:
    """Corresponds to the M_TensorView C type."""

    var name: StringRef
    var dtype: StringRef
    var shape: Pointer[Int64]
    var rank: Int
    var contents: Pointer[UInt8]
    var contentsSize: Int

    alias _FreeValueFnName = "M_freeTensorView"

    @staticmethod
    fn free(lib: DLHandle, ptr: Pointer[TensorView]):
        call_dylib_func(lib, Self._FreeValueFnName, ptr)


@value
@register_passable("trivial")
struct CBatch:
    var ptr: DTypePointer[DType.invalid]

    alias _NewFnName = "M_newBatch"
    alias _FreeFnName = "M_freeBatch"
    alias _SizeFnName = "M_batchSize"
    alias _RequestAtFn = "M_batchRequestAt"
    alias _ResponseAtFn = "M_batchResponseAt"

    @staticmethod
    fn new(lib: DLHandle) -> CBatch:
        return call_dylib_func[CBatch](lib, Self._NewFnName)

    fn free(owned self, lib: DLHandle):
        call_dylib_func(lib, Self._FreeFnName, self.ptr)

    fn size(self, lib: DLHandle) -> Int64:
        return call_dylib_func[Int64](lib, Self._SizeFnName, self.ptr)

    fn request_at(self, lib: DLHandle, index: Int64) -> CModelInferRequest:
        return call_dylib_func[CModelInferRequest](
            lib, Self._RequestAtFn, self.ptr, index
        )

    fn response_at(self, lib: DLHandle, index: Int64) -> CModelInferResponse:
        return call_dylib_func[CModelInferResponse](
            lib, Self._ResponseAtFn, self.ptr, index
        )


struct Batch(Sized, CollectionElement):
    var _ptr: CBatch
    var _lib: DLHandle
    var _session: InferenceSession

    fn __init__(inout self, lib: DLHandle, owned session: InferenceSession):
        self._ptr = CBatch.new(lib)
        self._lib = lib
        self._session = session^

    fn __init__(
        inout self, ptr: CBatch, lib: DLHandle, owned session: InferenceSession
    ):
        self._ptr = ptr
        self._lib = lib
        self._session = session^

    fn __moveinit__(inout self: Self, owned existing: Self):
        self._ptr = exchange[CBatch](
            existing._ptr, DTypePointer[DType.invalid]()
        )
        self._lib = existing._lib
        self._session = existing._session^

    fn __copyinit__(inout self: Self, existing: Self):
        self._ptr = existing._ptr
        self._lib = existing._lib
        self._session = existing._session

    fn __del__(owned self):
        self._ptr.free(self._lib)
        _ = self._session^

    fn __len__(self) -> Int:
        return int(self._ptr.size(self._lib))

    fn model_name(self) -> String:
        return ""

    fn model_version(self) -> String:
        return ""

    fn request_at(self, index: Int64) -> ModelInferRequest:
        return ModelInferRequest(
            self._ptr.request_at(self._lib, index),
            self._lib,
            self._session,
        )

    fn response_at(self, index: Int64) -> ModelInferResponse:
        return ModelInferResponse(
            self._ptr.response_at(self._lib, index),
            self._lib,
            self._session,
        )

    fn requests(self) -> List[ModelInferRequest]:
        var requests = List[ModelInferRequest](capacity=len(self))
        for i in range(len(self)):
            requests.append(self.request_at(i))
        return requests^

    fn responses(self) -> List[ModelInferResponse]:
        var responses = List[ModelInferResponse](capacity=len(self))
        for i in range(len(self)):
            responses.append(self.response_at(i))
        return responses^


@value
@register_passable("trivial")
struct AsyncCBatch:
    var ptr: DTypePointer[DType.invalid]

    alias _AsyncAndThenFnName = "M_asyncBatchAndThen"
    alias _GetFnName = "M_asyncBatchGet"
    alias _FreeValueFnName = "M_freeAsyncBatch"

    fn get(self, lib: DLHandle) -> CBatch:
        return call_dylib_func[CBatch](lib, Self._GetFnName, self)

    fn free(self, lib: DLHandle):
        call_dylib_func(lib, Self._FreeValueFnName, self)


struct AwaitableCBatch:
    var _ptr: AsyncCBatch
    var _lib: DLHandle

    fn __init__(
        inout self,
        ptr: AsyncCBatch,
        lib: DLHandle,
    ):
        self._ptr = ptr
        self._lib = lib

    fn __del__(owned self):
        self._ptr.free(self._lib)

    @always_inline
    fn __await__(self) -> CBatch:
        @always_inline
        @parameter
        fn await_body(cur_hdl: AnyCoroutine):
            call_dylib_func(
                self._lib,
                AsyncCBatch._AsyncAndThenFnName,
                _coro_resume_fn,
                self._ptr,
                cur_hdl,
            )

        _suspend_async[await_body]()
        return self._ptr.get(self._lib)


@value
@register_passable("trivial")
struct CMuttServerAsync:
    var ptr: DTypePointer[DType.invalid]

    alias _NewFnName = "M_newMuttServer"
    alias _FreeFnName = "M_freeMuttServer"
    alias _InitFnName = "M_muttInit"
    alias _RunFnName = "M_muttRun"
    alias _PopReadyFnName = "M_muttPopReady"
    alias _PushCompleteFnName = "M_muttPushComplete"
    alias _PushFailedFnName = "M_muttPushFailed"

    @staticmethod
    fn new(lib: DLHandle, address: StringRef) -> CMuttServerAsync:
        return call_dylib_func[CMuttServerAsync](lib, Self._NewFnName, address)

    fn free(owned self, lib: DLHandle):
        call_dylib_func(lib, Self._FreeFnName, self.ptr)

    fn init(owned self, lib: DLHandle, models: List[CCompiledModel]):
        call_dylib_func(
            lib, Self._InitFnName, self.ptr, models.data, len(models)
        )

    fn run(owned self, lib: DLHandle):
        call_dylib_func(lib, Self._RunFnName, self.ptr)

    async fn pop_ready(owned self, lib: DLHandle) -> CBatch:
        var ptr = call_dylib_func[AsyncCBatch](
            lib, self._PopReadyFnName, self.ptr
        )
        var batch = AwaitableCBatch(ptr, lib)
        return await batch

    fn push_complete(
        owned self,
        lib: DLHandle,
        batch: CBatch,
        index: Int64,
    ):
        call_dylib_func(
            lib,
            self._PushCompleteFnName,
            self.ptr,
            batch.ptr,
            index,
        )

    fn push_failed(
        owned self, lib: DLHandle, batch: CBatch, index: Int64, message: String
    ):
        call_dylib_func(
            lib,
            self._PushFailedFnName,
            self.ptr,
            batch.ptr,
            index,
            message._strref_dangerous(),
        )


struct MuttServerAsync:
    var _ptr: CMuttServerAsync
    var _lib: DLHandle
    var _session: InferenceSession

    fn __init__(
        inout self,
        address: String,
        lib: DLHandle,
        owned session: InferenceSession,
    ):
        self._ptr = CMuttServerAsync.new(lib, address._strref_dangerous())
        self._lib = lib
        self._session = session^

    fn __moveinit__(inout self: Self, owned existing: Self):
        self._ptr = exchange[CMuttServerAsync](
            existing._ptr, DTypePointer[DType.invalid]()
        )
        self._lib = existing._lib
        self._session = existing._session^

    fn __copyinit__(inout self: Self, existing: Self):
        self._ptr = existing._ptr
        self._lib = existing._lib
        self._session = existing._session

    fn __del__(owned self):
        self._ptr.free(self._lib)
        _ = self._session^

    fn init(self, models: List[Model]):
        var ptrs = List[CCompiledModel](capacity=len(models))
        for model in models:
            ptrs.append(model[]._compiled_model.ptr)
        self._ptr.init(self._lib, ptrs)

    fn run(self):
        self._ptr.run(self._lib)

    async fn pop_ready(self, inout batch: Batch) -> None:
        var ptr = await self._ptr.pop_ready(self._lib)
        batch = Batch(ptr, self._lib, self._session)

    fn push_complete(self, batch: Batch, index: Int64):
        self._ptr.push_complete(self._lib, batch._ptr, index)

    fn push_failed(self, batch: Batch, index: Int64, message: String):
        self._ptr.push_failed(self._lib, batch._ptr, index, message)


@value
@register_passable
struct ClientResult:
    """Corresponds to the M_ClientResult C type."""

    var response: DTypePointer[DType.invalid]
    var error: CString
    var code: Int

    alias _FreeValueFnName = "M_freeClientResult"

    @staticmethod
    fn free(lib: DLHandle, ptr: Pointer[ClientResult]):
        call_dylib_func(lib, Self._FreeValueFnName, ptr)


@value
@register_passable("trivial")
struct CKServeClientAsync:
    var ptr: DTypePointer[DType.invalid]

    alias _NewFnName = "M_newKServeClient"
    alias _FreeFnName = "M_freeKServeClient"
    alias _RunFnName = "M_runKServeClient"

    alias _ModelInferFnName = "M_modelInfer"
    alias _ModelInferCreateRequestFnName = "M_modelInferCreateRequest"
    alias _ModelInferTakeResultFnName = "M_modelInferTakeResult"

    @staticmethod
    fn new(lib: DLHandle, address: StringRef) -> CKServeClientAsync:
        return call_dylib_func[CKServeClientAsync](
            lib, Self._NewFnName, address
        )

    fn free(owned self, lib: DLHandle):
        call_dylib_func(lib, Self._FreeFnName, self.ptr)

    fn run(owned self, lib: DLHandle):
        call_dylib_func(lib, Self._RunFnName, self.ptr)

    # ModelInfer

    fn model_infer(
        owned self, lib: DLHandle, request: CModelInferRequest
    ) -> Pointer[Chain]:
        return call_dylib_func[Pointer[Chain]](
            lib, Self._ModelInferFnName, self.ptr, request
        )

    fn new_infer_request(
        owned self, lib: DLHandle, name: StringRef, version: StringRef
    ) -> CModelInferRequest:
        return call_dylib_func[CModelInferRequest](
            lib, Self._ModelInferCreateRequestFnName, self.ptr, name, version
        )

    fn take_infer_result(
        owned self, lib: DLHandle, request: CModelInferRequest
    ) -> Pointer[ClientResult]:
        return call_dylib_func[Pointer[ClientResult]](
            lib, Self._ModelInferTakeResultFnName, self.ptr, request
        )


struct KServeClientAsync:
    var _ptr: CKServeClientAsync
    var _lib: DLHandle
    var _session: InferenceSession

    fn __init__(
        inout self,
        address: String,
        lib: DLHandle,
        owned session: InferenceSession,
    ):
        self._ptr = CKServeClientAsync.new(lib, address._strref_dangerous())
        self._lib = lib
        self._session = session^

    fn __moveinit__(inout self: Self, owned existing: Self):
        self._ptr = exchange[CKServeClientAsync](
            existing._ptr, DTypePointer[DType.invalid]()
        )
        self._lib = existing._lib
        self._session = existing._session^

    fn __copyinit__(inout self: Self, existing: Self):
        self._ptr = existing._ptr
        self._lib = existing._lib
        self._session = existing._session

    fn __del__(owned self):
        self._ptr.free(self._lib)
        _ = self._session^

    fn run(self):
        self._ptr.run(self._lib)

    fn new_infer_request(
        self, name: String, version: String
    ) -> ModelInferRequest:
        return ModelInferRequest(
            self._ptr.new_infer_request(
                self._lib, name._strref_dangerous(), version._strref_dangerous()
            ),
            self._lib,
            self._session,
        )

    async fn model_infer(
        self,
        request: ModelInferRequest,
        inout response: Variant[ModelInferResponse, Error],
    ):
        _ = await ChainPromise(self._ptr.model_infer(self._lib, request._ptr)[])
        var result_ptr = self._ptr.take_infer_result(self._lib, request._ptr)
        var result = result_ptr[]
        if result.code != 0:
            response.set[Error](Error(result.error))
        else:
            response.set[ModelInferResponse](
                ModelInferResponse(
                    result.response, self._lib, self._session, True
                )
            )
        ClientResult.free(self._lib, result_ptr)
