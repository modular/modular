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

from .._serve_rt import (
    InferenceRequestImpl,
    InferenceResponseImpl,
    CInferenceRequest,
    CInferenceResponse,
)


@value
@register_passable
struct ClientResult:
    """Corresponds to the M_ClientResult C type."""

    var response: DTypePointer[DType.invalid]
    var error: CString
    var code: Int

    alias _FreeValueFnName = "M_freeClientResult"

    @staticmethod
    fn free(lib: DLHandle, ptr: UnsafePointer[ClientResult]):
        call_dylib_func(lib, Self._FreeValueFnName, ptr)


@value
@register_passable("trivial")
struct CKServeClientAsync:
    """Corresponds to the M_KServeClient C type."""

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
        owned self, lib: DLHandle, request: CInferenceRequest
    ) -> Pointer[Chain]:
        return call_dylib_func[Pointer[Chain]](
            lib, Self._ModelInferFnName, self.ptr, request
        )

    fn create_infer_request(
        owned self, lib: DLHandle, name: StringRef, version: StringRef
    ) -> CInferenceRequest:
        return call_dylib_func[CInferenceRequest](
            lib, Self._ModelInferCreateRequestFnName, self.ptr, name, version
        )

    fn take_infer_result(
        owned self, lib: DLHandle, request: CInferenceRequest
    ) -> UnsafePointer[ClientResult]:
        return call_dylib_func[UnsafePointer[ClientResult]](
            lib, Self._ModelInferTakeResultFnName, self.ptr, request
        )

    # TODO: Fill in rest of {method, new, take} triples.


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

    fn create_infer_request(
        self, name: String, version: String
    ) -> InferenceRequestImpl:
        return InferenceRequestImpl(
            self._ptr.create_infer_request(
                self._lib, name._strref_dangerous(), version._strref_dangerous()
            ),
            self._lib,
            self._session,
        )

    async fn model_infer(
        self,
        request: InferenceRequestImpl,
        inout response: Variant[InferenceResponseImpl, Error],
    ):
        _ = await ChainPromise(self._ptr.model_infer(self._lib, request._ptr)[])
        var result_ptr = self._ptr.take_infer_result(self._lib, request._ptr)
        var result = result_ptr[]
        if result.code != 0:
            response.set[Error](Error(result.error))
        else:
            response.set[InferenceResponseImpl](
                InferenceResponseImpl(
                    result.response, self._lib, self._session, True
                )
            )
        ClientResult.free(self._lib, result_ptr)
