# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Provides C bindings to ServeRT."""

from sys.ffi import DLHandle
from memory import UnsafePointer
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


struct ClientResult:
    """Corresponds to the M_ClientResult C type; should only be used as a Pointer.
    """

    var response: DTypePointer[DType.invalid]
    var error: CString
    var code: Int

    alias _FreeValueFnName = "M_freeClientResult"

    @staticmethod
    fn free(lib: DLHandle, ptr: UnsafePointer[ClientResult]):
        call_dylib_func(lib, Self._FreeValueFnName, ptr)


struct CKServeClientAsync:
    """Corresponds to the M_KServeClient C type."""

    var _lib: DLHandle
    var _ptr: DTypePointer[DType.invalid]

    alias _NewFnName = "M_newKServeClient"
    alias _FreeFnName = "M_freeKServeClient"
    alias _RunFnName = "M_runKServeClient"

    alias _ModelInferFnName = "M_modelInfer"
    alias _ModelInferCreateRequestFnName = "M_modelInferCreateRequest"
    alias _ModelInferTakeResultFnName = "M_modelInferTakeResult"

    fn __init__(inout self, lib: DLHandle, address: StringRef):
        self._lib = lib
        self._ptr = DTypePointer[DType.invalid]()
        call_dylib_func(
            self._lib,
            Self._NewFnName,
            address,
            UnsafePointer.address_of(self._ptr),
        )

    fn __moveinit__(inout self, owned existing: Self):
        self._lib = existing._lib
        self._ptr = exchange[DTypePointer[DType.invalid]](
            existing._ptr, DTypePointer[DType.invalid]()
        )

    fn __del__(owned self):
        call_dylib_func(self._lib, Self._FreeFnName, self._ptr)

    fn run(inout self):
        call_dylib_func(self._lib, Self._RunFnName, self._ptr)

    fn model_infer(inout self, request: CInferenceRequest) -> Chain:
        var chain = Chain()
        call_dylib_func(
            self._lib,
            Self._ModelInferFnName,
            self._ptr,
            request._ptr,
            UnsafePointer.address_of(chain.storage),
        )
        return chain

    fn create_infer_request(
        inout self, name: StringRef, version: StringRef
    ) -> CInferenceRequest:
        var ptr = DTypePointer[DType.invalid]()
        call_dylib_func(
            self._lib,
            Self._ModelInferCreateRequestFnName,
            self._ptr,
            name,
            version,
            UnsafePointer.address_of(ptr),
        )
        return CInferenceRequest(self._lib, ptr, owning=True)

    fn take_infer_result(
        inout self, request: CInferenceRequest
    ) -> UnsafePointer[ClientResult]:
        # The result is allocated inline and read directly into the passed
        # memory. This is because we have the full type corresponding to the C
        # type and don't need indirection.
        var result = UnsafePointer[ClientResult]()
        call_dylib_func(
            self._lib,
            Self._ModelInferTakeResultFnName,
            self._ptr,
            request._ptr,
            UnsafePointer.address_of(result),
        )
        return result


struct KServeClientAsync:
    var _impl: CKServeClientAsync
    var _session: InferenceSession

    fn __init__(
        inout self,
        lib: DLHandle,
        address: String,
        owned session: InferenceSession,
    ):
        self._impl = CKServeClientAsync(lib, address._strref_dangerous())
        self._session = session^

    fn __moveinit__(inout self: Self, owned existing: Self):
        self._impl = existing._impl^
        self._session = existing._session^

    fn run(inout self):
        self._impl.run()

    fn create_infer_request(
        inout self, name: String, version: String
    ) -> InferenceRequestImpl:
        return InferenceRequestImpl(
            self._impl.create_infer_request(
                name._strref_dangerous(), version._strref_dangerous()
            ),
            self._session,
        )

    async fn model_infer(
        inout self,
        request: InferenceRequestImpl,
        inout response: Variant[InferenceResponseImpl, Error],
    ):
        await ChainPromise(self._impl.model_infer(request._impl))
        var result = self._impl.take_infer_result(request._impl)
        if result[].code != 0:
            # The response should be null in this case.
            response.set[Error](Error(str(result[].error)))
        else:
            # This hands ownership of the response to the underlying
            # InferenceRequest object. It will be freed separately when this
            # object is freed.
            response.set[InferenceResponseImpl](
                InferenceResponseImpl(
                    CInferenceResponse(
                        self._impl._lib, result[].response, owning=True
                    ),
                    self._session,
                )
            )
        # Free the allocated result.
        ClientResult.free(self._impl._lib, result)
