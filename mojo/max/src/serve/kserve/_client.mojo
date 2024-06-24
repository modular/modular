# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Provides client-related C bindings to ServeRT."""

from sys.ffi import DLHandle
from memory import UnsafePointer
from memory.unsafe import DTypePointer
from runtime.llcl import Chain

from max_utils import call_dylib_func, exchange, CString

from ._types import CInferenceRequest, CInferenceResponse


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


struct CGRPCClient:
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
