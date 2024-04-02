# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Provides C bindings to MUTT."""

from sys.ffi import DLHandle
from memory.unsafe import DTypePointer, Pointer

from max.engine import InferenceSession
from max.engine._utils import call_dylib_func, exchange, CString

from .kserve import (
    ModelInferRequest,
    ModelInferResponse,
    CModelInferRequest,
    CModelInferResponse,
)
from .service import ModelInfo


@value
@register_passable("trivial")
struct CBatch:
    var ptr: DTypePointer[DType.invalid]

    alias _FreeFnName = "M_freeBatch"
    alias _SizeFnName = "M_batchSize"
    alias _RequestAtFn = "M_batchRequestAt"
    alias _ResponseAtFn = "M_batchResponseAt"

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
        return self._ptr.size(self._lib).to_int()

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
struct CMuttServerAsync:
    var ptr: DTypePointer[DType.invalid]

    alias _NewFnName = "M_newMuttServer"
    alias _FreeFnName = "M_freeMuttServer"
    alias _RunFnName = "M_muttRun"
    alias _PopReadyFnName = "M_muttPopReady"
    alias _PushCompleteFnName = "M_muttPushComplete"

    @staticmethod
    fn new(lib: DLHandle, address: StringRef) -> CMuttServerAsync:
        return call_dylib_func[CMuttServerAsync](lib, Self._NewFnName, address)

    fn free(owned self, lib: DLHandle):
        call_dylib_func(lib, Self._FreeFnName, self.ptr)

    fn run(owned self, lib: DLHandle):
        call_dylib_func(lib, Self._RunFnName, self.ptr)

    fn pop_ready(owned self, lib: DLHandle) -> CBatch:
        return call_dylib_func[CBatch](lib, self._PopReadyFnName, self.ptr)

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
        address._strref_keepalive()

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

    fn run(self):
        self._ptr.run(self._lib)

    fn pop_ready(self) -> Batch:
        return Batch(self._ptr.pop_ready(self._lib), self._lib, self._session)

    fn push_complete(self, batch: Batch, index: Int64):
        self._ptr.push_complete(self._lib, batch._ptr, index)
