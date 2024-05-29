# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Provides C bindings to KServe data structures."""

from memory.unsafe import DTypePointer
from sys.ffi import DLHandle
from runtime.llcl import ChainPromise
from builtin.coroutine import _coro_resume_fn, _suspend_async

from max.engine._utils import (
    CString,
    call_dylib_func,
    exchange,
)
from max.engine import InferenceSession, Model, TensorMap
from max.engine._compilation import CCompiledModel
from max.serve.service import InferenceRequest, InferenceResponse
from max.tensor import TensorSpec


# ===----------------------------------------------------------------------=== #
# Utilities
# ===----------------------------------------------------------------------=== #


fn get_tensor_spec(view: TensorView) -> TensorSpec:
    var dtype = view.dtype
    var shape = List[Int](capacity=view.rank)
    for i in range(view.rank):
        shape.append(int(view.shape[i]))

    if dtype == "BOOL":
        return TensorSpec(DType.bool, shape)
    elif dtype == "UINT8":
        return TensorSpec(DType.uint8, shape)
    elif dtype == "UINT16":
        return TensorSpec(DType.uint16, shape)
    elif dtype == "UINT32":
        return TensorSpec(DType.uint32, shape)
    elif dtype == "UINT64":
        return TensorSpec(DType.uint64, shape)
    elif dtype == "INT8":
        return TensorSpec(DType.int8, shape)
    elif dtype == "INT16":
        return TensorSpec(DType.int16, shape)
    elif dtype == "INT32":
        return TensorSpec(DType.int32, shape)
    elif dtype == "INT64":
        return TensorSpec(DType.int64, shape)
    elif dtype == "FP16":
        return TensorSpec(DType.float16, shape)
    elif dtype == "FP32":
        return TensorSpec(DType.float32, shape)
    elif dtype == "FP64":
        return TensorSpec(DType.float64, shape)
    elif dtype == "BF16":
        return TensorSpec(DType.bfloat16, shape)
    # BYTES
    return TensorSpec(DType.int8, shape)


fn get_tensors[
    size_fn: StringLiteral,
    get_tensor_fn: StringLiteral,
    T: AnyTrivialRegType,
](lib: DLHandle, ptr: T, session: InferenceSession) raises -> TensorMap:
    var map = session.new_tensor_map()
    var size = call_dylib_func[Int64](lib, size_fn, ptr)
    for i in range(size):
        var view_ptr = call_dylib_func[Pointer[TensorView]](
            lib, get_tensor_fn, ptr, i
        )
        var view = view_ptr.load()
        map.borrow(view.name, get_tensor_spec(view), view.contents)
        TensorView.free(lib, view_ptr)
    return map^


fn buffer_str[type: DType](map: TensorMap, name: String) raises -> StringRef:
    var buffer = map.buffer[type](name)
    return StringRef(buffer.data.bitcast[DType.int8](), buffer.bytecount())


fn set_tensors[
    add_tensor_fn: StringLiteral,
    T: AnyTrivialRegType,
](lib: DLHandle, ptr: T, names: List[String], map: TensorMap) raises:
    for i in range(len(names)):
        var name = names[i]
        var spec = map.get_spec(name)
        var dtype = spec.dtype()

        var dtype_str: StringRef
        var contents_str: StringRef
        if dtype == DType.bool:
            dtype_str = "BOOL"
            contents_str = buffer_str[DType.bool](map, name)
        elif dtype == DType.uint8:
            dtype_str = "UINT8"
            contents_str = buffer_str[DType.uint8](map, name)
        elif dtype == DType.uint16:
            dtype_str = "UINT16"
            contents_str = buffer_str[DType.uint16](map, name)
        elif dtype == DType.uint32:
            dtype_str = "UINT32"
            contents_str = buffer_str[DType.uint32](map, name)
        elif dtype == DType.uint64:
            dtype_str = "UINT64"
            contents_str = buffer_str[DType.uint64](map, name)
        elif dtype == DType.int8:
            dtype_str = "INT8"
            contents_str = buffer_str[DType.int8](map, name)
        elif dtype == DType.int16:
            dtype_str = "INT16"
            contents_str = buffer_str[DType.int16](map, name)
        elif dtype == DType.int32:
            dtype_str = "INT32"
            contents_str = buffer_str[DType.int32](map, name)
        elif dtype == DType.int64:
            dtype_str = "INT64"
            contents_str = buffer_str[DType.int64](map, name)
        elif dtype == DType.float16:
            dtype_str = "FP16"
            contents_str = buffer_str[DType.float16](map, name)
        elif dtype == DType.float32:
            dtype_str = "FP32"
            contents_str = buffer_str[DType.float32](map, name)
        elif dtype == DType.float64:
            dtype_str = "FP64"
            contents_str = buffer_str[DType.float64](map, name)
        elif dtype == DType.bfloat16:
            dtype_str = "BF16"
            contents_str = buffer_str[DType.bfloat16](map, name)
        else:
            dtype_str = "BYTES"
            contents_str = buffer_str[DType.int8](map, name)

        var shape = List[Int64](capacity=spec.rank())
        for i in range(spec.rank()):
            shape.append(spec[i])

        # call_dylib_func[NoneType](
        #     lib,
        #     add_tensor_fn,
        #     ptr,
        #     name._strref_dangerous(),
        #     dtype_str,
        #     shape.data,
        #     spec.rank(),
        #     contents_str,
        # )
        # name._strref_keepalive()
        # _ = shape^

        var view = TensorView(
            name._strref_dangerous(),
            dtype_str,
            shape.data,
            spec.rank(),
            contents_str.unsafe_ptr(),
            len(contents_str),
        )
        call_dylib_func[NoneType](
            lib, add_tensor_fn, ptr, UnsafePointer.address_of(view)
        )
        name._strref_keepalive()
        _ = shape^


# ===----------------------------------------------------------------------=== #
# C bindings
# ===----------------------------------------------------------------------=== #


@value
@register_passable("trivial")
struct TensorView:
    """Corresponds to the M_TensorView C type."""

    var name: StringRef
    var dtype: StringRef
    var shape: UnsafePointer[Int64]
    var rank: Int
    var contents: UnsafePointer[UInt8]
    var contentsSize: Int

    alias _FreeValueFnName = "M_freeTensorView"

    @staticmethod
    fn free(lib: DLHandle, ptr: Pointer[TensorView]):
        call_dylib_func(lib, Self._FreeValueFnName, ptr)


@value
@register_passable("trivial")
struct CInferenceBatch:
    """Corresponds to the InfererenceBatch C type."""

    var ptr: DTypePointer[DType.invalid]

    alias _NewFnName = "M_newBatch"
    alias _FreeFnName = "M_freeBatch"
    alias _SizeFnName = "M_batchSize"
    alias _RequestAtFn = "M_batchRequestAt"
    alias _ResponseAtFn = "M_batchResponseAt"

    @staticmethod
    fn new(lib: DLHandle) -> CInferenceBatch:
        return call_dylib_func[CInferenceBatch](lib, Self._NewFnName)

    fn free(owned self, lib: DLHandle):
        call_dylib_func(lib, Self._FreeFnName, self.ptr)

    fn size(self, lib: DLHandle) -> Int64:
        return call_dylib_func[Int64](lib, Self._SizeFnName, self.ptr)

    fn request_at(self, lib: DLHandle, index: Int64) -> CInferenceRequest:
        return call_dylib_func[CInferenceRequest](
            lib, Self._RequestAtFn, self.ptr, index
        )

    fn response_at(self, lib: DLHandle, index: Int64) -> CInferenceResponse:
        return call_dylib_func[CInferenceResponse](
            lib, Self._ResponseAtFn, self.ptr, index
        )


struct InferenceBatch(Sized, CollectionElement):
    var _ptr: CInferenceBatch
    var _lib: DLHandle
    var _session: InferenceSession

    fn __init__(inout self, lib: DLHandle, owned session: InferenceSession):
        self._ptr = CInferenceBatch.new(lib)
        self._lib = lib
        self._session = session^

    fn __init__(
        inout self,
        ptr: CInferenceBatch,
        lib: DLHandle,
        owned session: InferenceSession,
    ):
        self._ptr = ptr
        self._lib = lib
        self._session = session^

    fn __moveinit__(inout self: Self, owned existing: Self):
        self._ptr = exchange[CInferenceBatch](
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

    fn request_at(self, index: Int64) -> InferenceRequestImpl:
        return InferenceRequestImpl(
            self._ptr.request_at(self._lib, index),
            self._lib,
            self._session,
        )

    fn response_at(self, index: Int64) -> InferenceResponseImpl:
        return InferenceResponseImpl(
            self._ptr.response_at(self._lib, index),
            self._lib,
            self._session,
        )

    fn requests(self) -> List[InferenceRequestImpl]:
        var requests = List[InferenceRequestImpl](capacity=len(self))
        for i in range(len(self)):
            requests.append(self.request_at(i))
        return requests^

    fn responses(self) -> List[InferenceResponseImpl]:
        var responses = List[InferenceResponseImpl](capacity=len(self))
        for i in range(len(self)):
            responses.append(self.response_at(i))
        return responses^


# ===----------------------------------------------------------------------=== #
# InferenceRequest
# ===----------------------------------------------------------------------=== #


@value
@register_passable("trivial")
struct CInferenceRequest:
    var ptr: DTypePointer[DType.invalid]

    alias _APITypeFnName = "M_InferenceRequest_apiType"
    alias _PayloadTypeFnName = "M_InferenceRequest_payloadType"

    alias _ModelNameFnName = "M_InferenceRequest_modelName"
    alias _ModelVersionFnName = "M_InferenceRequest_modelVersion"

    alias _InputsSizeFnName = "M_InferenceRequest_inputsSize"
    alias _InputAtFnName = "M_InferenceRequest_inputAt"
    alias _AddInputFnName = "M_InferenceRequest_addInput"

    alias _OutputsSizeFnName = "M_InferenceRequest_outputsSize"
    alias _OutputAtFnName = "M_InferenceRequest_outputAt"
    alias _AddOutputFnName = "M_InferenceRequest_addOutput"

    fn api_type(owned self, lib: DLHandle) -> Int64:
        return call_dylib_func[Int64](lib, Self._APITypeFnName, self.ptr)

    fn payload_type(owned self, lib: DLHandle) -> Int64:
        return call_dylib_func[Int64](lib, Self._PayloadTypeFnName, self.ptr)

    fn model_name(owned self, lib: DLHandle) -> CString:
        return call_dylib_func[CString](lib, Self._ModelNameFnName, self.ptr)

    fn model_version(owned self, lib: DLHandle) -> CString:
        return call_dylib_func[CString](lib, Self._ModelVersionFnName, self.ptr)

    fn inputs_size(owned self, lib: DLHandle) -> Int64:
        return call_dylib_func[Int64](lib, Self._InputsSizeFnName, self.ptr)

    fn input_at(owned self, lib: DLHandle, index: Int64) -> CString:
        return call_dylib_func[CString](
            lib, Self._OutputAtFnName, self.ptr, index
        )

    fn add_input(owned self, lib: DLHandle, name: StringRef):
        call_dylib_func[NoneType](lib, Self._AddOutputFnName, self.ptr, name)

    fn outputs_size(owned self, lib: DLHandle) -> Int64:
        return call_dylib_func[Int64](lib, Self._OutputsSizeFnName, self.ptr)

    fn output_at(owned self, lib: DLHandle, index: Int64) -> CString:
        return call_dylib_func[CString](
            lib, Self._OutputAtFnName, self.ptr, index
        )

    fn add_output(owned self, lib: DLHandle, name: StringRef):
        call_dylib_func[NoneType](lib, Self._AddOutputFnName, self.ptr, name)


struct InferenceRequestImpl(InferenceRequest):
    var _ptr: CInferenceRequest
    var _lib: DLHandle
    var _session: InferenceSession

    fn __init__(
        inout self,
        ptr: CInferenceRequest,
        lib: DLHandle,
        owned session: InferenceSession,
    ):
        self._ptr = ptr
        self._lib = lib
        self._session = session^

    fn __moveinit__(inout self, owned existing: Self):
        self._ptr = exchange[CInferenceRequest](
            existing._ptr, DTypePointer[DType.invalid]()
        )
        self._lib = existing._lib
        self._session = existing._session^

    fn __copyinit__(inout self, existing: Self):
        self._ptr = existing._ptr
        self._lib = existing._lib
        self._session = existing._session

    fn get_api_type(self) -> Int64:
        return self._ptr.api_type(self._lib)

    fn get_payload_type(self) -> Int64:
        return self._ptr.payload_type(self._lib)

    fn get_model_name(self) raises -> String:
        return str(self._ptr.model_name(self._lib))

    fn get_model_version(self) raises -> String:
        return str(self._ptr.model_version(self._lib))

    fn get_input_tensors(self) raises -> TensorMap:
        return get_tensors[
            CInferenceRequest._InputsSizeFnName,
            CInferenceRequest._InputAtFnName,
            CInferenceRequest,
        ](self._lib, self._ptr, self._session)

    fn set_input_tensors(self, names: List[String], map: TensorMap) raises:
        set_tensors[
            CInferenceRequest._AddInputFnName,
            CInferenceRequest,
        ](self._lib, self._ptr, names, map)

    fn get_outputs(self) -> List[String]:
        # TODO: Pass back an array.
        var result = List[String](
            capacity=int(self._ptr.outputs_size(self._lib))
        )
        for i in range(self._ptr.outputs_size(self._lib)):
            result.append(self._ptr.output_at(self._lib, i).__str__())
        return result^

    fn set_outputs(self, outputs: List[String]) -> None:
        for output in outputs:
            self._ptr.add_output(self._lib, output[]._strref_dangerous())


# ===----------------------------------------------------------------------=== #
# InferenceResponse
# ===----------------------------------------------------------------------=== #


@value
@register_passable("trivial")
struct CInferenceResponse:
    var ptr: DTypePointer[DType.invalid]

    alias _FreeFnName = "M_InferenceResponse_free"

    alias _APITypeFnName = "M_InferenceResponse_apiType"
    alias _PayloadTypeFnName = "M_InferenceResponse_payloadType"

    alias _OutputsSizeFnName = "M_InferenceResponse_outputsSize"
    alias _OutputAtFnName = "M_InferenceResponse_outputAt"
    alias _AddOutputFnName = "M_InferenceResponse_addOutput"

    fn api_type(owned self, lib: DLHandle) -> Int64:
        return call_dylib_func[Int64](lib, Self._APITypeFnName, self.ptr)

    fn payload_type(owned self, lib: DLHandle) -> Int64:
        return call_dylib_func[Int64](lib, Self._PayloadTypeFnName, self.ptr)

    fn free(owned self, lib: DLHandle):
        call_dylib_func(lib, Self._FreeFnName, self.ptr)


struct InferenceResponseImpl(InferenceResponse):
    var _ptr: CInferenceResponse
    var _lib: DLHandle
    var _session: InferenceSession
    var _owning: Bool

    fn __init__(
        inout self,
        ptr: CInferenceResponse,
        lib: DLHandle,
        owned session: InferenceSession,
        owning: Bool = False,
    ):
        self._ptr = ptr
        self._lib = lib
        self._session = session^
        self._owning = owning

    fn __moveinit__(inout self, owned existing: Self):
        self._ptr = exchange[CInferenceResponse](
            existing._ptr, DTypePointer[DType.invalid]()
        )
        self._lib = existing._lib
        self._session = existing._session^
        self._owning = existing._owning

    fn __copyinit__(inout self, existing: Self):
        self._ptr = existing._ptr
        self._lib = existing._lib
        self._session = existing._session
        self._owning = existing._owning

    fn __del__(owned self):
        if self._owning:
            self._ptr.free(self._lib)

    fn get_api_type(self) -> Int64:
        return self._ptr.api_type(self._lib)

    fn get_payload_type(self) -> Int64:
        return self._ptr.payload_type(self._lib)

    fn get_output_tensors(self) raises -> TensorMap:
        return get_tensors[
            CInferenceResponse._OutputsSizeFnName,
            CInferenceResponse._OutputAtFnName,
            CInferenceResponse,
        ](self._lib, self._ptr, self._session)

    fn set_output_tensors(self, names: List[String], map: TensorMap) raises:
        set_tensors[
            CInferenceResponse._AddOutputFnName,
            CInferenceResponse,
        ](self._lib, self._ptr, names, map)


# ===----------------------------------------------------------------------=== #
# Batch
# ===----------------------------------------------------------------------=== #

# TODO: Migrate to ChainPromise and delete M_asyncBatch* functions.


@value
@register_passable("trivial")
struct AsyncCInferenceBatch:
    var ptr: DTypePointer[DType.invalid]

    alias _AsyncAndThenFnName = "M_asyncBatchAndThen"
    alias _GetFnName = "M_asyncBatchGet"
    alias _FreeValueFnName = "M_freeAsyncBatch"

    fn get(self, lib: DLHandle) -> CInferenceBatch:
        return call_dylib_func[CInferenceBatch](lib, Self._GetFnName, self)

    fn free(self, lib: DLHandle):
        call_dylib_func(lib, Self._FreeValueFnName, self)


struct AwaitableCInferenceBatch:
    var _ptr: AsyncCInferenceBatch
    var _lib: DLHandle

    fn __init__(
        inout self,
        ptr: AsyncCInferenceBatch,
        lib: DLHandle,
    ):
        self._ptr = ptr
        self._lib = lib

    fn __del__(owned self):
        self._ptr.free(self._lib)

    @always_inline
    fn __await__(self) -> CInferenceBatch:
        @always_inline
        @parameter
        fn await_body(cur_hdl: AnyCoroutine):
            call_dylib_func(
                self._lib,
                AsyncCInferenceBatch._AsyncAndThenFnName,
                _coro_resume_fn,
                self._ptr,
                cur_hdl,
            )

        _suspend_async[await_body]()
        return self._ptr.get(self._lib)


# ===----------------------------------------------------------------------=== #
# ServerAsync
# ===----------------------------------------------------------------------=== #


@value
@register_passable("trivial")
struct CServerAsync:
    var ptr: DTypePointer[DType.invalid]

    alias _NewFnName = "M_newMuttServer"
    alias _FreeFnName = "M_freeMuttServer"

    alias _InitFnName = "M_muttInit"
    alias _RunFnName = "M_muttRun"
    alias _StopFnName = "M_muttStopServer"
    alias _SignalStopFnName = "M_muttSignalStopServer"

    alias _PopReadyFnName = "M_muttPopReady"
    alias _PushCompleteFnName = "M_muttPushComplete"
    alias _PushFailedFnName = "M_muttPushFailed"

    @staticmethod
    fn new(lib: DLHandle, address: StringRef) -> CServerAsync:
        return call_dylib_func[CServerAsync](lib, Self._NewFnName, address)

    fn free(owned self, lib: DLHandle):
        call_dylib_func(lib, Self._FreeFnName, self.ptr)

    fn init(owned self, lib: DLHandle, models: List[CCompiledModel]):
        call_dylib_func(
            lib, Self._InitFnName, self.ptr, models.data, len(models)
        )

    fn run(owned self, lib: DLHandle):
        call_dylib_func(lib, Self._RunFnName, self.ptr)

    fn stop(owned self, lib: DLHandle):
        call_dylib_func(lib, Self._StopFnName, self.ptr)

    fn signal_stop(owned self, lib: DLHandle):
        call_dylib_func(lib, Self._SignalStopFnName, self.ptr)

    async fn pop_ready(owned self, lib: DLHandle) -> CInferenceBatch:
        var ptr = call_dylib_func[AsyncCInferenceBatch](
            lib, self._PopReadyFnName, self.ptr
        )
        var batch = AwaitableCInferenceBatch(ptr, lib)
        return await batch

    fn push_complete(
        owned self,
        lib: DLHandle,
        batch: CInferenceBatch,
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
        owned self,
        lib: DLHandle,
        batch: CInferenceBatch,
        index: Int64,
        message: String,
    ):
        call_dylib_func(
            lib,
            self._PushFailedFnName,
            self.ptr,
            batch.ptr,
            index,
            message._strref_dangerous(),
        )


struct ServerAsync:
    var _ptr: CServerAsync
    var _lib: DLHandle
    var _session: InferenceSession

    fn __init__(
        inout self,
        address: String,
        lib: DLHandle,
        owned session: InferenceSession,
    ):
        self._ptr = CServerAsync.new(lib, address._strref_dangerous())
        self._lib = lib
        self._session = session^

    fn __moveinit__(inout self: Self, owned existing: Self):
        self._ptr = exchange[CServerAsync](
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

    fn stop(self):
        self._ptr.stop(self._lib)

    fn signal_stop(self):
        self._ptr.signal_stop(self._lib)

    async fn pop_ready(self, inout batch: InferenceBatch) -> None:
        var ptr = await self._ptr.pop_ready(self._lib)
        batch = InferenceBatch(ptr, self._lib, self._session)

    fn push_complete(self, batch: InferenceBatch, index: Int64):
        self._ptr.push_complete(self._lib, batch._ptr, index)

    fn push_failed(self, batch: InferenceBatch, index: Int64, message: String):
        self._ptr.push_failed(self._lib, batch._ptr, index, message)
