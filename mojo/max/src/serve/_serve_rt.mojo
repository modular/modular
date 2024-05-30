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
](
    lib: DLHandle, ptr: DTypePointer[DType.invalid], session: InferenceSession
) raises -> TensorMap:
    var map = session.new_tensor_map()
    var size = call_dylib_func[Int64](lib, size_fn, ptr)
    for i in range(size):
        var view_ptr = Pointer[TensorView]()
        call_dylib_func(
            lib,
            get_tensor_fn,
            ptr,
            i,
            UnsafePointer.address_of(view_ptr),
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
](
    lib: DLHandle,
    ptr: DTypePointer[DType.invalid],
    names: List[String],
    map: TensorMap,
) raises:
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


struct CInferenceBatch:
    """Corresponds to the InfererenceBatch C type."""

    var _lib: DLHandle
    var _ptr: DTypePointer[DType.invalid]

    alias _NewFnName = "M_newBatch"
    alias _FreeFnName = "M_freeBatch"
    alias _SizeFnName = "M_batchSize"
    alias _RequestAtFn = "M_batchRequestAt"
    alias _ResponseAtFn = "M_batchResponseAt"

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

    fn request_at(self, index: Int64) -> CInferenceRequest:
        var ptr = DTypePointer[DType.invalid]()
        call_dylib_func[NoneType](
            self._lib,
            Self._RequestAtFn,
            self._ptr,
            index,
            UnsafePointer.address_of(ptr),
        )
        return CInferenceRequest(self._lib, ptr)

    fn response_at(self, index: Int64) -> CInferenceResponse:
        var ptr = DTypePointer[DType.invalid]()
        call_dylib_func[NoneType](
            self._lib,
            Self._ResponseAtFn,
            self._ptr,
            index,
            UnsafePointer.address_of(ptr),
        )
        return CInferenceResponse(self._lib, ptr)


struct InferenceBatch(Sized, Movable):
    var _impl: CInferenceBatch
    var _session: InferenceSession

    fn __init__(inout self, lib: DLHandle, owned session: InferenceSession):
        self._impl = CInferenceBatch(lib)
        self._session = session^

    fn __moveinit__(inout self: Self, owned existing: Self):
        self._impl = existing._impl^
        self._session = existing._session^

    fn __len__(self) -> Int:
        return int(self._impl.size())

    fn request_at(self, index: Int64) -> InferenceRequestImpl:
        return InferenceRequestImpl(
            self._impl.request_at(index),
            self._session,
        )

    fn response_at(self, index: Int64) -> InferenceResponseImpl:
        return InferenceResponseImpl(
            self._impl.response_at(index),
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


struct CInferenceRequest:
    # These are never owned, and only refer to the existing request memory
    # within some foreign object. They could be made owned by adding a flag
    # here and handling creation/destruction internally.

    var _lib: DLHandle
    var _ptr: DTypePointer[DType.invalid]
    var _owning: Bool

    alias _FreeFnName = "M_InferenceRequest_free"
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

    fn __init__(
        inout self,
        lib: DLHandle,
        ptr: DTypePointer[DType.invalid],
        owning: Bool = False,
    ):
        self._lib = lib
        self._ptr = ptr
        self._owning = owning

    fn __moveinit__(inout self, owned existing: Self):
        self._lib = existing._lib
        self._ptr = exchange[DTypePointer[DType.invalid]](
            existing._ptr, DTypePointer[DType.invalid]()
        )
        self._owning = existing._owning

    fn __copyinit__(inout self, existing: Self):
        self._lib = existing._lib
        self._ptr = existing._ptr
        self._owning = False

    fn __del__(owned self):
        if self._owning:
            call_dylib_func(self._lib, Self._FreeFnName, self._ptr)

    fn api_type(self) -> Int64:
        return call_dylib_func[Int64](self._lib, Self._APITypeFnName, self._ptr)

    fn payload_type(self) -> Int64:
        return call_dylib_func[Int64](
            self._lib, Self._PayloadTypeFnName, self._ptr
        )

    fn model_name(self) -> CString:
        return call_dylib_func[CString](
            self._lib, Self._ModelNameFnName, self._ptr
        )

    fn model_version(self) -> CString:
        return call_dylib_func[CString](
            self._lib, Self._ModelVersionFnName, self._ptr
        )

    fn inputs_size(self) -> Int64:
        return call_dylib_func[Int64](
            self._lib, Self._InputsSizeFnName, self._ptr
        )

    fn input_at(self, index: Int64) -> CString:
        return call_dylib_func[CString](
            self._lib, Self._OutputAtFnName, self._ptr, index
        )

    fn add_input(self, name: StringRef):
        call_dylib_func[NoneType](
            self._lib, Self._AddOutputFnName, self._ptr, name
        )

    fn outputs_size(self) -> Int64:
        return call_dylib_func[Int64](
            self._lib, Self._OutputsSizeFnName, self._ptr
        )

    fn output_at(self, index: Int64) -> CString:
        return call_dylib_func[CString](
            self._lib, Self._OutputAtFnName, self._ptr, index
        )

    fn add_output(self, name: StringRef):
        call_dylib_func[NoneType](
            self._lib, Self._AddOutputFnName, self._ptr, name
        )


@value
struct InferenceRequestImpl(InferenceRequest):
    var _impl: CInferenceRequest
    var _session: InferenceSession

    fn __init__(
        inout self,
        owned impl: CInferenceRequest,
        owned session: InferenceSession,
    ):
        self._impl = impl^
        self._session = session^

    fn get_api_type(self) -> Int64:
        return self._impl.api_type()

    fn get_payload_type(self) -> Int64:
        return self._impl.payload_type()

    fn get_ptr(self) -> DTypePointer[DType.invalid]:
        return self._impl._ptr

    fn get_model_name(self) raises -> String:
        return str(self._impl.model_name())

    fn get_model_version(self) raises -> String:
        return str(self._impl.model_version())

    fn get_input_tensors(self) raises -> TensorMap:
        return get_tensors[
            CInferenceRequest._InputsSizeFnName,
            CInferenceRequest._InputAtFnName,
        ](self._impl._lib, self._impl._ptr, self._session)

    fn set_input_tensors(self, names: List[String], map: TensorMap) raises:
        set_tensors[CInferenceRequest._AddInputFnName,](
            self._impl._lib, self._impl._ptr, names, map
        )

    fn get_outputs(self) -> List[String]:
        # TODO: Pass back an array.
        var result = List[String](capacity=int(self._impl.outputs_size()))
        for i in range(self._impl.outputs_size()):
            result.append(self._impl.output_at(i).__str__())
        return result^

    fn set_outputs(self, outputs: List[String]) -> None:
        for output in outputs:
            self._impl.add_output(output[]._strref_dangerous())


# ===----------------------------------------------------------------------=== #
# InferenceResponse
# ===----------------------------------------------------------------------=== #


struct CInferenceResponse:
    var _lib: DLHandle
    var _ptr: DTypePointer[DType.invalid]
    var _owning: Bool

    alias _FreeFnName = "M_InferenceResponse_free"

    alias _APITypeFnName = "M_InferenceResponse_apiType"
    alias _PayloadTypeFnName = "M_InferenceResponse_payloadType"

    alias _OutputsSizeFnName = "M_InferenceResponse_outputsSize"
    alias _OutputAtFnName = "M_InferenceResponse_outputAt"
    alias _AddOutputFnName = "M_InferenceResponse_addOutput"

    fn __init__(
        inout self,
        lib: DLHandle,
        ptr: DTypePointer[DType.invalid],
        owning: Bool = False,
    ):
        self._lib = lib
        self._ptr = ptr
        self._owning = owning

    fn __moveinit__(inout self, owned existing: Self):
        self._lib = existing._lib
        self._ptr = exchange[DTypePointer[DType.invalid]](
            existing._ptr, DTypePointer[DType.invalid]()
        )
        self._owning = existing._owning

    fn __copyinit__(inout self, existing: Self):
        self._lib = existing._lib
        self._ptr = existing._ptr
        self._owning = False

    fn __del__(owned self):
        if self._owning:
            call_dylib_func(self._lib, Self._FreeFnName, self._ptr)

    fn api_type(self) -> Int64:
        return call_dylib_func[Int64](self._lib, Self._APITypeFnName, self._ptr)

    fn payload_type(self) -> Int64:
        return call_dylib_func[Int64](
            self._lib, Self._PayloadTypeFnName, self._ptr
        )


@value
struct InferenceResponseImpl(InferenceResponse):
    var _impl: CInferenceResponse
    var _session: InferenceSession

    fn __init__(
        inout self,
        owned impl: CInferenceResponse,
        owned session: InferenceSession,
    ):
        self._impl = impl^
        self._session = session^

    fn get_api_type(self) -> Int64:
        return self._impl.api_type()

    fn get_payload_type(self) -> Int64:
        return self._impl.payload_type()

    fn get_output_tensors(self) raises -> TensorMap:
        return get_tensors[
            CInferenceResponse._OutputsSizeFnName,
            CInferenceResponse._OutputAtFnName,
        ](self._impl._lib, self._impl._ptr, self._session)

    fn set_output_tensors(self, names: List[String], map: TensorMap) raises:
        set_tensors[CInferenceResponse._AddOutputFnName,](
            self._impl._lib, self._impl._ptr, names, map
        )


# ===----------------------------------------------------------------------=== #
# Batch
# ===----------------------------------------------------------------------=== #

# TODO: Migrate to ChainPromise and delete M_asyncBatch* functions.


struct AsyncCInferenceBatch:
    var _lib: DLHandle
    var _ptr: DTypePointer[DType.invalid]

    alias _PopReadyFnName = "M_kservePopReady"
    alias _AsyncAndThenFnName = "M_asyncBatchAndThen"
    alias _GetFnName = "M_asyncBatchGet"
    alias _FreeValueFnName = "M_freeAsyncBatch"

    fn __init__(inout self, lib: DLHandle, server: CServerAsync):
        self._lib = lib
        self._ptr = DTypePointer[DType.invalid]()
        call_dylib_func(
            self._lib,
            self._PopReadyFnName,
            server._ptr,
            UnsafePointer.address_of(self._ptr),
        )

    fn __del__(owned self):
        call_dylib_func(self._lib, Self._FreeValueFnName, self._ptr)

    @always_inline
    fn __await__(self) -> CInferenceBatch:
        @always_inline
        @parameter
        fn await_body(cur_hdl: AnyCoroutine):
            call_dylib_func(
                self._lib,
                Self._AsyncAndThenFnName,
                _coro_resume_fn,
                self._ptr,
                cur_hdl,
            )

        _suspend_async[await_body]()

        # Return the allocated batch.
        var ptr = DTypePointer[DType.invalid]()
        call_dylib_func(
            self._lib, Self._GetFnName, self._ptr, UnsafePointer.address_of(ptr)
        )
        return CInferenceBatch(self._lib, ptr)


# ===----------------------------------------------------------------------=== #
# ServerAsync
# ===----------------------------------------------------------------------=== #


struct CServerAsync:
    var _lib: DLHandle
    var _ptr: DTypePointer[DType.invalid]

    alias _NewFnName = "M_newKServeServer"
    alias _FreeFnName = "M_freeKServeServer"

    alias _InitFnName = "M_kserveInit"
    alias _RunFnName = "M_kserveRun"
    alias _StopFnName = "M_kserveStopServer"
    alias _SignalStopFnName = "M_kserveSignalStopServer"

    alias _PushCompleteFnName = "M_kservePushComplete"
    alias _PushFailedFnName = "M_kservePushFailed"

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

    fn init(inout self, models: List[CCompiledModel]):
        call_dylib_func(
            self._lib, Self._InitFnName, self._ptr, models.data, len(models)
        )

    fn run(inout self):
        call_dylib_func(self._lib, Self._RunFnName, self._ptr)

    fn stop(inout self):
        call_dylib_func(self._lib, Self._StopFnName, self._ptr)

    fn signal_stop(inout self):
        call_dylib_func(self._lib, Self._SignalStopFnName, self._ptr)

    async fn pop_ready(inout self, inout batch: CInferenceBatch):
        batch = await AsyncCInferenceBatch(self._lib, self)

    fn push_complete(
        inout self,
        batch: CInferenceBatch,
        index: Int64,
    ):
        call_dylib_func(
            self._lib,
            self._PushCompleteFnName,
            self._ptr,
            batch._ptr,
            index,
        )

    fn push_failed(
        inout self,
        batch: CInferenceBatch,
        index: Int64,
        message: String,
    ):
        call_dylib_func(
            self._lib,
            self._PushFailedFnName,
            self._ptr,
            batch._ptr,
            index,
            message._strref_dangerous(),
        )


struct ServerAsync:
    var _impl: CServerAsync
    var _session: InferenceSession

    fn __init__(
        inout self,
        address: String,
        lib: DLHandle,
        owned session: InferenceSession,
    ):
        self._impl = CServerAsync(lib, address._strref_dangerous())
        self._session = session^

    fn __moveinit__(inout self: Self, owned existing: Self):
        self._impl = existing._impl^
        self._session = existing._session^

    fn __del__(owned self):
        _ = self._session^

    fn init(inout self, model: Model):
        var ptrs = List(model._compiled_model.ptr)
        self._impl.init(ptrs)

    fn init(inout self, models: List[Model]):
        var ptrs = List[CCompiledModel](capacity=len(models))
        for model in models:
            ptrs.append(model[]._compiled_model.ptr)
        self._impl.init(ptrs)

    fn run(inout self):
        self._impl.run()

    fn stop(inout self):
        self._impl.stop()

    fn signal_stop(inout self):
        self._impl.signal_stop()

    async fn pop_ready(inout self, inout batch: InferenceBatch):
        await self._impl.pop_ready(batch._impl)

    fn push_complete(inout self, batch: InferenceBatch, index: Int64):
        self._impl.push_complete(batch._impl, index)

    fn push_failed(
        inout self, batch: InferenceBatch, index: Int64, message: String
    ):
        self._impl.push_failed(batch._impl, index, message)
