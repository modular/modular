# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Provides C bindings to KServe and basic RPC server implementations."""


from algorithm import parallelize
from collections import Dict
from memory.unsafe import DTypePointer
from runtime.llcl import (
    MojoCallContextPtr,
    MojoCallRaisingTask,
    MojoCallTask,
    Runtime,
    TaskGroup,
    TaskGroupTask,
    TaskGroupTaskList,
)
from sys.ffi import DLHandle
from tensor import Tensor, TensorSpec

from max.engine import InferenceSession, Model, TensorMap
from max.engine._utils import (
    CString,
    call_dylib_func,
    exchange,
    handle_from_config,
)

from ._mutt import Batch, MuttServerAsync
from ._c import TensorView
from .service import (
    InferenceRequest,
    InferenceResponse,
    InferenceService,
    FileModel,
)

# ===----------------------------------------------------------------------=== #
# Utilities
# ===----------------------------------------------------------------------=== #


fn get_tensor_spec(view: TensorView) -> TensorSpec:
    var dtype = view.dtype
    var shape = List[Int](capacity=view.shapeSize)
    for i in range(view.shapeSize):
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
    T: AnyRegType,
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
    add_tensor_contents_fn: StringLiteral,
    T: AnyRegType,
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

        call_dylib_func[NoneType](
            lib,
            add_tensor_fn,
            ptr,
            name._strref_dangerous(),
            dtype_str,
            shape.data,
            spec.rank(),
        )
        name._strref_keepalive()
        _ = shape^

        call_dylib_func(lib, add_tensor_contents_fn, ptr, contents_str)


# ===----------------------------------------------------------------------=== #
# KServe C bindings
# ===----------------------------------------------------------------------=== #


@value
@register_passable("trivial")
struct CModelInferRequest:
    var ptr: DTypePointer[DType.invalid]

    alias _OutputsSizeFnName = "M_modelInferRequestOutputsSize"
    alias _OutputNameFnName = "M_modelInferRequestOutputName"

    fn outputs_size(owned self, lib: DLHandle) -> Int64:
        return call_dylib_func[Int64](lib, Self._OutputsSizeFnName, self.ptr)

    fn output_name(owned self, lib: DLHandle, index: Int64) -> CString:
        return call_dylib_func[CString](
            lib, Self._OutputNameFnName, self.ptr, index
        )


struct ModelInferRequest(InferenceRequest, CollectionElement):
    var _ptr: CModelInferRequest
    var _lib: DLHandle
    var _session: InferenceSession

    fn __init__(
        inout self,
        ptr: CModelInferRequest,
        lib: DLHandle,
        owned session: InferenceSession,
    ):
        self._ptr = ptr
        self._lib = lib
        self._session = session^

    fn __moveinit__(inout self, owned existing: Self):
        self._ptr = exchange[CModelInferRequest](
            existing._ptr, DTypePointer[DType.invalid]()
        )
        self._lib = existing._lib
        self._session = existing._session^

    fn __copyinit__(inout self, existing: Self):
        self._ptr = existing._ptr
        self._lib = existing._lib
        self._session = existing._session

    fn get_input_tensors(self) raises -> TensorMap:
        return get_tensors[
            "M_modelInferRequestInputsSize",
            "M_modelInferRequestInput",
            CModelInferRequest,
        ](self._lib, self._ptr, self._session)

    fn get_requested_outputs(self) -> List[String]:
        var result = List[String](
            capacity=int(self._ptr.outputs_size(self._lib))
        )
        for i in range(self._ptr.outputs_size(self._lib)):
            result.append(self._ptr.output_name(self._lib, i).__str__())
        return result^

    fn set_input_tensors(self, names: List[String], map: TensorMap) raises:
        set_tensors[
            "M_modelInferRequestAddInput",
            "M_modelInferRequestAddRawInputContents",
            CModelInferRequest,
        ](self._lib, self._ptr, names, map)


@value
@register_passable("trivial")
struct CModelInferResponse:
    var ptr: DTypePointer[DType.invalid]


struct ModelInferResponse(InferenceResponse, CollectionElement):
    var _ptr: CModelInferResponse
    var _lib: DLHandle
    var _session: InferenceSession

    fn __init__(
        inout self,
        ptr: CModelInferResponse,
        lib: DLHandle,
        owned session: InferenceSession,
    ):
        self._ptr = ptr
        self._lib = lib
        self._session = session^

    fn __moveinit__(inout self, owned existing: Self):
        self._ptr = exchange[CModelInferResponse](
            existing._ptr, DTypePointer[DType.invalid]()
        )
        self._lib = existing._lib
        self._session = existing._session^

    fn __copyinit__(inout self, existing: Self):
        self._ptr = existing._ptr
        self._lib = existing._lib
        self._session = existing._session

    fn get_output_tensors(self) raises -> TensorMap:
        return get_tensors[
            "M_modelInferRequestOutputsSize",
            "M_modelInferRequestOutput",
            CModelInferResponse,
        ](self._lib, self._ptr, self._session)

    fn set_output_tensors(self, names: List[String], map: TensorMap) raises:
        set_tensors[
            "M_modelInferResponseAddOutput",
            "M_modelInferResponseAddRawOutputContents",
            CModelInferResponse,
        ](self._lib, self._ptr, names, map)


# ===----------------------------------------------------------------------=== #
# Server-related
# ===----------------------------------------------------------------------=== #


struct SingleModelInferenceService(InferenceService):
    """Inference service that serves a single model."""

    var _model: Model
    var _session: InferenceSession

    fn __init__(
        inout self,
        model: FileModel,
        owned session: InferenceSession,
    ) raises:
        self._session = session^
        self._model = self._session.load(model.path)

    fn infer[
        req_type: InferenceRequest, resp_type: InferenceResponse
    ](self, request: req_type, inout response: resp_type) -> None:
        _ = self.async_infer(request, response)()

    async fn async_infer[
        req_type: InferenceRequest, resp_type: InferenceResponse
    ](self, request: req_type, inout response: resp_type) -> None:
        try:
            var inputs = request.get_input_tensors()
            var outputs = self._model.execute(inputs^)
            response.set_output_tensors(
                request.get_requested_outputs(), outputs
            )
        except e:
            pass


struct GRPCInferenceServer:
    """Inference server implementing the KServe protocol over gRPC."""

    var _lib: DLHandle
    var _session: InferenceSession
    var _impl: MuttServerAsync
    var _num_listeners: Int

    fn __init__(
        inout self,
        address: String,
        owned session: InferenceSession,
        num_listeners: Int = 8,
    ) raises:
        """Constructs a gRPC inference server.

        Args:
            address: Address to serve on.
            session: Current inference context.
            num_listeners: Number of listener tasks.
        """

        self._lib = handle_from_config("serving", "max.serve_lib")
        self._session = session^
        self._impl = MuttServerAsync(address, self._lib, self._session)
        self._num_listeners = num_listeners

    fn serve[
        service_type: InferenceService
    ](self, service: service_type) -> None:
        @always_inline
        @parameter
        async fn handle(
            request: ModelInferRequest, inout response: ModelInferResponse
        ) capturing -> None:
            service.infer(request, response)

        self.serve[handle]()

    fn serve[
        handle_fn:
        async fn (ModelInferRequest, inout ModelInferResponse) capturing -> None
    ](self) -> None:
        var rt = Runtime()
        var task = rt.create_task[NoneType](self.async_serve[handle_fn]())
        task.wait()

    async fn async_serve[
        handle_fn:
        async fn (ModelInferRequest, inout ModelInferResponse) capturing -> None
    ](self) -> None:
        @always_inline
        @parameter
        async fn process(batch: Batch) capturing -> None:
            var requests = batch.requests()
            var responses = batch.responses()
            for i in range(len(batch)):
                var req = requests[i]
                var resp = responses[i]
                await handle_fn(req, resp)
                self._impl.push_complete(batch, i)
                _ = resp^

        @always_inline
        @parameter
        async fn listen() capturing -> None:
            while True:
                var batch = Batch(self._lib, self._session)
                await self._impl.async_pop_ready(batch)
                await process(batch)
                _ = batch^

        self._impl.run()

        var tasks = TaskGroupTaskList[NoneType](self._num_listeners)
        var rt = Runtime()
        var tg = TaskGroup(rt)
        for i in range(self._num_listeners):
            var task = tg.create_task[NoneType](listen())
            tasks.add(task^)

        await tg
        _ = tasks^
