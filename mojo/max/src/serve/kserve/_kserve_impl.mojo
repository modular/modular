# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Provides C bindings to KServe data structures."""

from memory.unsafe import DTypePointer
from sys.ffi import DLHandle

from max.engine import InferenceSession, TensorMap
from max.engine._utils import (
    CString,
    call_dylib_func,
    exchange,
)
from max.serve.service import InferenceRequest, InferenceResponse
from max.tensor import TensorSpec

from ._serve_rt import TensorView

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

    alias _ModelNameFnName = "M_modelInferRequestModelName"
    alias _ModelVersionFnName = "M_modelInferRequestModelVersion"
    alias _OutputsSizeFnName = "M_modelInferRequestOutputsSize"
    alias _OutputNameFnName = "M_modelInferRequestOutputName"
    alias _AddOutputFnName = "M_modelInferRequestAddOutput"

    fn model_name(owned self, lib: DLHandle) -> CString:
        return call_dylib_func[CString](lib, Self._ModelNameFnName, self.ptr)

    fn model_version(owned self, lib: DLHandle) -> CString:
        return call_dylib_func[CString](lib, Self._ModelVersionFnName, self.ptr)

    fn outputs_size(owned self, lib: DLHandle) -> Int64:
        return call_dylib_func[Int64](lib, Self._OutputsSizeFnName, self.ptr)

    fn output_name(owned self, lib: DLHandle, index: Int64) -> CString:
        return call_dylib_func[CString](
            lib, Self._OutputNameFnName, self.ptr, index
        )

    fn add_output(owned self, lib: DLHandle, name: StringRef):
        call_dylib_func[NoneType](lib, Self._AddOutputFnName, self.ptr, name)


struct ModelInferRequest(InferenceRequest):
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

    fn get_model_name(self) raises -> String:
        return self._ptr.model_name(self._lib)

    fn get_model_version(self) raises -> String:
        return self._ptr.model_version(self._lib)

    fn get_input_tensors(self) raises -> TensorMap:
        return get_tensors[
            "M_modelInferRequestInputsSize",
            "M_modelInferRequestInput",
            CModelInferRequest,
        ](self._lib, self._ptr, self._session)

    fn set_input_tensors(self, names: List[String], map: TensorMap) raises:
        set_tensors[
            "M_modelInferRequestAddInput",
            "M_modelInferRequestAddRawInputContents",
            CModelInferRequest,
        ](self._lib, self._ptr, names, map)

    fn get_requested_outputs(self) -> List[String]:
        # TODO: Pass back an array.
        var result = List[String](
            capacity=int(self._ptr.outputs_size(self._lib))
        )
        for i in range(self._ptr.outputs_size(self._lib)):
            result.append(self._ptr.output_name(self._lib, i).__str__())
        return result^

    fn set_requested_outputs(self, outputs: List[String]) -> None:
        for output in outputs:
            self._ptr.add_output(self._lib, output[]._strref_dangerous())


@value
@register_passable("trivial")
struct CModelInferResponse:
    var ptr: DTypePointer[DType.invalid]

    alias _NewFnName = "M_newModelInferResponse"
    alias _FreeFnName = "M_freeModelInferResponse"

    @staticmethod
    fn new(lib: DLHandle) -> CModelInferResponse:
        return call_dylib_func[CModelInferResponse](lib, Self._NewFnName)

    fn free(owned self, lib: DLHandle):
        call_dylib_func(lib, Self._FreeFnName, self.ptr)


struct ModelInferResponse(InferenceResponse):
    var _ptr: CModelInferResponse
    var _lib: DLHandle
    var _session: InferenceSession
    var _owning: Bool

    fn __init__(
        inout self,
        ptr: CModelInferResponse,
        lib: DLHandle,
        owned session: InferenceSession,
        owning: Bool = False,
    ):
        self._ptr = ptr
        self._lib = lib
        self._session = session^
        self._owning = owning

    fn __init__(
        inout self,
        inout ptr: DTypePointer[DType.invalid],
        lib: DLHandle,
        owned session: InferenceSession,
        owning: Bool = False,
    ):
        self._ptr = ptr
        # TODO - SERV-119 - Fix ownership of response ptr
        ptr = DTypePointer[DType.invalid]()
        self._lib = lib
        self._session = session^
        self._owning = owning

    fn __moveinit__(inout self, owned existing: Self):
        self._ptr = exchange[CModelInferResponse](
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

    fn get_output_tensors(self) raises -> TensorMap:
        return get_tensors[
            "M_modelInferResponseOutputsSize",
            "M_modelInferResponseOutput",
            CModelInferResponse,
        ](self._lib, self._ptr, self._session)

    fn set_output_tensors(self, names: List[String], map: TensorMap) raises:
        set_tensors[
            "M_modelInferResponseAddOutput",
            "M_modelInferResponseAddRawOutputContents",
            CModelInferResponse,
        ](self._lib, self._ptr, names, map)
