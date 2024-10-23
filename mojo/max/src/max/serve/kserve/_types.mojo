# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Provides C bindings to KServe types and data structures."""

from memory import UnsafePointer
from sys.ffi import DLHandle
from utils import StringRef, StringSlice
from max.tensor import TensorSpec

from max.engine import InferenceSession, TensorMap
from max._utils import CString, call_dylib_func, exchange


# ===----------------------------------------------------------------------=== #
# Utilities
# ===----------------------------------------------------------------------=== #


@value
@register_passable("trivial")
struct CTensorView:
    """Corresponds to the M_TensorView C type."""

    var name: StringRef
    var dtype: StringRef
    var shape: UnsafePointer[Int64]
    var rank: Int
    var contents: UnsafePointer[UInt8]
    var contentsSize: Int

    alias _FreeValueFnName = "M_freeTensorView"

    @staticmethod
    fn free(lib: DLHandle, ptr: UnsafePointer[CTensorView]):
        call_dylib_func(lib, Self._FreeValueFnName, ptr)


fn _get_tensor_spec(view: CTensorView) -> TensorSpec:
    var dtype = view.dtype
    var shape = List[Int, hint_trivial_type=True](capacity=view.rank)
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


fn _get_tensors[
    size_fn: StringLiteral,
    get_tensor_fn: StringLiteral,
](
    lib: DLHandle, ptr: UnsafePointer[NoneType], session: InferenceSession
) raises -> TensorMap:
    var map = session.new_tensor_map()
    var size = call_dylib_func[Int64](lib, size_fn, ptr)
    for i in range(size):
        var view_ptr = UnsafePointer[CTensorView]()
        call_dylib_func(
            lib,
            get_tensor_fn,
            ptr,
            i,
            UnsafePointer.address_of(view_ptr),
        )
        var view = view_ptr[]
        map.borrow(view.name, _get_tensor_spec(view), view.contents)
        CTensorView.free(lib, view_ptr)
    return map^


fn _buffer_str[type: DType](map: TensorMap, name: String) raises -> StringRef:
    var buffer = map.buffer[type](name)

    var ptr = buffer.data.bitcast[DType.int8]()

    return StringRef(ptr, buffer.bytecount())


fn _set_tensors[
    add_tensor_fn: StringLiteral,
](
    lib: DLHandle,
    ptr: UnsafePointer[NoneType],
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
            contents_str = _buffer_str[DType.bool](map, name)
        elif dtype == DType.uint8:
            dtype_str = "UINT8"
            contents_str = _buffer_str[DType.uint8](map, name)
        elif dtype == DType.uint16:
            dtype_str = "UINT16"
            contents_str = _buffer_str[DType.uint16](map, name)
        elif dtype == DType.uint32:
            dtype_str = "UINT32"
            contents_str = _buffer_str[DType.uint32](map, name)
        elif dtype == DType.uint64:
            dtype_str = "UINT64"
            contents_str = _buffer_str[DType.uint64](map, name)
        elif dtype == DType.int8:
            dtype_str = "INT8"
            contents_str = _buffer_str[DType.int8](map, name)
        elif dtype == DType.int16:
            dtype_str = "INT16"
            contents_str = _buffer_str[DType.int16](map, name)
        elif dtype == DType.int32:
            dtype_str = "INT32"
            contents_str = _buffer_str[DType.int32](map, name)
        elif dtype == DType.int64:
            dtype_str = "INT64"
            contents_str = _buffer_str[DType.int64](map, name)
        elif dtype == DType.float16:
            dtype_str = "FP16"
            contents_str = _buffer_str[DType.float16](map, name)
        elif dtype == DType.float32:
            dtype_str = "FP32"
            contents_str = _buffer_str[DType.float32](map, name)
        elif dtype == DType.float64:
            dtype_str = "FP64"
            contents_str = _buffer_str[DType.float64](map, name)
        elif dtype == DType.bfloat16:
            dtype_str = "BF16"
            contents_str = _buffer_str[DType.bfloat16](map, name)
        else:
            dtype_str = "BYTES"
            contents_str = _buffer_str[DType.int8](map, name)

        var shape = List[Int64](capacity=spec.rank())
        for i in range(spec.rank()):
            shape.append(spec[i])

        var view = CTensorView(
            name.unsafe_ptr(),
            dtype_str,
            shape.data,
            spec.rank(),
            contents_str.unsafe_ptr(),
            len(contents_str),
        )
        call_dylib_func[NoneType](
            lib, add_tensor_fn, ptr, UnsafePointer.address_of(view)
        )
        _ = view
        _ = shape^


# ===----------------------------------------------------------------------=== #
# CInferenceRequest
# ===----------------------------------------------------------------------=== #


struct CInferenceRequest:
    # These are never owned, and only refer to the existing request memory
    # within some foreign object. They could be made owned by adding a flag
    # here and handling creation/destruction internally.

    var _lib: DLHandle
    var _ptr: UnsafePointer[NoneType]
    var _owning: Bool

    alias _FreeFnName = "M_freeRequest"

    alias _ModelNameFnName = "M_kserveRequestModelName"
    alias _ModelVersionFnName = "M_kserveRequestModelVersion"

    alias _InputsSizeFnName = "M_kserveRequestInputsSize"
    alias _InputAtFnName = "M_kserveRequestInputAt"
    alias _AddInputFnName = "M_kserveRequestAddInput"

    alias _OutputsSizeFnName = "M_kserveRequestOutputsSize"
    alias _OutputAtFnName = "M_kserveRequestOutputAt"
    alias _AddOutputFnName = "M_kserveRequestAddOutput"

    fn __init__(
        inout self,
        lib: DLHandle,
        ptr: UnsafePointer[NoneType],
    ):
        self._lib = lib
        self._ptr = ptr
        self._owning = False

    fn __init__(
        inout self,
        lib: DLHandle,
        ptr: UnsafePointer[NoneType],
        owning: Bool = False,
    ):
        self._lib = lib
        self._ptr = ptr
        self._owning = owning

    fn __moveinit__(inout self, owned existing: Self):
        self._lib = existing._lib
        self._ptr = exchange[UnsafePointer[NoneType]](
            existing._ptr, UnsafePointer[NoneType]()
        )
        # Regardless of whether existing owned it or not, we copy owning.
        self._owning = existing._owning
        # But after copying it, existing doesn't own it anymore.
        existing._owning = False

    fn __copyinit__(inout self, existing: Self):
        self._lib = existing._lib
        self._ptr = existing._ptr
        self._owning = False

    fn __del__(owned self):
        if self._owning:
            call_dylib_func(self._lib, Self._FreeFnName, self._ptr)

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

    fn add_output(self, name: StringSlice):
        call_dylib_func[NoneType](
            self._lib, Self._AddOutputFnName, self._ptr, name.unsafe_ptr()
        )


# ===----------------------------------------------------------------------=== #
# CInferenceResponse
# ===----------------------------------------------------------------------=== #


struct CInferenceResponse:
    var _lib: DLHandle
    var _ptr: UnsafePointer[NoneType]
    var _owning: Bool

    alias _FreeFnName = "M_freeResponse"

    alias _OutputsSizeFnName = "M_kserveResponseOutputsSize"
    alias _OutputAtFnName = "M_kserveResponseOutputAt"
    alias _AddOutputFnName = "M_kserveResponseAddOutput"

    fn __init__(
        inout self,
        lib: DLHandle,
        ptr: UnsafePointer[NoneType],
    ):
        self._lib = lib
        self._ptr = ptr
        self._owning = False

    fn __init__(
        inout self,
        lib: DLHandle,
        ptr: UnsafePointer[NoneType],
        owning: Bool = False,
    ):
        self._lib = lib
        self._ptr = ptr
        self._owning = owning

    fn __moveinit__(inout self, owned existing: Self):
        self._lib = existing._lib
        self._ptr = exchange[UnsafePointer[NoneType]](
            existing._ptr, UnsafePointer[NoneType]()
        )
        self._owning = existing._owning
        existing._owning = False

    fn __copyinit__(inout self, existing: Self):
        self._lib = existing._lib
        self._ptr = existing._ptr
        self._owning = False

    fn __del__(owned self):
        if self._owning:
            call_dylib_func(self._lib, Self._FreeFnName, self._ptr)
