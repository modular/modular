# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Provides basic Python server wrapping."""

from memory.unsafe import DTypePointer
from python.python import _get_global_python_itf
from sys.ffi import DLHandle

from max.engine._utils import call_dylib_func

from .server import ServerAsync
from .http.runtime import PythonEntry
from ._serve_rt import (
    InferenceRequestImpl,
    InferenceResponseImpl,
)


struct ProtocolHandler:
    var _ptr: DTypePointer[DType.invalid]
    var _lib: DLHandle

    fn __init__(inout self, lib: DLHandle, ptr: DTypePointer[DType.invalid]):
        self._lib = lib
        self._ptr = ptr

    fn handle_python(
        self,
        request: InferenceRequestImpl,
        func: fn (PythonEntry) escaping -> None,
    ):
        var api_type = request.get_api_type()
        var payload_type = request.get_payload_type()
        if api_type == 1:
            # OpenAI
            if payload_type == 0:
                # gRPC
                print("OpenAI API compatibility is only supported via HTTP.")
            else:
                # HTTP
                var entry = PythonEntry()
                var cpython = _get_global_python_itf().cpython()
                call_dylib_func[NoneType](
                    self._lib,
                    "M_OpenAIInferenceRequest_fillEntry",
                    self._ptr,
                    request._impl._ptr,
                    UnsafePointer.address_of(entry),
                )
                var state = cpython.PyGILState_Ensure()
                func(entry)
                cpython.PyGILState_Release(state)

    fn handle_ok(
        inout self,
        request: InferenceRequestImpl,
        response: InferenceResponseImpl,
    ):
        fn handle(entry: PythonEntry) -> None:
            var response = PythonObject(entry.response)
            var handler = PythonObject(entry.handler)
            # TODO: Assign response.choices from output tensors!
            var cpython = _get_global_python_itf().cpython()
            cpython.Py_IncRef(response.py_object)

        self.handle_python(request, handle)

    fn handle_fail(inout self, request: InferenceRequestImpl, error: String):
        fn handle(entry: PythonEntry) -> None:
            var handler = PythonObject(entry.handler)
            try:
                handler.error = error
            except e:
                print(e)

        self.handle_python(request, handle)
