# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Provides bindings to a Python handler."""

from memory import UnsafePointer
from python import Python, PythonObject
from python.python import _get_global_python_itf
from python._cpython import PyObjectPtr
from sys.ffi import DLHandle
from utils import StringRef
from max._utils import call_dylib_func, exchange

from .._batch import CBatch, CRequest, CResponse


# ===----------------------------------------------------------------------=== #
# PythonRequest
# ===----------------------------------------------------------------------=== #


struct CPythonRequest(CRequest):
    var _lib: DLHandle
    var _ptr: UnsafePointer[NoneType]

    alias _LoadFnName = "M_pythonLoadRequest"

    fn __init__(inout self, lib: DLHandle, ptr: UnsafePointer[NoneType]):
        self._lib = lib
        self._ptr = ptr

    fn __moveinit__(inout self, owned existing: Self):
        self._lib = existing._lib
        self._ptr = exchange[UnsafePointer[NoneType]](
            existing._ptr, UnsafePointer[NoneType]()
        )

    fn load(self) -> PythonObject:
        var ptr = PyObjectPtr()
        call_dylib_func(
            self._lib,
            Self._LoadFnName,
            self._ptr,
            UnsafePointer.address_of(ptr),
        )
        return PythonObject.from_borrowed_ptr(ptr)


# ===----------------------------------------------------------------------=== #
# PythonResponse
# ===----------------------------------------------------------------------=== #


struct CPythonResponse(CResponse):
    var _lib: DLHandle
    var _ptr: UnsafePointer[NoneType]

    alias _LoadFnName = "M_pythonLoadResponse"

    fn __init__(inout self, lib: DLHandle, ptr: UnsafePointer[NoneType]):
        self._lib = lib
        self._ptr = ptr

    fn __moveinit__(inout self, owned existing: Self):
        self._lib = existing._lib
        self._ptr = exchange[UnsafePointer[NoneType]](
            existing._ptr, UnsafePointer[NoneType]()
        )

    fn load(self) -> PythonObject:
        var ptr = PyObjectPtr()
        call_dylib_func(
            self._lib,
            Self._LoadFnName,
            self._ptr,
            UnsafePointer.address_of(ptr),
        )
        return PythonObject.from_borrowed_ptr(ptr)


# ===----------------------------------------------------------------------=== #
# PythonServer
# ===----------------------------------------------------------------------=== #


struct CPythonServer:
    var _lib: DLHandle
    var _ptr: UnsafePointer[NoneType]
    var _httpd: PythonObject

    alias _NewFnName = "M_newPythonServer"
    alias _FreeFnName = "M_freePythonServer"
    alias _PopReadyFnName = "M_pythonPopReady"

    alias _Stub = """
import json                                    # disable-validate
import ctypes                                  # disable-validate
import threading                               # disable-validate
from http.server import BaseHTTPRequestHandler # disable-validate
from http.server import ThreadingHTTPServer    # disable-validate
from urllib.parse import urlparse              # disable-validate

def _load_dylib():
    return ctypes.CDLL(None)


class TrampolineHTTPServer(ThreadingHTTPServer):
    def __init__(self, ptr, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ptr = ptr


class TrampolineHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.json = json # For convenience.
        super().__init__(*args, **kwargs)

    def _handle(self, body):
        _load_dylib().M_pythonPush(
            ctypes.c_int64(self.server.ptr),
            ctypes.py_object(body),
            ctypes.py_object(self),
        )

    def do_HEAD(self):
        self._handle(None)

    def do_GET(self):
        self._handle(None)

    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        body = self.rfile.read(content_length)
        if self.headers["Content-Type"] == "application/json":
            body = json.loads(body)
            self._handle(body)
        else:
            self._handle(body)


def create(ptr, address):
    if not ptr:
        raise Exception("Must pass in non-null server pointer!")

    parsed = urlparse("//" + address)
    if not (parsed.hostname and parsed.port):
        raise Exception("Could not parse hostname and port from: " + address)

    httpd = TrampolineHTTPServer(
        ptr, (parsed.hostname, parsed.port), TrampolineHandler,
    )
    t = threading.Thread(name="httpd", target=httpd.serve_forever)
    t.start()
    return t
"""

    fn __init__(inout self, lib: DLHandle, address: StringRef) raises:
        self._lib = lib
        self._ptr = UnsafePointer[NoneType]()
        call_dylib_func(
            self._lib,
            Self._NewFnName,
            UnsafePointer.address_of(self._ptr),
        )
        var mod = Python.evaluate(Self._Stub, file=True)
        var server_ptr = UnsafePointer(self._ptr)
        self._httpd = mod.create(int(server_ptr), address)

    fn __moveinit__(inout self, owned existing: Self):
        self._lib = existing._lib
        self._ptr = exchange[UnsafePointer[NoneType]](
            existing._ptr, UnsafePointer[NoneType]()
        )
        self._httpd = existing._httpd^

    fn __del__(owned self):
        call_dylib_func(self._lib, Self._FreeFnName, self._ptr)

    async fn pop_ready(inout self, inout batch: CBatch):
        await batch.load[Self._PopReadyFnName](self._ptr)
