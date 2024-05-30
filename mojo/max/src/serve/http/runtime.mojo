# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from python import Python
from python.object import PyObjectPtr
from runtime.llcl import Runtime
from memory import UnsafePointer

from .._serve_rt import CServerAsync


@register_passable
struct PythonEntry:
    fn __init__(inout self):
        self.request = PyObjectPtr()
        self.response = PyObjectPtr()
        self.handler = PyObjectPtr()

    var request: PyObjectPtr
    var response: PyObjectPtr
    var handler: PyObjectPtr


@always_inline
@parameter
async fn run_http_rt(server: CServerAsync):
    try:
        var http_rt = Python.import_module("max.serve.http_rt")
        var server_ptr = UnsafePointer(server._ptr.address.address)
        http_rt.run(int(server_ptr), "0.0.0.0:80")
    except e:
        print(e)
