# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Provides wrapper around a Python server.

  from max.serve.http import PythonServer, PythonService

  struct MyHandler(PythonService):
      async fn handle(inout self, inout body: PythonObject, inout handler: PythonObject):
        if handler.method == "POST" and handler.path == "/v1/api":
            handler.wfile.write(...)
"""

from algorithm import parallelize
from collections import Optional
from sys.ffi import DLHandle
from memory.unsafe_pointer import UnsafePointer
from python import PythonObject
from python.python import _get_global_python_itf
from runtime.asyncrt import TaskGroup
from time import perf_counter_ns
from utils import StringRef

from max._utils import handle_from_config

from ..util.callbacks import (
    ServerCallbacks,
    NoopServerCallbacks,
    Guarded,
    CallbackSet,
)
from ..util.stats import ServerStats
from ..util.config import STATS_ENABLED
from .._batch import CBatch

from ._server import CPythonRequest, CPythonResponse, CPythonServer
from .service import PythonService


struct PythonBatch(Sized, Movable):
    """A wrapper over a CBatch instance to be used with Python modules."""

    var _impl: CBatch

    fn __init__(inout self, lib: DLHandle):
        self._impl = CBatch(lib)

    fn __moveinit__(inout self: Self, owned existing: Self):
        self._impl = existing._impl^

    fn __len__(self) -> Int:
        return int(self._impl.size())

    fn request_at(self, index: Int64) -> PythonObject:
        """Returns the request at the specified index in the form of a python
        object.

        Args:
            index: The index of the request to access.

        Returns:
            A python object corresponding to the request at the index.

        """
        return self._impl.request_at[CPythonRequest](index).load()

    fn response_at(self, index: Int64) -> PythonObject:
        """Returns the response at the specified index in the form of a python
        object.

        Args:
            index: The index of the response to access.

        Returns:
            A python object corresponding to the response at the index.

        """
        return self._impl.response_at[CPythonResponse](index).load()


struct PythonServer[
    Callbacks: ServerCallbacks = NoopServerCallbacks,
]:
    """Server allowing access to a Python handler."""

    alias handle_fn_type = fn (
        owned PythonObject, owned PythonObject
    ) capturing raises -> None

    var _lib: DLHandle
    var _impl: CPythonServer

    var _callbacks: CallbackSet[
        Guarded[ServerStats, STATS_ENABLED],
        Callbacks,
    ]

    @staticmethod
    fn create(address: String) raises -> PythonServer:
        return PythonServer(address, NoopServerCallbacks())

    fn __init__(
        inout self,
        address: String,
        owned callbacks: Callbacks,
    ) raises:
        """Constructs an generic Python server.

        Args:
            address: Address to serve on.
            callbacks: Extra lifecycle callbacks.
        """
        self._lib = handle_from_config("serving", ".serve_lib")
        self._impl = CPythonServer(
            self._lib, StringRef(ptr=address.unsafe_ptr())
        )
        self._callbacks = CallbackSet(
            Guarded[ServerStats, STATS_ENABLED](ServerStats()),
            callbacks^,
        )
        self._callbacks.on_server_start()

    fn __del__(owned self):
        self._callbacks.on_server_stop()

    fn serve[
        service_type: PythonService
    ](
        inout self, inout service: service_type, num_listeners: Int = 4
    ) raises -> None:
        @always_inline
        @parameter
        fn handle(
            owned body: PythonObject,
            owned handler: PythonObject,
        ) raises -> None:
            service.handle(body^, handler^)

        self.serve[handle](num_listeners=num_listeners)

    fn serve[
        handle_fn: Self.handle_fn_type
    ](inout self, num_listeners: Int = 4) raises -> None:
        var tg = TaskGroup()
        var cpython = _get_global_python_itf().cpython()
        for i in range(num_listeners):
            tg.create_task(self._serve[handle_fn]())
        var tstate = cpython.PyEval_SaveThread()
        tg.wait()
        cpython.PyEval_RestoreThread(tstate)

    async fn _serve[handle_fn: Self.handle_fn_type](inout self) -> None:
        var cpython = _get_global_python_itf().cpython()
        while True:
            var batch = PythonBatch(self._lib)
            await self._impl.pop_ready(batch._impl)
            self._callbacks.on_batch_receive(len(batch))
            var start = perf_counter_ns()
            var state = cpython.PyGILState_Ensure()
            for i in range(len(batch)):
                self._callbacks.on_request_receive()
                try:
                    handle_fn(batch.request_at(i), batch.response_at(i))
                    self._callbacks.on_request_ok(start)
                    batch._impl.push_complete(i)
                except err:
                    self._callbacks.on_request_fail(str(err))
                    batch._impl.push_failed(i, str(err))
            cpython.PyGILState_Release(state)
            self._callbacks.on_batch_complete(start, len(batch))
