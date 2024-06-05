# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Provides basic KServe service/server implementations."""

from algorithm import parallelize
from sys.ffi import DLHandle
from runtime.llcl import (
    Runtime,
    TaskGroup,
    Atomic,
)
from python.python import _get_global_python_itf, Python, CPython

from tensor import TensorSpec
from time import now
from utils.variant import Variant

from max.engine import InferenceSession, InputSpec, Model
from max.engine._model_impl import CModel
from max.engine._utils import handle_from_config, call_dylib_func
from max.serve.service import (
    InferenceRequest,
    InferenceResponse,
    InferenceService,
    FileModel,
)

from .util import (
    ServerCallbacks,
    Guarded,
    NoopServerCallbacks,
    ServerCallbacks,
    CallbackSet,
    BATCH_HEAT_MAP_ENABLED,
    STATS_ENABLED,
    ServerStats,
    ServerStatsOptions,
    BatchHeatMap,
)

from .protocol import ProtocolHandler
from .http.runtime import run_http_rt
from ._serve_rt import (
    InferenceRequestImpl,
    InferenceResponseImpl,
    InferenceBatch,
    ServerAsync,
    TensorView,
)


struct InferenceServer[
    HTTP_SERVER_ENABLED: Bool = False,
    Callbacks: ServerCallbacks = NoopServerCallbacks,
    EnableProtocol: Bool = False,
]:
    """Inference server implementing the KServe protocol over gRPC."""

    alias handle_fn_type = async fn (
        InferenceRequestImpl, inout Variant[InferenceResponseImpl, Error]
    ) capturing -> None

    var _lib: DLHandle
    var _session: InferenceSession
    var _num_listeners: Int
    var _impl: ServerAsync
    var _stop_flag: Atomic[DType.int64]

    var _handler: ProtocolHandler
    var _callbacks: CallbackSet[
        Guarded[ServerStats, STATS_ENABLED],
        Guarded[BatchHeatMap, BATCH_HEAT_MAP_ENABLED],
        Callbacks,
    ]

    var _tstate: Int64

    @staticmethod
    fn create[
        HTTP_SERVER_ENABLED: Bool = False
    ](
        address: String, owned session: InferenceSession
    ) raises -> InferenceServer[HTTP_SERVER_ENABLED, NoopServerCallbacks]:
        return InferenceServer[HTTP_SERVER_ENABLED](
            address, NoopServerCallbacks(), session^
        )

    fn __init__(
        inout self,
        address: String,
        owned callbacks: Callbacks,
        owned session: InferenceSession,
        num_listeners: Int = 4,
    ) raises:
        """Constructs a gRPC inference server.

        Args:
            address: Address to serve on.
            callbacks: Extra lifecycle callbacks.
            session: Current inference context.
            num_listeners: Number of listener tasks.
        """
        self._lib = handle_from_config("serving", ".serve_lib")
        self._session = session^
        self._num_listeners = num_listeners
        self._impl = ServerAsync(address, self._lib, self._session)
        self._stop_flag = Atomic[DType.int64](0)

        self._handler = ProtocolHandler(self._lib, self._impl._impl._ptr)
        self._callbacks = CallbackSet(
            Guarded[ServerStats, STATS_ENABLED](ServerStats()),
            Guarded[BatchHeatMap, BATCH_HEAT_MAP_ENABLED](BatchHeatMap()),
            callbacks^,
        )
        self._callbacks.on_server_start()

        self._tstate = 0

    fn __del__(owned self):
        self._callbacks.on_server_stop()

    fn serve[
        service_type: InferenceService
    ](inout self, inout service: service_type) -> None:
        @always_inline
        @parameter
        async fn handle(
            request: InferenceRequestImpl,
            inout response: Variant[InferenceResponseImpl, Error],
        ) capturing -> None:
            await service.async_infer(request, response)

        self.serve[handle]()

    fn serve[handle_fn: Self.handle_fn_type](inout self) -> None:
        var rt = Runtime()
        _ = rt.run(self.async_serve[handle_fn]())

    async fn async_serve[handle_fn: Self.handle_fn_type](inout self) -> None:
        @always_inline
        @parameter
        async fn process(batch: InferenceBatch) capturing -> None:
            var requests = batch.requests()
            var responses = batch.responses()
            var size = len(batch)
            for i in range(size):
                var req = requests[i]
                var resp = responses[i]

                # TODO: Record start closer to actual request receipt.
                var start = now()
                self._callbacks.on_request_receive()
                var respOr = Variant[InferenceResponseImpl, Error](resp^)
                await handle_fn(req, respOr)

                if respOr.isa[Error]():
                    var err = str(respOr.unsafe_take[Error]())
                    self._callbacks.on_request_fail(err)

                    @parameter
                    if EnableProtocol:
                        self._handler.handle_fail(req, err)
                    self._impl.push_failed(batch, i, err)
                else:
                    var resp = respOr.unsafe_take[InferenceResponseImpl]()
                    self._callbacks.on_request_ok(start)

                    @parameter
                    if EnableProtocol:
                        self._handler.handle_ok(req, resp)
                    self._impl.push_complete(batch, i)
                    _ = resp^

        @always_inline
        @parameter
        async fn listen() capturing -> None:
            while not self._stop_flag.load():
                # TODO: Construct merged request (per model) for batching.
                var batch = InferenceBatch(self._lib, self._session)
                await self._impl.pop_ready(batch)
                var start = now()
                self._callbacks.on_batch_receive(len(batch))
                await process(batch)
                self._callbacks.on_batch_complete(start, len(batch))
                _ = batch^

        self._impl.run()

        var rt = Runtime()
        var tg = TaskGroup[__lifetime_of()](rt)
        for _ in range(self._num_listeners):
            _ = tg.create_task(listen())

        var server_thread = PythonObject()

        @parameter
        if HTTP_SERVER_ENABLED:
            server_thread = run_http_rt(self._impl._impl)
            var cpython = _get_global_python_itf().cpython()
            self._tstate = cpython.PyEval_SaveThread()

        await tg
        _ = server_thread^
        self._impl.stop()

    fn signal_stop(inout self):
        """Signals the server to be stop serving requests."""
        _ = self._stop_flag.fetch_add(1)
        self._impl.signal_stop()


struct MuxInferenceService(InferenceService):
    """Inference service that multiplexes across a list of models."""

    alias versions_dict_type = Dict[String, UnsafePointer[Model]]
    alias model_dict_type = Dict[String, UnsafePointer[Self.versions_dict_type]]

    var _models: List[Model]
    var _version_dicts: List[Self.versions_dict_type]
    var _model_dict: Self.model_dict_type

    var _session: InferenceSession

    fn __init__(
        inout self,
        models: List[FileModel],
        owned session: InferenceSession,
    ) raises:
        self._session = session^
        self._model_dict = Self.model_dict_type()
        self._version_dicts = List[Self.versions_dict_type](
            capacity=len(models)
        )
        self._models = List[Model](capacity=len(models))
        for model in models:
            var name = model[].name
            if name not in self._model_dict:
                self._version_dicts.append(Self.versions_dict_type())
                var back = self._version_dicts.__get_ref(-1)
                self._model_dict[name] = UnsafePointer.address_of(back[])

            var version = model[].version
            var versioned = self._model_dict[name]
            if version not in versioned[]:
                self._models.append(
                    self._session.load(
                        model[].path, input_specs=model[].input_specs
                    )
                )
                var back = self._models.__get_ref(-1)
                versioned[][version] = UnsafePointer.address_of(back[])
            else:
                raise Error(
                    "Cannot add duplicate version: "
                    + version
                    + " for model: "
                    + name
                )

    fn __del__(owned self):
        _ = self._model_dict^
        _ = self._version_dicts^
        _ = self._models^
        _ = self._session^

    fn init(inout self, inout server: InferenceServer) raises:
        server._impl.init(self._models)

    fn infer[
        req_type: InferenceRequest, resp_type: InferenceResponse
    ](inout self, request: req_type, inout response: resp_type) raises -> None:
        var respOr = Variant[resp_type, Error](response^)
        var rt = Runtime()
        _ = rt.run(self.async_infer(request, respOr))
        if respOr.isa[Error]():
            raise respOr.unsafe_take[Error]()
        else:
            response = respOr.unsafe_take[resp_type]()

    async fn async_infer[
        req_type: InferenceRequest, resp_type: InferenceResponse
    ](
        inout self, request: req_type, inout response: Variant[resp_type, Error]
    ) -> None:
        try:
            var name = request.get_model_name()
            var version = request.get_model_version()
            # TODO: Choose latest if version is not set.
            var model = self._model_dict[name][][version]

            var inputs = request.get_input_tensors()
            var outputs = model[].execute(inputs^)
            response[resp_type].set_output_tensors(
                request.get_outputs(), outputs^
            )
        except e:
            response.set[Error](e)
