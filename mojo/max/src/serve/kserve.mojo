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
    TaskGroupTask,
    TaskGroupTaskList,
)
from time import now

from max.engine import InferenceSession, Model
from max.engine._utils import handle_from_config

from .callbacks import (
    CallbacksPair,
    ServerCallbacks,
    NoopServerCallbacks,
    make_callbacks_pair,
)
from .service import (
    InferenceRequest,
    InferenceResponse,
    InferenceService,
    FileModel,
)
from .stats import ServerStats, ServerStatsOptions
from ._kserve_impl import ModelInferRequest, ModelInferResponse
from ._serve_rt import Batch, MuttServerAsync, TensorView


# TODO(yihualou): Make this multi-model
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

    alias handle_fn_type = async fn (
        ModelInferRequest, inout ModelInferResponse
    ) capturing -> None

    var _lib: DLHandle
    var _session: InferenceSession
    var _num_listeners: Int
    var _impl: MuttServerAsync

    var _callbacks: CallbacksPair[ServerStats, NoopServerCallbacks]

    fn __init__(
        inout self,
        address: String,
        owned session: InferenceSession,
        num_listeners: Int = 4,
    ) raises:
        """Constructs a gRPC inference server.

        Args:
            address: Address to serve on.
            session: Current inference context.
            num_listeners: Number of listener tasks.
        """
        self._lib = handle_from_config("serving", "max.serve_lib")
        self._session = session^
        self._num_listeners = num_listeners
        self._impl = MuttServerAsync(address, self._lib, self._session)

        self._callbacks = make_callbacks_pair(
            ServerStats(), NoopServerCallbacks()
        )
        self._callbacks.on_server_start()

    fn __del__(owned self):
        self._callbacks.on_server_stop()

    fn serve[
        service_type: InferenceService
    ](inout self, service: service_type) -> None:
        @always_inline
        @parameter
        async fn handle(
            request: ModelInferRequest, inout response: ModelInferResponse
        ) capturing -> None:
            service.infer(request, response)

        self.serve[handle]()

    fn serve[handle_fn: Self.handle_fn_type](inout self) -> None:
        var rt = Runtime()
        rt.run(self.async_serve[handle_fn]())

    async fn async_serve[handle_fn: Self.handle_fn_type](inout self) -> None:
        @always_inline
        @parameter
        async fn process(batch: Batch) capturing -> None:
            var requests = batch.requests()
            var responses = batch.responses()
            var size = len(batch)
            for i in range(size):
                var req = requests[i]
                var resp = responses[i]
                var start = now()
                self._callbacks.on_request_receive(req)
                await handle_fn(req, resp)
                self._impl.push_complete(batch, i)
                self._callbacks.on_request_ok(start, req)
                _ = resp^

        @always_inline
        @parameter
        async fn listen() capturing -> None:
            while True:
                var batch = Batch(self._lib, self._session)
                await self._impl.pop_ready(batch)
                var start = now()
                self._callbacks.on_batch_receive(batch)
                await process(batch)
                self._callbacks.on_batch_complete(start, batch)
                _ = batch^

        self._impl.run()

        var rt = Runtime()
        var tasks = TaskGroupTaskList[NoneType](self._num_listeners)
        var tg = TaskGroup(rt)
        for i in range(self._num_listeners):
            var task = tg.create_task[NoneType](listen())
            tasks.add(task^)

        await tg
        _ = tasks^
