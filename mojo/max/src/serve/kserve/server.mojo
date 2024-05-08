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
from tensor import TensorSpec
from time import now
from utils.variant import Variant

from max.engine import InferenceSession, InputSpec, Model
from max.engine._model_impl import CModel
from max.engine._utils import handle_from_config
from max.serve.service import (
    InferenceRequest,
    InferenceResponse,
    InferenceService,
    FileModel,
)

from .callbacks import (
    CallbacksPair,
    Guarded,
    NoopServerCallbacks,
    ServerCallbacks,
    make_callbacks_triple,
)
from .config import BATCH_HEAT_MAP_ENABLED, STATS_ENABLED

from ._kserve_impl import ModelInferRequest, ModelInferResponse
from ._serve_rt import Batch, MuttServerAsync, TensorView
from .stats import ServerStats, ServerStatsOptions
from .debug import BatchHeatMap


struct GRPCInferenceServer[Callbacks: ServerCallbacks = NoopServerCallbacks]:
    """Inference server implementing the KServe protocol over gRPC."""

    alias handle_fn_type = async fn (
        ModelInferRequest, inout Variant[ModelInferResponse, Error]
    ) capturing -> None

    var _lib: DLHandle
    var _session: InferenceSession
    var _num_listeners: Int
    var _impl: MuttServerAsync

    var _callbacks: CallbacksPair[
        CallbacksPair[
            Guarded[ServerStats, STATS_ENABLED],
            Guarded[BatchHeatMap, BATCH_HEAT_MAP_ENABLED],
        ],
        Callbacks,
    ]

    @staticmethod
    fn create(
        address: String, owned session: InferenceSession
    ) raises -> GRPCInferenceServer[NoopServerCallbacks]:
        return GRPCInferenceServer(address, NoopServerCallbacks(), session^)

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
        self._impl = MuttServerAsync(address, self._lib, self._session)

        self._callbacks = make_callbacks_triple(
            Guarded[ServerStats, STATS_ENABLED](ServerStats()),
            Guarded[BatchHeatMap, BATCH_HEAT_MAP_ENABLED](BatchHeatMap()),
            callbacks^,
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
            request: ModelInferRequest,
            inout response: Variant[ModelInferResponse, Error],
        ) capturing -> None:
            await service.async_infer(request, response)

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

                # TODO: Record start closer to actual request receipt.
                var start = now()
                self._callbacks.on_request_receive(req)
                var respOr = Variant[ModelInferResponse, Error](resp^)
                await handle_fn(req, respOr)
                if respOr.isa[Error]():
                    self._impl.push_failed(batch, i, str(respOr.take[Error]()))
                    self._callbacks.on_request_fail(req)
                else:
                    self._impl.push_complete(batch, i)
                    self._callbacks.on_request_ok(start, req)
                    _ = respOr.take[ModelInferResponse]()

        @always_inline
        @parameter
        async fn listen() capturing -> None:
            while True:
                var batch = Batch(self._lib, self._session)
                # TODO: Construct merged request (per model) for batching.
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


struct MuxInferenceService(InferenceService):
    """Inference service that multiplexes across a list of models."""

    alias versions_dict_type = Dict[String, Pointer[Model]]
    alias model_dict_type = Dict[String, Pointer[Self.versions_dict_type]]

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
                self._model_dict[name] = LegacyPointer.address_of(back.value)

            var version = model[].version
            var versioned = self._model_dict[name]
            if version not in versioned[]:
                self._models.append(
                    self._session.load(
                        model[].path, input_specs=model[].input_specs
                    )
                )
                var back = self._models.__get_ref(-1)
                versioned[][version] = LegacyPointer.address_of(back.value)
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

    fn init(self, server: GRPCInferenceServer) raises:
        server._impl.init(self._models)

    fn infer[
        req_type: InferenceRequest, resp_type: InferenceResponse
    ](self, request: req_type, inout response: resp_type) raises -> None:
        var respOr = Variant[resp_type, Error](response^)
        var rt = Runtime()
        rt.run(self.async_infer(request, respOr))
        if respOr.isa[Error]():
            raise respOr.take[Error]()
        else:
            response = respOr.take[resp_type]()

    async fn async_infer[
        req_type: InferenceRequest, resp_type: InferenceResponse
    ](
        self, request: req_type, inout response: Variant[resp_type, Error]
    ) -> None:
        try:
            var name = request.get_model_name()
            var version = request.get_model_version()
            # TODO: Choose latest if version is not set.
            var model = self._model_dict[name][][version]

            var inputs = request.get_input_tensors()
            var outputs = model[].execute(inputs^)
            response.get[resp_type]()[].set_output_tensors(
                request.get_requested_outputs(), outputs^
            )
        except e:
            response.set[Error](e)
