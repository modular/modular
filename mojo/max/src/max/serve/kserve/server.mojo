# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Provides C bindings to KServe data structures."""

from os import Atomic
from runtime.asyncrt import run, TaskGroup
from sys.ffi import DLHandle
from time import perf_counter_ns
from utils import StringRef

from max.engine import InferenceSession, Model
from max.engine._compilation import CCompiledModel
from max._utils import handle_from_config

from ..util.callbacks import (
    ServerCallbacks,
    NoopServerCallbacks,
    Guarded,
    CallbackSet,
)
from ..util.stats import ServerStats
from ..util.config import (
    STATS_ENABLED,
    BATCH_HEAT_MAP_ENABLED,
)
from ..util.debug import BatchHeatMap

from .types import InferenceRequest, InferenceResponse
from .service import InferenceService
from ._server import CGRPCServer


struct GRPCServer[
    Callbacks: ServerCallbacks = NoopServerCallbacks,
]:
    """Inference server implementing the KServe protocol over gRPC."""

    alias handle_fn_type = fn (
        InferenceRequest, inout InferenceResponse
    ) capturing raises -> None

    var _lib: DLHandle
    var _impl: CGRPCServer
    var _session: InferenceSession
    var _num_listeners: Int
    var _stop_flag: Atomic[DType.int64]

    var _callbacks: CallbackSet[
        Guarded[ServerStats, STATS_ENABLED],
        Guarded[BatchHeatMap, BATCH_HEAT_MAP_ENABLED],
        Callbacks,
    ]

    @staticmethod
    fn create(
        address: String, owned session: InferenceSession
    ) raises -> GRPCServer:
        return GRPCServer(address, session^, NoopServerCallbacks())

    fn __init__(
        inout self,
        address: String,
        owned session: InferenceSession,
        owned callbacks: Callbacks,
        num_listeners: Int = 4,
    ) raises:
        """Constructs a gRPC inference server.

        Args:
            address: Address to serve on.
            session: Current inference context.
            callbacks: Extra lifecycle callbacks.
            num_listeners: Number of listener tasks.
        """
        self._lib = handle_from_config("serving", ".serve_lib")
        self._session = session^
        self._num_listeners = num_listeners
        self._impl = CGRPCServer(self._lib, StringRef(ptr=address.unsafe_ptr()))
        self._stop_flag = Atomic[DType.int64](0)
        self._callbacks = CallbackSet(
            Guarded[ServerStats, STATS_ENABLED](ServerStats()),
            Guarded[BatchHeatMap, BATCH_HEAT_MAP_ENABLED](BatchHeatMap()),
            callbacks^,
        )
        self._callbacks.on_server_start()

    fn __del__(owned self):
        self._callbacks.on_server_stop()

    fn serve[
        service_type: InferenceService
    ](inout self, inout service: service_type) raises -> None:
        @parameter
        fn _serve_start_NoOp() -> NoneType:
            return None

        self.serve[service_type, _serve_start_NoOp](service)

    fn serve[
        service_type: InferenceService, start_fn: fn () capturing -> NoneType
    ](inout self, inout service: service_type) raises -> None:
        @always_inline
        @parameter
        fn add_models(models: List[Model]) raises -> None:
            var ptrs = List[CCompiledModel](capacity=len(models))
            for model in models:
                ptrs.append(model[]._compiled_model.ptr)
            self._impl.init(ptrs)

        @always_inline
        @parameter
        fn handle(
            request: InferenceRequest,
            inout response: InferenceResponse,
        ) raises -> None:
            service.infer(request, response)

        service.init[add_models]()
        start_fn()
        self.serve[handle]()

    fn serve[handle_fn: Self.handle_fn_type](inout self) -> None:
        run(self._serve[handle_fn]())

    async fn _serve[handle_fn: Self.handle_fn_type](inout self) -> None:
        @always_inline
        @parameter
        async fn process(inout batch: InferenceBatch) -> None:
            for i in range(len(batch)):
                var req = batch.request_at(i)
                var resp = batch.response_at(i)

                # TODO: Record start closer to actual request receipt.
                var start = perf_counter_ns()
                self._callbacks.on_request_receive()
                try:
                    handle_fn(req, resp)
                    self._callbacks.on_request_ok(start)
                    batch._impl.push_complete(i)
                    _ = resp^
                except err:
                    self._callbacks.on_request_fail(str(err))
                    batch._impl.push_failed(i, str(err))

        @always_inline
        @parameter
        async fn listen() -> None:
            while not self._stop_flag.load():
                # TODO: Construct merged request (per model) for batching.
                var batch = InferenceBatch(self._lib, self._session)
                await self._impl.pop_ready(batch._impl)
                var start = perf_counter_ns()
                self._callbacks.on_batch_receive(len(batch))
                await process(batch)
                self._callbacks.on_batch_complete(start, len(batch))
                _ = batch^

        self._impl.run()

        var tg = TaskGroup()
        for _ in range(self._num_listeners):
            tg.create_task(listen())

        await tg
        self._impl.stop()

    fn signal_stop(inout self):
        """Signals the server to be stop serving requests."""
        _ = self._stop_flag.fetch_add(1)
        self._impl.signal_stop()
