# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import logging
from typing import Any, Optional

from max.serve.scheduler.async_queue import AsyncCallConsumer  # type: ignore
from max.serve.telemetry.common import (
    TelemetryConfig,
    metrics_resource,
    otelBaseUrl,
)
from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
    OTLPMetricExporter,
)
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.metrics import get_meter_provider, set_meter_provider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    MetricReader,
    PeriodicExportingMetricReader,
)

logger = logging.getLogger("max.serve")


class _NoOpMetric:
    """No op metric to support tests etc."""

    def add(self, *args, **kwargs) -> None:
        pass

    def record(self, *args, **kwargs) -> None:
        pass


def sync(f, *args, **kw):
    return f(*args, **kw)


def configure_metrics():
    default_config = TelemetryConfig.from_env()
    metrics_egress_enabled = default_config.metrics_egress_enabled

    meterProviders: list[MetricReader] = [PrometheusMetricReader(True)]
    if metrics_egress_enabled:
        meterProviders.append(
            PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=otelBaseUrl + "/v1/metrics")
            )
        )
    set_meter_provider(MeterProvider(meterProviders, metrics_resource))


class _Metrics:
    """Centralizes metrics to encapsulate the OTEL dependency and avoid breaking schema changes"""

    def __init__(self):
        self._req_count: Any = _NoOpMetric()
        self._req_time: Any = _NoOpMetric()
        self._input_time: Any = _NoOpMetric()
        self._output_time: Any = _NoOpMetric()
        self._ttft: Any = _NoOpMetric()
        self._input_tokens: Any = _NoOpMetric()
        self._output_tokens: Any = _NoOpMetric()
        self._reqs_queued: Any = _NoOpMetric()
        self._reqs_running: Any = _NoOpMetric()
        self._model_load_time: Any = _NoOpMetric()
        self._itl: Any = _NoOpMetric()
        self._pipeline_load: Any = _NoOpMetric()
        self._configured = False

        # by default, configure _Metrics to public sync
        self.call_async = False
        self.started = False
        self._call = sync
        self.aq: Optional[AsyncCallConsumer] = None

    async def configure(self, async_metrics: Optional[bool] = None):
        default_config = TelemetryConfig.from_env()
        self.call_async = (
            async_metrics
            if async_metrics is not None
            else default_config.async_metrics
        )

        if not self._configured:
            self._configured = True
            _meter = get_meter_provider().get_meter("modular")
            self._req_count = _meter.create_counter(
                "maxserve.request_count", description="Http request count"
            )
            self._req_time = _meter.create_histogram(
                "maxserve.request_time", "ms", "Time spent in requests"
            )
            self._input_time = _meter.create_histogram(
                "maxserve.input_processing_time", "ms", "Input processing time"
            )
            self._output_time = _meter.create_histogram(
                "maxserve.output_processing_time",
                "ms",
                "Output processing time",
            )
            self._ttft = _meter.create_histogram(
                "maxserve.time_to_first_token", "ms", "Time to first token"
            )
            self._input_tokens = _meter.create_counter(
                "maxserve.num_input_tokens", description="Count of input tokens"
            )
            self._output_tokens = _meter.create_counter(
                "maxserve.num_output_tokens",
                description="Count of generated tokens",
            )
            self._reqs_queued = _meter.create_up_down_counter(
                "maxserve.num_requests_queued",
                description="Count of requests waiting to be processed",
            )
            self._reqs_running = _meter.create_up_down_counter(
                "maxserve.num_requests_running",
                description="Count of requests currently being processed",
            )
            self._model_load_time = _meter.create_histogram(
                "maxserve.model_load_time",
                unit="ms",
                description="Time to load a model",
            )
            self._itl = _meter.create_histogram(
                "maxserve.itl", unit="ms", description="Inter token latency"
            )
            self._pipeline_load = _meter.create_counter(
                "maxserve.pipeline_load",
                description="Count of pipelines loaded for each model",
            )

        if self.call_async and not self.started:
            self.aq = AsyncCallConsumer()
            self.started = False
            try:
                self.aq.start()
            except Exception as e:
                logger.exception("failed to start consumer")
            self.started = True
            self._call = self.aq.call

    async def shutdown(self):
        if self.call_async and self.started:
            assert self.aq is not None
            self._call = sync
            await self.aq.shutdown()
            self.started = False
            self.aq = None

    def request_count(self, responseCode: int, urlPath: str) -> None:
        self._call(
            self._req_count.add, 1, {"code": responseCode, "path": urlPath}
        )

    def request_time(self, value: float, urlPath: str) -> None:
        self._call(self._req_time.record, value, {"path": urlPath})

    def input_time(self, value: float) -> None:
        self._call(self._input_time.record, value)

    def output_time(self, value: float) -> None:
        self._call(self._output_time.record, value)

    def ttft(self, value: float) -> None:
        self._call(self._ttft.record, value)

    def input_tokens(self, value: int) -> None:
        self._call(self._input_tokens.add, value)

    def output_tokens(self, value: int) -> None:
        self._call(self._output_tokens.add, value)

    def reqs_queued(self, value: int) -> None:
        self._call(self._reqs_queued.add, value)

    def reqs_running(self, value: int) -> None:
        self._call(self._reqs_running.add, value)

    def model_load_time(self, ms: float) -> None:
        self._call(self._model_load_time.record, ms)

    def itl(self, ms: float) -> None:
        self._call(self._itl.record, ms)

    def pipeline_load(self, name: str) -> None:
        self._call(self._pipeline_load.add, 1, {"model": name})


METRICS = _Metrics()
