# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from typing import Any

from max.serve.telemetry.common import otelBaseUrl, metrics_resource

from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
    OTLPMetricExporter,
)
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.metrics import get_meter_provider, set_meter_provider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader


class _NoOpMetric:
    """No op metric to support tests etc."""

    def add(self, *args, **kwargs) -> None:
        pass

    def record(self, *args, **kwargs) -> None:
        pass


class _Metrics:
    """Centralizes metrics to encapsulate the OTEL dependency and avoid breaking schema changes."""

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

    def configure(self, egress_enabled: bool):
        meterProviders = [PrometheusMetricReader("metrics")]  # type: ignore
        if egress_enabled:
            meterProviders.append(
                PeriodicExportingMetricReader(  # type: ignore
                    OTLPMetricExporter(endpoint=otelBaseUrl + "/v1/metrics")
                )
            )
        set_meter_provider(MeterProvider(meterProviders, metrics_resource))
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
            "maxserve.output_processing_time", "ms", "Output processing time"
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

    def requestCount(self, responseCode: int, urlPath: str) -> None:
        self._req_count.add(1, {"code": responseCode, "path": urlPath})

    def requestTime(self, value: float, urlPath: str) -> None:
        self._req_time.record(value, {"path": urlPath})

    def inputTime(self, value: float) -> None:
        self._input_time.record(value)

    def outputTime(self, value: float) -> None:
        self._output_time.record(value)

    def ttft(self, value: float) -> None:
        self._ttft.record(value)

    def inputTokens(self, value: int) -> None:
        self._input_tokens.add(value)

    def outputTokens(self, value: int) -> None:
        self._output_tokens.add(value)

    def reqsQueued(self, value: int) -> None:
        self._reqs_queued.add(value)

    def reqsRunning(self, value: int) -> None:
        self._reqs_running.add(value)

    def modelLoadTime(self, ms: int) -> None:
        self._model_load_time.record(ms)


METRICS = _Metrics()
