# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import os
from typing import Optional

from max.serve.telemetry.common import otelBaseUrl

from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
    OTLPMetricExporter,
)
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.metrics import get_meter_provider, set_meter_provider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource


class _NoOpMetric:
    """No op metric to support tests etc."""

    def add(self, *args, **kwargs):
        pass

    def record(self, *args, **kwargs):
        pass


class _Metrics:
    """Centralizes metrics to encapsulate the OTEL dependency and avoid breaking schema changes.
    """

    def __init__(self):
        self._req_count = _NoOpMetric()
        self._req_time = _NoOpMetric()
        self._input_time = _NoOpMetric()
        self._output_time = _NoOpMetric()
        self._ttft = _NoOpMetric()
        self._input_tokens = _NoOpMetric()
        self._output_tokens = _NoOpMetric()
        self._reqs_queued = _NoOpMetric()
        self._reqs_running = _NoOpMetric()

    def configure(self, otlp_level: Optional[int] = None):
        meterProviders = [PrometheusMetricReader("metrics")]  # type: ignore
        if otlp_level is not None:
            meterProviders.append(
                PeriodicExportingMetricReader(  # type: ignore
                    OTLPMetricExporter(endpoint=otelBaseUrl + "/v1/metrics")
                )
            )
        resource = Resource.create(
            {"deployment.id": os.environ.get("MAX_SERVE_DEPLOYMENT_ID", "")}
        )
        set_meter_provider(MeterProvider(meterProviders, resource))
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

    def requestCount(self, responseCode: int, urlPath: str):
        self._req_count.add(1, {"code": responseCode, "path": urlPath})  # type: ignore

    def requestTime(self, value: float, urlPath: str):
        self._req_time.record(value, {"path": urlPath})  # type: ignore

    def inputTime(self, value: float):
        self._input_time.record(value)  # type: ignore

    def outputTime(self, value: float):
        self._output_time.record(value)  # type: ignore

    def ttft(self, value: float):
        self._ttft.record(value)  # type: ignore

    def inputTokens(self, value: int):
        self._input_tokens.add(value)  # type: ignore

    def outputTokens(self, value: int):
        self._output_tokens.add(value)  # type: ignore

    def reqsQueued(self, value: int):
        self._reqs_queued.add(value)  # type: ignore

    def reqsRunning(self, value: int):
        self._reqs_running.add(value)  # type: ignore


METRICS = _Metrics()
