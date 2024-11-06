# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from typing import Optional


from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
    OTLPMetricExporter,
)
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.metrics import get_meter_provider, set_meter_provider


class _Metrics:
    """Centralizes metrics to encapsulate the OTEL dependency and avoid breaking schema changes.
    """

    def __init__(self):
        self._req_count = None
        self._req_time = None
        self._input_time = None
        self._output_time = None
        self._ttft = None
        self._input_tokens = None
        self._output_tokens = None

    def configure(self, otlp_level: Optional[int] = None):
        meterProviders = [PrometheusMetricReader("metrics")]  # type: ignore
        if otlp_level is not None:
            meterProviders.append(
                PeriodicExportingMetricReader(  # type: ignore
                    OTLPMetricExporter(
                        endpoint="https://telemetry.modular.com:443/v1/metrics"
                    )
                )
            )
        set_meter_provider(MeterProvider(meterProviders))
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


METRICS = _Metrics()
