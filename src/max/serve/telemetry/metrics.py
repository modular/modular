# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import abc
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Union, get_args

from max.serve.config import MetricLevel, Settings
from opentelemetry import context
from opentelemetry.metrics import get_meter_provider
from opentelemetry.metrics._internal import instrument as api_instrument
from opentelemetry.sdk.metrics._internal import instrument as sdk_instrument
from opentelemetry.sdk.metrics._internal import measurement

"""!! Jank alert !!

We want to use OTEL for propagating telemetry. It is the best vendor-agnositc
metrics system, but that doens't mean that it is _good_.  OTEL is _slow_. If we
use it directly, it significally degrades the perf of Max Serve. Consequently,
we have all this machinery to observe some metric (MaxMeasurement) and record
the observation async.

OTEL actively obscures its machinery, uses bunch of proxy classes, has an baroque inheritance tree, and is generally awful.
To record an observation at a specific point in time you do the following:
`meter.create_{foo}._real_instrument._measurement_consumer(Measurement(value, timestamp, instument, ...))`

Here is how you work with metrics (Instruments) observations (Measurements) and recording them (Consumers):
Lets unpack:
1. meter.create_{foo} gives you a proxy instrument with an obscured type eg _internal.instrument._ProxyCounter.
2. `._real_instrument` The proxy can't do anything, you need to grab the _real_ instrument to record.
3. `._measurement_consuemr` The _real_ instrument doesn't expose a way to set the time of the observation, so you have to directly talk to the consumer.
4. `Measurement(...)` now we can create a measurement with a timestamp & pass it down.
"""
logger = logging.getLogger("max.serve")
_meter = get_meter_provider().get_meter("modular")


NumberType = Union[float, int]
OtelAttributes = Optional[dict[str, str]]

# API_PROXIES the "types" of measurments we make from a meter
# SDK instruments are the "types" that actually do recording
API_PROXIES = Union[
    api_instrument._ProxyCounter,
    api_instrument._ProxyHistogram,
    api_instrument._ProxyUpDownCounter,
]
SDK_INSTRUMENTS = Union[
    sdk_instrument._Counter,
    sdk_instrument._Histogram,
    sdk_instrument._UpDownCounter,
]
SupportedInstruments = Union[API_PROXIES, SDK_INSTRUMENTS]


# Sorry for the type ignores, OTEL goes out of its way to obscure its types.
# We need to use the fact that these are actually _Proxy{Type} objects  rather
# than {Type} objects
SERVE_METRICS: dict[str, SupportedInstruments] = {
    "maxserve.request_count": _meter.create_counter(
        "maxserve.request_count", description="Http request count"
    ),  # type: ignore
    "maxserve.request_time": _meter.create_histogram(
        "maxserve.request_time", unit="ms", description="Time spent in requests"
    ),  # type: ignore
    "maxserve.input_processing_time": _meter.create_histogram(
        "maxserve.input_processing_time",
        unit="ms",
        description="Input processing time",
    ),  # type: ignore
    "maxserve.output_processing_time": _meter.create_histogram(
        "maxserve.output_processing_time",
        unit="ms",
        description="Output processing time",
    ),  # type: ignore
    "maxserve.time_to_first_token": _meter.create_histogram(
        "maxserve.time_to_first_token",
        unit="ms",
        description="Time to first token",
    ),  # type: ignore
    "maxserve.num_input_tokens": _meter.create_counter(
        "maxserve.num_input_tokens", description="Count of input tokens"
    ),  # type: ignore
    "maxserve.num_output_tokens": _meter.create_counter(
        "maxserve.num_output_tokens", description="Count of generated tokens"
    ),  # type: ignore
    "maxserve.num_requests_queued": _meter.create_up_down_counter(
        "maxserve.num_requests_queued",
        description="Count of requests waiting to be processed",
    ),  # type: ignore
    "maxserve.num_requests_running": _meter.create_up_down_counter(
        "maxserve.num_requests_running",
        description="Count of requests currently being processed",
    ),  # type: ignore
    "maxserve.model_load_time": _meter.create_histogram(
        "maxserve.model_load_time",
        unit="ms",
        description="Time to load a model",
    ),  # type: ignore
    "maxserve.itl": _meter.create_histogram(
        "maxserve.itl", unit="ms", description="inter token latency"
    ),  # type: ignore
    "maxserve.pipeline_load": _meter.create_counter(
        "maxserve.pipeline_load",
        description="Count of pipelines loaded for each model",
    ),  # type: ignore
    "maxserve.batch_size": _meter.create_histogram(
        "maxserve.batch_size",
        description="Distribution of batch sizes",
    ),  # type: ignore
}


class UnknownMetric(Exception):
    pass


@dataclass
class MaxMeasurement:
    """Shim around the recording of a metric observation

    Simplifies decoupling the observation of a metric from its recording.
    """

    instrument_name: str
    value: NumberType
    attributes: Optional[OtelAttributes] = None
    time_unix_nano: int = field(default_factory=time.time_ns)

    def commit(self):
        # find the instrument
        try:
            instrument = SERVE_METRICS[self.instrument_name]
        except KeyError as e:
            raise UnknownMetric(self.instrument_name) from e

        # Sometimes the instrument is a proxy.  Unrap it.
        if isinstance(instrument, get_args(API_PROXIES)):
            instrument = instrument._real_instrument
            # bail if there is no underlying instrument
            if instrument is None:
                return

        # instrument should be one of the supported sdk types now
        if not isinstance(instrument, get_args(SDK_INSTRUMENTS)):
            return

        # convert to an otel measurement
        m = measurement.Measurement(
            self.value,
            self.time_unix_nano,
            instrument,
            context.get_current(),
            self.attributes,
        )
        # record the measurement
        consumer = instrument._measurement_consumer
        consumer.consume_measurement(m)


TelemetryFn = Callable[[MaxMeasurement], None]


class MetricClient(abc.ABC):
    @abc.abstractmethod
    def send_measurement(
        self, metric: MaxMeasurement, level: MetricLevel
    ) -> None:
        pass


class NoopClient(MetricClient):
    def send_measurement(self, m: MaxMeasurement, level: MetricLevel) -> None:
        pass


class SyncClient(MetricClient):
    def __init__(self, settings: Settings):
        self.level = settings.metric_level

    def send_measurement(self, m: MaxMeasurement, level: MetricLevel) -> None:
        if level > self.level:
            return
        m.commit()


class _AsyncMetrics:
    """Centralizes metrics to encapsulate the OTEL dependency and avoid breaking schema changes

    Produce metric measurements to be consumed elsewhere
    """

    def __init__(self):
        self.client: MetricClient = NoopClient()

    def configure(self, client: MetricClient) -> None:
        self.client = client

    def request_count(self, responseCode: int, urlPath: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.request_count",
                1,
                {"code": f"{responseCode:d}", "path": urlPath},
            ),
            MetricLevel.BASIC,
        )

    def request_time(self, value: float, urlPath: str) -> None:
        self.client.send_measurement(
            MaxMeasurement("maxserve.request_time", value, {"path": urlPath}),
            MetricLevel.BASIC,
        )

    def input_time(self, value: float) -> None:
        self.client.send_measurement(
            MaxMeasurement("maxserve.input_processing_time", value),
            MetricLevel.BASIC,
        )

    def output_time(self, value: float) -> None:
        self.client.send_measurement(
            MaxMeasurement("maxserve.output_processing_time", value),
            MetricLevel.BASIC,
        )

    def ttft(self, value: float) -> None:
        self.client.send_measurement(
            MaxMeasurement("maxserve.time_to_first_token", value),
            MetricLevel.BASIC,
        )

    def input_tokens(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement("maxserve.num_input_tokens", value),
            MetricLevel.BASIC,
        )

    def output_tokens(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement("maxserve.num_output_tokens", value),
            MetricLevel.BASIC,
        )

    def reqs_queued(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement("maxserve.num_requests_queued", value),
            MetricLevel.BASIC,
        )

    def reqs_running(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement("maxserve.num_requests_running", value),
            MetricLevel.BASIC,
        )

    def model_load_time(self, ms: float) -> None:
        self.client.send_measurement(
            MaxMeasurement("maxserve.model_load_time", ms),
            MetricLevel.BASIC,
        )

    def itl(self, ms: float) -> None:
        self.client.send_measurement(
            MaxMeasurement("maxserve.itl", ms), MetricLevel.DETAILED
        )

    def pipeline_load(self, name: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.pipeline_load",
                1,
                attributes={"model": name},
            ),
            MetricLevel.BASIC,
        )

    def batch_size(self, size: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.batch_size",
                size,
            ),
            MetricLevel.DETAILED,
        )


METRICS = _AsyncMetrics()
