# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Union, get_args

from max.serve.config import Settings
from max.serve.scheduler.async_queue import AsyncCallConsumer  # type: ignore
from max.serve.telemetry.common import TelemetryConfig
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
# Note: this can not use max.loggers.get_logger due to being in telemetry.
# max.loggers.get_logger configures telemetry.
logger = logging.getLogger(__name__)
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
        "maxserve.request_time", "ms", "Time spent in requests"
    ),  # type: ignore
    "maxserve.input_processing_time": _meter.create_histogram(
        "maxserve.input_processing_time", "ms", "Input processing time"
    ),  # type: ignore
    "maxserve.output_processing_time": _meter.create_histogram(
        "maxserve.output_processing_time", "ms", "Output processing time"
    ),  # type: ignore
    "maxserve.time_to_first_token": _meter.create_histogram(
        "maxserve.time_to_first_token", "ms", "Time to first token"
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
}


class RecordMode(enum.Enum):
    NOOP = enum.auto()
    SYNC = enum.auto()
    ASYNCIO = enum.auto()
    PROCESS = enum.auto()


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


class _Metrics:
    """Centralizes metrics to encapsulate the OTEL dependency and avoid breaking schema changes"""

    def __init__(self):
        self.started = False
        self.mode = RecordMode.NOOP
        self.aq: Optional[AsyncCallConsumer] = None

    async def configure(self, server_settings: Settings):
        default_config = TelemetryConfig.from_config(server_settings)

        new_mode = RecordMode.SYNC
        if default_config.async_metrics:
            new_mode = RecordMode.ASYNCIO

        if new_mode == RecordMode.ASYNCIO and not self.started:
            self.aq = AsyncCallConsumer()
            try:
                self.aq.start()
            except Exception as e:
                logger.exception("failed to start consumer")
            self.started = True

        if new_mode == RecordMode.SYNC:
            pass

        self.mode = new_mode

    def _call(self, m: MaxMeasurement):
        if self.mode == RecordMode.NOOP:
            return
        elif self.mode == RecordMode.SYNC:
            m.commit()
        elif self.mode == RecordMode.ASYNCIO:
            if self.aq is not None:
                self.aq.call(m.commit)
        elif self.mode == RecordMode.PROCESS:
            raise NotImplementedError()
        else:
            raise Exception("Unrecognized mode")

    async def shutdown(self):
        if self.mode == RecordMode.ASYNCIO and self.started:
            assert self.aq is not None
            self.mode = RecordMode.SYNC
            await self.aq.shutdown()
            self.started = False
            self.aq = None

    def request_count(self, responseCode: int, urlPath: str) -> None:
        self._call(
            MaxMeasurement(
                "maxserve.request_count",
                1,
                {"code": f"{responseCode:d}", "path": urlPath},
            )
        )

    def request_time(self, value: float, urlPath: str) -> None:
        self._call(
            MaxMeasurement("maxserve.request_time", value, {"path": urlPath})
        )

    def input_time(self, value: float) -> None:
        self._call(MaxMeasurement("maxserve.input_processing_time", value))

    def output_time(self, value: float) -> None:
        self._call(MaxMeasurement("maxserve.output_processing_time", value))

    def ttft(self, value: float) -> None:
        self._call(MaxMeasurement("maxserve.time_to_first_token", value))

    def input_tokens(self, value: int) -> None:
        self._call(MaxMeasurement("maxserve.num_input_tokens", value))

    def output_tokens(self, value: int) -> None:
        self._call(MaxMeasurement("maxserve.num_output_tokens", value))

    def reqs_queued(self, value: int) -> None:
        self._call(MaxMeasurement("maxserve.num_requests_queued", value))

    def reqs_running(self, value: int) -> None:
        self._call(MaxMeasurement("maxserve.num_requests_running", value))

    def model_load_time(self, ms: float) -> None:
        self._call(MaxMeasurement("maxserve.model_load_time", ms))

    def itl(self, ms: float) -> None:
        self._call(MaxMeasurement("maxserve.itl", ms))

    def pipeline_load(self, name: str) -> None:
        self._call(
            MaxMeasurement(
                "maxserve.pipeline_load",
                1,
                attributes={"model": name},
            )
        )


METRICS = _Metrics()
