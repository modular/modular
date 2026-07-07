# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import abc
import functools
import logging
import time
from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from typing import get_args

from max.serve.config import MetricLevel, Settings
from opentelemetry import context
from opentelemetry.metrics import get_meter_provider
from opentelemetry.metrics._internal import instrument as api_instrument
from opentelemetry.sdk.metrics._internal import instrument as sdk_instrument
from opentelemetry.sdk.metrics._internal import measurement

"""!! Jank alert !!

We want to use OTEL for propagating telemetry. It is the best vendor-agnostic
metrics system, but that doesn't mean that it is _good_.  OTEL is _slow_. If we
use it directly, it significally degrades the perf of Max Serve. Consequently,
we have all this machinery to observe some metric (MaxMeasurement) and record
the observation async.

OTEL actively obscures its machinery, uses bunch of proxy classes, has an baroque inheritance tree, and is generally awful.
To record an observation at a specific point in time you do the following:
`meter.create_{foo}._real_instrument._measurement_consumer(Measurement(value, timestamp, instrument, ...))`

Here is how you work with metrics (Instruments) observations (Measurements) and recording them (Consumers):
Lets unpack:
1. meter.create_{foo} gives you a proxy instrument with an obscured type eg _internal.instrument._ProxyCounter.
2. `._real_instrument` The proxy can't do anything, you need to grab the _real_ instrument to record.
3. `._measurement_consumer` The _real_ instrument doesn't expose a way to set the time of the observation, so you have to directly talk to the consumer.
4. `Measurement(...)` now we can create a measurement with a timestamp & pass it down.
"""
logger = logging.getLogger("max.serve")
_meter = get_meter_provider().get_meter("modular")


NumberType = float | int
OtelAttributes = dict[str, str] | None

# API_PROXIES the "types" of measurements we make from a meter
# SDK instruments are the "types" that actually do recording
API_PROXIES = (
    api_instrument._ProxyCounter
    | api_instrument._ProxyGauge
    | api_instrument._ProxyHistogram
    | api_instrument._ProxyUpDownCounter
)

SDK_INSTRUMENTS = (
    sdk_instrument._Counter
    | sdk_instrument._Gauge
    | sdk_instrument._Histogram
    | sdk_instrument._UpDownCounter
)

SupportedInstruments = API_PROXIES | SDK_INSTRUMENTS


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
    "maxserve.num_input_characters": _meter.create_counter(
        "maxserve.num_input_characters", description="Count of input characters"
    ),  # type: ignore
    "maxserve.num_output_tokens": _meter.create_counter(
        "maxserve.num_output_tokens", description="Count of generated tokens"
    ),  # type: ignore
    "maxserve.num_requests_queued": _meter.create_gauge(
        "maxserve.num_requests_queued",
        description=(
            "Current depth of the scheduler's CE / prefill queue, "
            "sampled once per scheduler iteration. Mirrors the "
            "'Pending: N reqs' value in scheduler logs."
        ),
    ),  # type: ignore
    "maxserve.num_requests_running": _meter.create_up_down_counter(
        "maxserve.num_requests_running",
        description="Count of requests currently being processed",
    ),  # type: ignore
    "maxserve.num_requests_awaiting_admission": _meter.create_up_down_counter(
        "maxserve.num_requests_awaiting_admission",
        description=(
            "Count of requests received by the API server but not yet handed "
            "off to the model worker (i.e. still in tokenization / pre-submit "
            "on the API side). Incremented on arrival and decremented just "
            "before the request is enqueued to the model worker, so a "
            "persistently high value indicates a backlog stuck in the API "
            "server rather than in the scheduler."
        ),
    ),  # type: ignore
    "maxserve.requests_awaiting_admission": _meter.create_histogram(
        "maxserve.requests_awaiting_admission",
        description=(
            "Distribution of the ingress backlog (requests accepted by the "
            "API server but not yet handed off to the model worker), sampled "
            "periodically. Companion to the "
            "'maxserve.num_requests_awaiting_admission' up/down counter: the "
            "counter is the live value for dashboards, while this histogram "
            "captures the distribution / tail (p50/p99) over time."
        ),
    ),  # type: ignore
    "maxserve.num_responses_buffered": _meter.create_gauge(
        "maxserve.num_responses_buffered",
        description=(
            "Egress backlog: model-worker responses received by the API "
            "server but not yet consumed by the streaming layer (sum of the "
            "per-request output-queue depths), sampled periodically. A "
            "persistently high value means the API server is shipping tokens "
            "back to clients (detokenize + serialize + network) slower than "
            "the model produces them, and the unbounded output queues are "
            "accumulating in API-process memory."
        ),
    ),  # type: ignore
    "maxserve.responses_buffered": _meter.create_histogram(
        "maxserve.responses_buffered",
        description=(
            "Distribution of the egress backlog (responses received by the "
            "API server but not yet streamed to clients), sampled "
            "periodically. Companion to the 'maxserve.num_responses_buffered' "
            "gauge: the gauge shows the latest value for live dashboards, "
            "while this histogram captures the distribution / tail (p50/p99) "
            "over time, which a scrape-interval gauge sample would miss."
        ),
    ),  # type: ignore
    "maxserve.response_queue_time": _meter.create_histogram(
        "maxserve.response_queue_time",
        unit="ms",
        description=(
            "Time a model-worker response waits in the API server's "
            "per-request output queue before the streaming layer consumes it. "
            "Sampled at the head of line (once per consumer wake), so it "
            "tracks the egress-side delay the user experiences when the API "
            "server falls behind the model on decode."
        ),
    ),  # type: ignore
    "maxserve.model_load_time": _meter.create_histogram(
        "maxserve.model_load_time",
        unit="ms",
        description=(
            "Time to load a model. Recorded once per model-worker startup, "
            "both as an untagged aggregate and split by the 'component' tag "
            "(build, compile, init, graph_capture, pinned_memory, spawn, "
            "total), mirroring the per-phase breakdown in the model worker's "
            "startup log lines."
        ),
    ),  # type: ignore
    "maxserve.itl": _meter.create_histogram(
        "maxserve.itl", unit="ms", description="inter token latency"
    ),  # type: ignore
    "maxserve.time_per_output_token": _meter.create_histogram(
        "maxserve.time_per_output_token",
        unit="ms",
        description=(
            "Mean decode-phase latency per generated token, emitted once per "
            "request: decode_time / (num_generated_tokens - 1). Excludes the "
            "first token and prefill/TTFT; accounts for speculative decoding."
        ),
    ),  # type: ignore
    "maxserve.pipeline_load": _meter.create_counter(
        "maxserve.pipeline_load",
        description="Count of pipelines loaded for each model",
    ),  # type: ignore
    "maxserve.batch_size": _meter.create_histogram(
        "maxserve.batch_size",
        description=(
            "Distribution of batch sizes (number of requests), labeled by "
            "'batch_type' (CE prefill or TG decode). For TG this is the "
            "decode batch size; for CE see 'batch_input_tokens' for the "
            "token-count view."
        ),
    ),  # type: ignore
    "maxserve.batch_execution_time": _meter.create_histogram(
        "maxserve.batch_execution_time",
        unit="ms",
        description="Distribution of batch execution time",
    ),  # type: ignore
    "maxserve.cache.num_used_blocks": _meter.create_gauge(
        "maxserve.cache.num_used_blocks",
        unit="blocks",
        description="Number of used blocks or pages, measured at the scheduler after batch work.",
    ),  # type: ignore
    "maxserve.cache.num_total_blocks": _meter.create_gauge(
        "maxserve.cache.num_total_blocks",
        unit="blocks",
        description="Total number of blocks or pages, measured at the scheduler after batch work.",
    ),  # type: ignore
    "maxserve.cache.hit_rate": _meter.create_histogram(
        "maxserve.cache.hit_rate",
        unit="percent_utilization",
        description=(
            "Per-request KV cache hit rate (cached prefix tokens / prompt "
            "tokens), emitted once per admitted request."
        ),
    ),  # type: ignore
    "maxserve.cache.preemption_count": _meter.create_counter(
        "maxserve.cache.preemption_count",
        description="Total number of preemptions",
    ),  # type: ignore
    "maxserve.cache.hits": _meter.create_counter(
        "maxserve.cache.hits",
        unit="tokens",
        description=(
            "Cumulative KV cache hit tokens across all CE batches "
            "(prompt tokens served from prefix cache)."
        ),
    ),  # type: ignore
    "maxserve.cache.misses": _meter.create_counter(
        "maxserve.cache.misses",
        unit="tokens",
        description=(
            "Cumulative KV cache miss tokens across all CE batches "
            "(prompt tokens actually prefilled by the model)."
        ),
    ),  # type: ignore
    "maxserve.input_tokens_per_request": _meter.create_histogram(
        "maxserve.input_tokens_per_request",
        unit="tokens",
        description="Distribution of input tokens per request",
    ),  # type: ignore
    "maxserve.output_tokens_per_request": _meter.create_histogram(
        "maxserve.output_tokens_per_request",
        unit="tokens",
        description="Distribution of output tokens per request",
    ),  # type: ignore
    "maxserve.dkv.nixl_read_latency": _meter.create_histogram(
        "maxserve.dkv.nixl_read_latency",
        unit="ms",
        description="NIXL READ transfer latency",
    ),  # type: ignore
    "maxserve.dkv.nixl_write_latency": _meter.create_histogram(
        "maxserve.dkv.nixl_write_latency",
        unit="ms",
        description="NIXL WRITE transfer latency",
    ),  # type: ignore
    "maxserve.dkv.rpc_acquire_latency": _meter.create_histogram(
        "maxserve.dkv.rpc_acquire_latency",
        unit="ms",
        description="dKV acquire_blocks RPC latency",
    ),  # type: ignore
    "maxserve.dkv.rpc_read_latency": _meter.create_histogram(
        "maxserve.dkv.rpc_read_latency",
        unit="ms",
        description="dKV read_blocks RPC latency",
    ),  # type: ignore
    "maxserve.spec_decode.acceptance_rate_per_position": _meter.create_histogram(
        "maxserve.spec_decode.acceptance_rate_per_position",
        unit="percent",
        description="Draft token acceptance rate per position (0-100%)",
    ),  # type: ignore
    "maxserve.batch_input_tokens": _meter.create_histogram(
        "maxserve.batch_input_tokens",
        unit="tokens",
        description="Distribution of input tokens per scheduler batch (CE prefill or TG decode).",
    ),  # type: ignore
    "maxserve.batch_context_tokens": _meter.create_histogram(
        "maxserve.batch_context_tokens",
        unit="tokens",
        description="Distribution of accumulated context tokens per scheduler batch.",
    ),  # type: ignore
    "maxserve.batch_creation_time": _meter.create_histogram(
        "maxserve.batch_creation_time",
        unit="ms",
        description="Distribution of scheduler batch creation time.",
    ),  # type: ignore
    "maxserve.batch_prompt_throughput": _meter.create_histogram(
        "maxserve.batch_prompt_throughput",
        unit="tokens/s",
        description="Per-batch prompt-side throughput in tokens/second.",
    ),  # type: ignore
    "maxserve.batch_generation_throughput": _meter.create_histogram(
        "maxserve.batch_generation_throughput",
        unit="tokens/s",
        description="Per-batch generation-side throughput in tokens/second.",
    ),  # type: ignore
    "maxserve.batch_terminated_reqs": _meter.create_histogram(
        "maxserve.batch_terminated_reqs",
        unit="reqs",
        description="Distribution of requests terminated per scheduler batch.",
    ),  # type: ignore
    "maxserve.batch_pending_reqs": _meter.create_histogram(
        "maxserve.batch_pending_reqs",
        unit="reqs",
        description="Distribution of requests pending in the queue, sampled once per scheduler batch.",
    ),  # type: ignore
    "maxserve.cache.used_kv_pct": _meter.create_histogram(
        "maxserve.cache.used_kv_pct",
        unit="percent",
        description="Percentage of device KV cache blocks in use (0-100%), sampled once per scheduler batch.",
    ),  # type: ignore
    "maxserve.cache.used_host_kv_pct": _meter.create_histogram(
        "maxserve.cache.used_host_kv_pct",
        unit="percent",
        description="Percentage of host KV cache blocks in use (0-100%), sampled once per scheduler batch when host paging is enabled.",
    ),  # type: ignore
    "maxserve.cache.h2d_blocks_copied": _meter.create_counter(
        "maxserve.cache.h2d_blocks_copied",
        unit="blocks",
        description="Cumulative host->device KV block copies.",
    ),  # type: ignore
    "maxserve.cache.d2h_blocks_copied": _meter.create_counter(
        "maxserve.cache.d2h_blocks_copied",
        unit="blocks",
        description="Cumulative device->host KV block copies.",
    ),  # type: ignore
    "maxserve.cache.disk_blocks_read": _meter.create_counter(
        "maxserve.cache.disk_blocks_read",
        unit="blocks",
        description="Cumulative KV blocks read from the disk cache tier.",
    ),  # type: ignore
    "maxserve.cache.disk_blocks_written": _meter.create_counter(
        "maxserve.cache.disk_blocks_written",
        unit="blocks",
        description="Cumulative KV blocks written to the disk cache tier.",
    ),  # type: ignore
    "maxserve.spec_decode.avg_acceptance_length": _meter.create_histogram(
        "maxserve.spec_decode.avg_acceptance_length",
        unit="tokens",
        description="Mean draft-token acceptance length per spec-decode batch.",
    ),  # type: ignore
    "maxserve.dkv.nixl_read_gib_per_s": _meter.create_histogram(
        "maxserve.dkv.nixl_read_gib_per_s",
        unit="GiB/s",
        description="NIXL READ throughput.",
    ),  # type: ignore
    "maxserve.dkv.nixl_write_gib_per_s": _meter.create_histogram(
        "maxserve.dkv.nixl_write_gib_per_s",
        unit="GiB/s",
        description="NIXL WRITE throughput.",
    ),  # type: ignore
    "maxserve.cache.used_disk_kv_pct": _meter.create_histogram(
        "maxserve.cache.used_disk_kv_pct",
        unit="percent",
        description="Percentage of disk KV cache blocks in use (0-100%), sampled once per scheduler batch when disk paging is enabled.",
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
    attributes: OtelAttributes | None = None
    time_unix_nano: int = field(default_factory=time.time_ns)

    def commit(self) -> None:
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
                logger.error(f"instrument is None for {self.instrument_name}")
                return

        # instrument should be one of the supported sdk types now
        if not isinstance(instrument, get_args(SDK_INSTRUMENTS)):
            # If you're hitting this, metrics were likely not configured properly.
            logger.error(
                f"instrument {self.instrument_name} is not one of the supported sdk types"
            )
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
        logger.debug(f"consumed measurement for {self.instrument_name}")


TelemetryFn = Callable[[MaxMeasurement], None]


class MetricClient(abc.ABC):
    @abc.abstractmethod
    def send_measurement(
        self, metric: MaxMeasurement, level: MetricLevel
    ) -> None: ...

    @abc.abstractmethod
    def cross_process_factory(
        self,
        settings: Settings,
    ) -> Callable[[], AbstractAsyncContextManager[MetricClient]]:
        """Get a copier for use of this client in another process.

        To use a MetricClient across processes, call cross_process_factory in
        the parent process and pass the result across the process boundary.
        Then in the child process, use an 'async with' to get a semantically
        identical MetricClient that can be used.

        This is needed because some metric clients require reinitialization on
        the other side of a process boundary before they can be safely used.
        """
        ...


@asynccontextmanager
async def _trivially_picklable_xprocess_factory(
    client: MetricClient,
) -> AsyncGenerator[MetricClient, None]:
    yield client


class NoopClient(MetricClient):
    def send_measurement(self, m: MaxMeasurement, level: MetricLevel) -> None:
        pass

    def cross_process_factory(
        self,
        settings: Settings,
    ) -> Callable[[], AbstractAsyncContextManager[MetricClient]]:
        return functools.partial(_trivially_picklable_xprocess_factory, self)


class SyncClient(MetricClient):
    def __init__(self, settings: Settings) -> None:
        self.level = settings.metric_level

    def send_measurement(self, m: MaxMeasurement, level: MetricLevel) -> None:
        if level > self.level:
            return
        m.commit()

    def cross_process_factory(
        self,
        settings: Settings,
    ) -> Callable[[], AbstractAsyncContextManager[MetricClient]]:
        return functools.partial(_trivially_picklable_xprocess_factory, self)


class _AsyncMetrics:
    """Centralizes metrics to encapsulate the OTEL dependency and avoid breaking schema changes

    Produce metric measurements to be consumed elsewhere
    """

    def __init__(self) -> None:
        self.client: MetricClient = NoopClient()
        self.extra_attributes: dict[str, str] = {}

    def configure(
        self,
        client: MetricClient,
        extra_attributes: dict[str, str] | None = None,
    ) -> None:
        self.client = client
        self.extra_attributes = extra_attributes or {}

    def request_count(self, responseCode: int, urlPath: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.request_count",
                1,
                {
                    **self.extra_attributes,
                    "code": f"{responseCode:d}",
                    "path": urlPath,
                },
            ),
            MetricLevel.BASIC,
        )

    def request_time(self, value: float, urlPath: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.request_time",
                value,
                {**self.extra_attributes, "path": urlPath},
            ),
            MetricLevel.BASIC,
        )

    def input_time(self, value: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.input_processing_time", value, self.extra_attributes
            ),
            MetricLevel.BASIC,
        )

    def output_time(self, value: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.output_processing_time", value, self.extra_attributes
            ),
            MetricLevel.BASIC,
        )

    def ttft(self, value: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.time_to_first_token", value, self.extra_attributes
            ),
            MetricLevel.BASIC,
        )

    def input_tokens(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.num_input_tokens", value, self.extra_attributes
            ),
            MetricLevel.BASIC,
        )

    def input_characters(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.num_input_characters", value, self.extra_attributes
            ),
            MetricLevel.BASIC,
        )

    def output_tokens(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.num_output_tokens", value, self.extra_attributes
            ),
            MetricLevel.BASIC,
        )

    def reqs_queued(self, value: int) -> None:
        """Publish the current depth of the scheduler's CE / prefill queue.

        ``maxserve.num_requests_queued`` is a synchronous gauge: every call
        replaces the previously reported value rather than accumulating.
        Schedulers should call this once per iteration with
        ``len(all_ce_reqs)`` (or the equivalent for their queue layout) at
        the same point that the "Pending: N reqs" log line is computed.
        """
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.num_requests_queued", value, self.extra_attributes
            ),
            MetricLevel.BASIC,
        )

    def reqs_running(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.num_requests_running", value, self.extra_attributes
            ),
            MetricLevel.BASIC,
        )

    def reqs_awaiting_admission(self, value: int) -> None:
        """Adjust the count of API-side requests not yet handed to the worker.

        ``maxserve.num_requests_awaiting_admission`` is an up/down counter:
        call with ``1`` when a request is accepted by the API server (before
        tokenization) and ``-1`` just before it is enqueued to the model
        worker. A persistently high value means requests are backing up in the
        API server (e.g. tokenization) rather than in the scheduler queue.
        """
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.num_requests_awaiting_admission",
                value,
                self.extra_attributes,
            ),
            MetricLevel.BASIC,
        )

    def requests_awaiting_admission_dist(self, value: int) -> None:
        """Record a sample of the ingress backlog for distribution analysis.

        Companion to :meth:`reqs_awaiting_admission` (the live up/down
        counter): a periodic sample of the same running count is fed into the
        ``maxserve.requests_awaiting_admission`` histogram so p50/p99 ingress
        backlog over time can be recovered.
        """
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.requests_awaiting_admission",
                value,
                self.extra_attributes,
            ),
            MetricLevel.BASIC,
        )

    def responses_buffered(self, value: int) -> None:
        """Publish the current egress backlog (sum of output-queue depths).

        ``maxserve.num_responses_buffered`` is a synchronous gauge: every call
        replaces the previously reported value. The API server should sample
        it periodically with the total number of model-worker responses
        received but not yet consumed by the streaming layer.
        """
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.num_responses_buffered",
                value,
                self.extra_attributes,
            ),
            MetricLevel.BASIC,
        )

    def responses_buffered_dist(self, value: int) -> None:
        """Record a sample of the egress backlog for distribution analysis.

        Companion to :meth:`responses_buffered` (the live gauge): the same
        periodic sample is also fed into the ``maxserve.responses_buffered``
        histogram so p50/p99 backlog over time can be recovered, which a
        scrape-interval gauge sample alone cannot provide.
        """
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.responses_buffered", value, self.extra_attributes
            ),
            MetricLevel.BASIC,
        )

    def response_queue_time(self, ms: float) -> None:
        """Record how long a response waited in the API output queue (ms)."""
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.response_queue_time", ms, self.extra_attributes
            ),
            MetricLevel.BASIC,
        )

    def model_load_time(self, ms: float, component: str | None = None) -> None:
        """Record a model-worker startup duration in milliseconds.

        Args:
            ms: The duration in milliseconds.
            component: Optional phase name (e.g. ``"build"``, ``"compile"``,
                ``"init"``, ``"graph_capture"``, ``"pinned_memory"``,
                ``"spawn"``, ``"total"``). Recorded as the ``component`` tag
                so a single metric can be split by startup phase. When
                omitted, records the untagged model-load aggregate.
        """
        attributes = self.extra_attributes
        if component is not None:
            attributes = {**attributes, "component": component}
        self.client.send_measurement(
            MaxMeasurement("maxserve.model_load_time", ms, attributes),
            MetricLevel.BASIC,
        )

    def itl(self, ms: float) -> None:
        self.client.send_measurement(
            MaxMeasurement("maxserve.itl", ms, self.extra_attributes),
            MetricLevel.BASIC,
        )

    def time_per_output_token(self, ms: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.time_per_output_token", ms, self.extra_attributes
            ),
            MetricLevel.BASIC,
        )

    def pipeline_load(self, name: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.pipeline_load",
                1,
                {**self.extra_attributes, "model": name},
            ),
            MetricLevel.BASIC,
        )

    def batch_size(self, size: int, batch_type: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.batch_size",
                size,
                {**self.extra_attributes, "batch_type": batch_type},
            ),
            MetricLevel.BASIC,
        )

    def batch_execution_time(
        self, execution_time: float, batch_type: str
    ) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.batch_execution_time",
                execution_time,
                {**self.extra_attributes, "batch_type": batch_type},
            ),
            MetricLevel.DETAILED,
        )

    def cache_num_used_blocks(self, num_used_blocks: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.cache.num_used_blocks",
                num_used_blocks,
                self.extra_attributes,
            ),
            MetricLevel.DETAILED,
        )

    def cache_num_total_blocks(self, total_blocks: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.cache.num_total_blocks",
                total_blocks,
                self.extra_attributes,
            ),
            MetricLevel.DETAILED,
        )

    def cache_hit_rate(self, hit_rate: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.cache.hit_rate", hit_rate, self.extra_attributes
            ),
            MetricLevel.BASIC,
        )

    def cache_hits(self, hits: int) -> None:
        self.client.send_measurement(
            MaxMeasurement("maxserve.cache.hits", hits, self.extra_attributes),
            MetricLevel.DETAILED,
        )

    def cache_misses(self, cache_misses: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.cache.misses", cache_misses, self.extra_attributes
            ),
            MetricLevel.DETAILED,
        )

    def preemption(self) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.cache.preemption_count", 1, self.extra_attributes
            ),
            MetricLevel.DETAILED,
        )

    def input_tokens_per_request(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.input_tokens_per_request",
                value,
                self.extra_attributes,
            ),
            MetricLevel.BASIC,
        )

    def output_tokens_per_request(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.output_tokens_per_request",
                value,
                self.extra_attributes,
            ),
            MetricLevel.BASIC,
        )

    def dkv_nixl_read_latency(self, latency_ms: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.dkv.nixl_read_latency",
                latency_ms,
                self.extra_attributes,
            ),
            MetricLevel.DETAILED,
        )

    def dkv_nixl_write_latency(self, latency_ms: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.dkv.nixl_write_latency",
                latency_ms,
                self.extra_attributes,
            ),
            MetricLevel.DETAILED,
        )

    def dkv_rpc_acquire_latency(self, latency_ms: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.dkv.rpc_acquire_latency",
                latency_ms,
                self.extra_attributes,
            ),
            MetricLevel.DETAILED,
        )

    def dkv_rpc_read_latency(self, latency_ms: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.dkv.rpc_read_latency",
                latency_ms,
                self.extra_attributes,
            ),
            MetricLevel.DETAILED,
        )

    def spec_decode_acceptance_rate_per_position(
        self, position: int, acceptance_rate: float
    ) -> None:
        """Emit draft token acceptance rate for a specific position.

        Args:
            position: The draft token position (0-indexed).
            acceptance_rate: The acceptance rate as a percentage (0-100).
        """
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.spec_decode.acceptance_rate_per_position",
                acceptance_rate,
                {**self.extra_attributes, "position": str(position)},
            ),
            MetricLevel.DETAILED,
        )

    def batch_input_tokens(self, value: int, batch_type: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.batch_input_tokens",
                value,
                {**self.extra_attributes, "batch_type": batch_type},
            ),
            MetricLevel.BASIC,
        )

    def batch_context_tokens(self, value: int, batch_type: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.batch_context_tokens",
                value,
                {**self.extra_attributes, "batch_type": batch_type},
            ),
            MetricLevel.BASIC,
        )

    def batch_creation_time(self, ms: float, batch_type: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.batch_creation_time",
                ms,
                {**self.extra_attributes, "batch_type": batch_type},
            ),
            MetricLevel.BASIC,
        )

    def batch_prompt_throughput(self, tps: float, batch_type: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.batch_prompt_throughput",
                tps,
                {**self.extra_attributes, "batch_type": batch_type},
            ),
            MetricLevel.BASIC,
        )

    def batch_generation_throughput(self, tps: float, batch_type: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.batch_generation_throughput",
                tps,
                {**self.extra_attributes, "batch_type": batch_type},
            ),
            MetricLevel.BASIC,
        )

    def batch_terminated_reqs(self, value: int, batch_type: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.batch_terminated_reqs",
                value,
                {**self.extra_attributes, "batch_type": batch_type},
            ),
            MetricLevel.DETAILED,
        )

    def batch_pending_reqs(self, value: int, batch_type: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.batch_pending_reqs",
                value,
                {**self.extra_attributes, "batch_type": batch_type},
            ),
            MetricLevel.DETAILED,
        )

    def cache_used_kv_pct(self, ratio: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.cache.used_kv_pct", ratio, self.extra_attributes
            ),
            MetricLevel.BASIC,
        )

    def cache_used_host_kv_pct(self, ratio: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.cache.used_host_kv_pct",
                ratio,
                self.extra_attributes,
            ),
            MetricLevel.DETAILED,
        )

    def cache_h2d_blocks_copied(self, count: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.cache.h2d_blocks_copied",
                count,
                self.extra_attributes,
            ),
            MetricLevel.DETAILED,
        )

    def cache_d2h_blocks_copied(self, count: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.cache.d2h_blocks_copied",
                count,
                self.extra_attributes,
            ),
            MetricLevel.DETAILED,
        )

    def cache_disk_blocks_read(self, count: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.cache.disk_blocks_read",
                count,
                self.extra_attributes,
            ),
            MetricLevel.DETAILED,
        )

    def cache_disk_blocks_written(self, count: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.cache.disk_blocks_written",
                count,
                self.extra_attributes,
            ),
            MetricLevel.DETAILED,
        )

    def cache_used_disk_kv_pct(self, ratio: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.cache.used_disk_kv_pct",
                ratio,
                self.extra_attributes,
            ),
            MetricLevel.DETAILED,
        )

    def spec_decode_avg_acceptance_length(self, length: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.spec_decode.avg_acceptance_length",
                length,
                self.extra_attributes,
            ),
            MetricLevel.DETAILED,
        )

    def dkv_nixl_read_gib_per_s(self, gib_per_s: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.dkv.nixl_read_gib_per_s",
                gib_per_s,
                self.extra_attributes,
            ),
            MetricLevel.DETAILED,
        )

    def dkv_nixl_write_gib_per_s(self, gib_per_s: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "maxserve.dkv.nixl_write_gib_per_s",
                gib_per_s,
                self.extra_attributes,
            ),
            MetricLevel.DETAILED,
        )


METRICS = _AsyncMetrics()
