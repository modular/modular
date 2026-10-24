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

import json
import logging
import logging.handlers
import os
import platform
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from time import time

import requests
from max.serve.config import Settings
from max.serve.telemetry import _log_context

# Bound by name: the tracing-off formatter reads it on every line.
from max.serve.telemetry._log_context import request_id_ctx as _request_id_ctx
from max.serve.telemetry.metrics import (
    HISTOGRAM_SHADOW_SUFFIX,
    configure_histogram_shadow_emission,
)
from max.support._telemetry import (
    LATENCY_BUCKETS_MS,
    SkipExponentialHistogramsPrometheusReader,
    drop_metrics_matching,
    log_spaced_buckets,
)
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
    OTLPMetricExporter,
)
from opentelemetry.metrics import set_meter_provider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import Counter, Histogram, MeterProvider
from opentelemetry.sdk.metrics._internal.aggregation import (
    ExplicitBucketHistogramAggregation,
    ExponentialBucketHistogramAggregation,
)
from opentelemetry.sdk.metrics._internal.point import (
    Histogram as ExplicitHistogramData,
)
from opentelemetry.sdk.metrics.export import (
    AggregationTemporality,
    MetricReader,
    MetricsData,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.metrics.view import View
from opentelemetry.sdk.resources import Resource
from pythonjsonlogger import jsonlogger

otelBaseUrl = "https://telemetry.modular.com:443"

# Non-empty so the SDK skips OTEL_EXPORTER_OTLP_*HEADERS; it sends this anyway.
_MODULAR_OTLP_HEADERS = {"Content-Type": "application/x-protobuf"}


def _operator_set_endpoint(signal_var: str) -> bool:
    """Whether the operator set ``signal_var`` or the generic endpoint variable.

    The exporter must then be built with no explicit endpoint so the SDK can
    resolve it: passing the value read here as ``endpoint=`` would let the
    generic variable beat the signal-specific one.
    """
    return bool(
        os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        or os.environ.get(signal_var)
    )


def _telemetry_disabled(settings: Settings) -> bool:
    """Whether egress is off, by MAX setting or by ``OTEL_SDK_DISABLED``.

    The SDK reads that variable too but only no-ops instruments, leaving the
    providers, their exporter threads and ``send_telemetry_log``'s raw POST
    running; reading it here stops those. Parsed as the SDK parses it.
    """
    if os.environ.get("OTEL_SDK_DISABLED", "").lower().strip() == "true":
        return True
    return settings.disable_telemetry


def _getCloudProvider() -> str:
    providers = ["amazon", "google", "microsoft", "oracle"]
    path = "/sys/class/dmi/id/"
    if os.path.isdir(path):
        for idFile in os.listdir(path):
            try:
                with open(idFile) as file:
                    contents = file.read().lower()
                    for provider in providers:
                        if provider in contents:
                            return provider
            except Exception:
                pass
    return ""


def _getWebUserId() -> str:
    try:
        idFile = os.path.expanduser("~") + "/.modular/webUserId"
        with open(idFile) as file:
            return file.readline().rstrip("\n")
    except Exception:
        return ""


_LOGS_RESOURCE_ATTRS = {
    "event.domain": "serve",
    "telemetry.session": uuid.uuid4().hex,
    "web.user.id": _getWebUserId(),
    "enduser.id": os.environ.get("MODULAR_USER_ID", ""),
    "os.type": platform.system(),
    "os.version": platform.release(),
    "cpu.description": platform.processor(),
    "cpu.arch": platform.architecture()[0],
    "system.cloud": _getCloudProvider(),
    "deployment.id": os.environ.get("MAX_SERVE_DEPLOYMENT_ID", ""),
}
logs_resource = Resource.create(_LOGS_RESOURCE_ATTRS)

_METRICS_RESOURCE_ATTRS = {
    "enduser.id": os.environ.get("MODULAR_USER_ID", ""),
    "deployment.id": os.environ.get("MAX_SERVE_DEPLOYMENT_ID", ""),
}
metrics_resource = Resource.create(_METRICS_RESOURCE_ATTRS)


HISTOGRAM_LATENCY_BUCKETS_MS: tuple[float, ...] = LATENCY_BUCKETS_MS

# Percentages / utilization ratios (0-100). Linear rather than geometric, since
# utilization is only actionable near saturation: 5% steps through the bulk,
# tightening to 1% above 90% so a nearly-full KV cache tier resolves to a
# single point instead of a range. Kept to 29 boundaries because several of
# these metrics carry a label that multiplies the series count (batch_type,
# draft position).
HISTOGRAM_PERCENT_BUCKETS: tuple[float, ...] = tuple(
    float(pct) for pct in (*range(0, 91, 5), *range(91, 101))
)

# Token counts per request/batch and other unbounded counts. Log-spaced from 1
# up to 100M, so batch-level token totals over very long contexts stay on scale.
HISTOGRAM_COUNT_BUCKETS: tuple[float, ...] = log_spaced_buckets(
    1.0, 100_000_000.0, 1.5
)

# Batch size (number of requests in a batch), from a single request up to the
# largest batch the scheduler forms.
HISTOGRAM_BATCH_SIZE_BUCKETS: tuple[float, ...] = (
    0.0,
    *(float(size) for size in range(1, 21)),  # Tracks exactly for 1-20
    # [2:] drops the ladder's leading 0.0 and its 20.0, already present.
    *log_spaced_buckets(20.0, 512.0, 1.05)[2:],  # 5% steps from 21 - 512
)

# Mean speculative-decode acceptance length per batch. Because it is an average
# it takes fractional values, so use fine 0.25-wide buckets from 0 to 16 tokens
# (boundaries 0.25, 0.50, ..., 16.0).
HISTOGRAM_ACCEPTANCE_LENGTH_BUCKETS: tuple[float, ...] = tuple(
    i * 0.25 for i in range(1, 65)
)

# Throughput in tokens/second, from a slow single request to a large prefill
# batch.
HISTOGRAM_THROUGHPUT_TOKENS_BUCKETS: tuple[float, ...] = log_spaced_buckets(
    10.0, 1_000_000.0, 1.5
)

# Transfer throughput in GiB/s (e.g. NIXL over fast fabric). Coarser steps than
# the other ladders: this spans four decades and only feeds transfer dashboards.
HISTOGRAM_GIB_PER_S_BUCKETS: tuple[float, ...] = log_spaced_buckets(
    0.1, 800.0, 2.0
)

# Encoded media payload sizes. Log-spaced 1 KiB to 1 GiB at ratio 2 (22
# boundaries): the per-request media budget defaults to 100 MiB and a single
# item can use all of it, and the limit is disableable (Settings.max_media_bytes
# is ``ge=0``), so the ladder clears the default with headroom.
HISTOGRAM_BYTES_BUCKETS: tuple[float, ...] = log_spaced_buckets(
    1024.0, 1024.0**3, 2.0
)

# Image pixel counts (width * height), pre-resize. 19 boundaries, 1 K to 100 M:
# 1 K covers a 32x32 thumbnail, 100 M a 10000x10000 image, above which the
# decoded-size estimate rejects it against the default budget anyway.
HISTOGRAM_PIXELS_BUCKETS: tuple[float, ...] = log_spaced_buckets(
    1024.0, 1.0e8, 2.0
)

# Per-metric histogram bucket boundaries, matched by exact instrument name.
#
# Each Histogram instrument is matched to exactly one View by its exact name, so
# there is no risk of an instrument matching multiple Views (which would emit
# duplicate Prometheus series). Any histogram not listed here falls back to the
# OTEL SDK default buckets, so keep this in sync with the Histogram instruments
# in metrics.py SERVE_METRICS. A unit test
# (test_all_histograms_have_explicit_buckets) enforces that.
HISTOGRAM_BUCKETS_BY_METRIC: dict[str, tuple[float, ...]] = {
    # Latency / time (ms)
    "maxserve.request_time": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.input_processing_time": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.output_processing_time": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.time_to_first_token": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.time_to_first_token_by_prompt_length": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.cache.connector_cordon_wait_time": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.response_queue_time": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.model_load_time": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.itl": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.time_per_output_token": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.batch_execution_time": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.scheduler.batch_step_time": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.batch_creation_time": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.di.api_pre_tokenize": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.di.api_submit": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.di.mw_queue_wait": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.di.decode_admission_queue_wait_time": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.di.decode_send_time": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.di.dispatch_rtt": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.di.prefill_span": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.di.reply_rtt": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.di.prefill_queue_wait_time": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.ce_pool_wait_time": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.di.decode_postprocess_time": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.di.early_sync_time": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.di.handoff_to_first_token_time": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.structured_output.grammar_build_time": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.video.encoding_time_milliseconds": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.vision.image_admission_decode_time": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.media.resolve_time": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.dkv.nixl_read_latency": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.dkv.nixl_read_latency_max": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.dkv.nixl_write_latency": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.dkv.rpc_acquire_latency": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.dkv.rpc_read_latency": HISTOGRAM_LATENCY_BUCKETS_MS,
    # Percentages
    "maxserve.dp_active_token_occupancy": HISTOGRAM_PERCENT_BUCKETS,
    "maxserve.dp_context_token_occupancy": HISTOGRAM_PERCENT_BUCKETS,
    "maxserve.cache.request_prefix_coverage": HISTOGRAM_PERCENT_BUCKETS,
    "maxserve.prefix_coverage_per_request": HISTOGRAM_PERCENT_BUCKETS,
    "maxserve.cache.used_kv_pct": HISTOGRAM_PERCENT_BUCKETS,
    "maxserve.cache.used_host_kv_pct": HISTOGRAM_PERCENT_BUCKETS,
    "maxserve.cache.used_disk_kv_pct": HISTOGRAM_PERCENT_BUCKETS,
    "maxserve.spec_decode.acceptance_rate_per_position": HISTOGRAM_PERCENT_BUCKETS,
    "maxserve.vision.cache_hit_rate": HISTOGRAM_PERCENT_BUCKETS,
    # Generic counts (tokens / occupancy)
    "maxserve.input_tokens_per_request": HISTOGRAM_COUNT_BUCKETS,
    "maxserve.output_tokens_per_request": HISTOGRAM_COUNT_BUCKETS,
    "maxserve.batch_input_tokens": HISTOGRAM_COUNT_BUCKETS,
    "maxserve.batch_context_tokens": HISTOGRAM_COUNT_BUCKETS,
    "maxserve.batch_terminated_reqs": HISTOGRAM_COUNT_BUCKETS,
    "maxserve.batch_pending_reqs": HISTOGRAM_COUNT_BUCKETS,
    "maxserve.requests_awaiting_admission": HISTOGRAM_COUNT_BUCKETS,
    "maxserve.responses_buffered": HISTOGRAM_COUNT_BUCKETS,
    "maxserve.di.ce_preempted_tg_pending_count": HISTOGRAM_COUNT_BUCKETS,
    # Batch size
    "maxserve.batch_size": HISTOGRAM_BATCH_SIZE_BUCKETS,
    # Real tool inventories sit in the low tens; 512 clears all of them.
    "maxserve.tool_call.tools_per_request": HISTOGRAM_BATCH_SIZE_BUCKETS,
    # MiniMax-M3's video processor samples up to 512 frames per clip
    # (see max_private/minimax_m3/vision_processor.py); this bucket set's
    # upper bound (512) matches exactly.
    "maxserve.video.frames_per_clip": HISTOGRAM_BATCH_SIZE_BUCKETS,
    # Media items of one kind on a single request; the same ladder as batch
    # size, since both count per-request work items.
    "maxserve.media.items_per_request": HISTOGRAM_BATCH_SIZE_BUCKETS,
    # Exact through 20 passes, where a cordon that costs a pass or two lives.
    "maxserve.cache.connector_cordon_passes": HISTOGRAM_BATCH_SIZE_BUCKETS,
    # Throughput (tokens/s)
    "maxserve.batch_prompt_throughput": HISTOGRAM_THROUGHPUT_TOKENS_BUCKETS,
    "maxserve.batch_generation_throughput": HISTOGRAM_THROUGHPUT_TOKENS_BUCKETS,
    # Encoded media payload bytes
    "maxserve.media.item_size": HISTOGRAM_BYTES_BUCKETS,
    # Image pixel counts
    "maxserve.media.image_size": HISTOGRAM_PIXELS_BUCKETS,
    # Transfer throughput (GiB/s)
    "maxserve.dkv.nixl_read_gib_per_s": HISTOGRAM_GIB_PER_S_BUCKETS,
    "maxserve.dkv.nixl_write_gib_per_s": HISTOGRAM_GIB_PER_S_BUCKETS,
    # Acceptance length (tokens)
    "maxserve.spec_decode.avg_acceptance_length": HISTOGRAM_ACCEPTANCE_LENGTH_BUCKETS,
}


def get_log_level(settings: Settings) -> int | str | None:
    otlp_level: int | str | None = (
        logging.getLevelName(settings.logs_otlp_level)
        if settings.logs_otlp_level
        else None
    )

    if _telemetry_disabled(settings):
        otlp_level = None

    return otlp_level


COLOR_MAP = {
    "green": "\033[92m",
    "blue": "\033[94m",
    "red": "\033[91m",
}


class PrefixFormatter(logging.Formatter):
    """Custom formatter that adds a prefix to log messages."""

    def __init__(self, prefix: str, base_formatter: logging.Formatter):
        super().__init__()
        self.prefix = prefix
        self.base_formatter = base_formatter

    def format(self, record: logging.LogRecord) -> str:
        # Format the message using the base formatter
        formatted_message = self.base_formatter.format(record)
        # Add the prefix to the message
        return f"{self.prefix} {formatted_message}"


class _CorrelatedJsonFormatter(jsonlogger.JsonFormatter):
    """Renders the correlation IDs into this handler's JSON only.

    ``configure_logging`` uses it only with tracing on. The IDs go into the
    output dict, never onto the record: every handler shares one record, and
    the OTLP handler exports its attributes to Modular, where an inbound
    trace ID is the caller's data.
    """

    def add_fields(
        self,
        log_record: dict[str, object],
        record: logging.LogRecord,
        message_dict: dict[str, object],
    ) -> None:
        super().add_fields(log_record, record, message_dict)
        log_record.update(_log_context.correlation_fields())


class _RequestIdJsonFormatter(jsonlogger.JsonFormatter):
    """Renders only ``request_id``, the same way, for tracing off."""

    def add_fields(
        self,
        log_record: dict[str, object],
        record: logging.LogRecord,
        message_dict: dict[str, object],
    ) -> None:
        super().add_fields(log_record, record, message_dict)
        request_id = _request_id_ctx.get()
        if request_id is not None:
            log_record["request_id"] = request_id


def _log_exporter() -> OTLPLogExporter:
    return OTLPLogExporter(
        endpoint=otelBaseUrl + "/v1/logs", headers=_MODULAR_OTLP_HEADERS
    )


# Configure logging to console and OTEL.  This should be called before any
# 3rd party imports whose logging you wish to capture.
# Note that the color is not propagated to subprocesses. eg: ModelWorker
def configure_logging(
    settings: Settings,
    color: str | None = None,
    silent: bool = True,
    *,
    sets_request_ids: bool = True,
) -> None:
    # Structured logs carry request_id; only traced requests capture the
    # trace context, so without tracing there is no dd.trace_id field.
    correlate = _log_context.tracing_enabled()
    json_formatter: type[jsonlogger.JsonFormatter] = (
        _CorrelatedJsonFormatter if correlate else _RequestIdJsonFormatter
    )
    if not (correlate or sets_request_ids):
        json_formatter = jsonlogger.JsonFormatter
    trace_id_field = " %(dd.trace_id)s" if correlate else ""
    otlp_level = get_log_level(settings)
    egress_enabled = not _telemetry_disabled(settings)

    logging_handlers: list[logging.Handler] = []

    # Set up log filtering
    # ``uvicorn`` owns the HTTP error log: an exception escaping the ASGI app,
    # a malformed request, and the cancellation of in-flight requests when the
    # graceful-shutdown drain expires are all reported there and nowhere else.
    # Dropping them left connection-level failures with no server-side trace at
    # all. The ``uvicorn`` logger is pinned to WARNING below, so this admits
    # warnings and errors without the per-request ``uvicorn.access`` stream.
    components_to_log = [
        "root",
        "max._entrypoints",
        "max.pipelines",
        "max.serve",
        "uvicorn",
    ]
    try:
        if settings.logs_enable_components is not None:
            components = settings.logs_enable_components.split(",")
            components_to_log.extend(components)
    except Exception:
        print(
            "ERROR: Failed to parse logging components setting!  Using default."
        )

    def LogFilter(record: logging.LogRecord) -> bool:
        # Check if the logger name starts with any of the allowed component prefixes
        # This handles hierarchical logger names like "max.pipelines.architectures.llama3"
        return any(
            record.name == component or record.name.startswith(component + ".")
            for component in components_to_log
        )

    # Create a console handler
    if settings.logs_console_level is not None:
        if color is not None:
            if color not in COLOR_MAP:
                raise ValueError(f"Invalid color: {color}")
            color_code = COLOR_MAP[color]
            color_terminator = "\033[0m"
        else:
            color_code = ""
            color_terminator = ""

        console_handler = logging.StreamHandler()
        console_formatter: logging.Formatter
        if settings.structured_logging:
            console_formatter = json_formatter(
                f"{color_code}%(levelname)s: %(message)s %(request_id)s{trace_id_field}{color_terminator}",
                timestamp=True,
            )
        else:
            console_formatter = logging.Formatter(
                (
                    f"{color_code}%(asctime)s.%(msecs)03d %(levelname)s:{color_terminator} %(message)s"
                ),
                datefmt="%H:%M:%S",
            )

        # Apply log prefix if provided
        if settings.log_prefix is not None:
            console_formatter = PrefixFormatter(
                settings.log_prefix, console_formatter
            )

        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(settings.logs_console_level)
        console_handler.addFilter(LogFilter)

        logging_handlers.append(console_handler)

    if (
        settings.logs_file_level is not None
        and settings.logs_file_path is not None
    ):
        # Create a file handler
        file_handler = logging.FileHandler(settings.logs_file_path)
        file_formatter: logging.Formatter
        if settings.structured_logging:
            file_formatter = json_formatter(
                f"%(levelname)s %(message)s %(request_id)s{trace_id_field}",
                timestamp=True,
            )
        else:
            file_formatter = logging.Formatter(
                ("%(asctime)s.%(msecs)03d %(levelname)s: %(message)s"),
                datefmt="%y:%m:%d-%H:%M:%S",
            )

        # Apply log prefix if provided
        if settings.log_prefix is not None:
            file_formatter = PrefixFormatter(
                settings.log_prefix, file_formatter
            )

        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(settings.logs_file_level)
        file_handler.addFilter(LogFilter)
        logging_handlers.append(file_handler)

    if egress_enabled and otlp_level is not None:
        # Create an OTEL handler
        logger_provider = LoggerProvider(logs_resource)
        set_logger_provider(logger_provider)
        exporter = _log_exporter()
        logger_provider.add_log_record_processor(
            BatchLogRecordProcessor(exporter)
        )
        otlp_handler = LoggingHandler(
            level=logging.getLevelName(otlp_level),
            logger_provider=logger_provider,
        )
        otlp_handler.addFilter(LogFilter)
        logging_handlers.append(otlp_handler)

    # Configure root logger level
    logger = logging.getLogger()
    if len(logging_handlers) > 0:
        # Clear existing handlers to prevent duplicates when configure_logging is called multiple times
        logger.handlers.clear()

        logger_level = min(h.level for h in logging_handlers)
        logger.setLevel(logger_level)
        for handler in logging_handlers:
            logger.addHandler(handler)

        # TODO use FastAPIInstrumentor once Motel supports traces.
        # For now, manually configure uvicorn.
        logging.getLogger("uvicorn").setLevel(logging.WARNING)
        # Explicit levels to reduce noise
        logging.getLogger("sse_starlette.sse").setLevel(
            max(logger_level, logging.INFO)
        )

    if not silent:
        logger.info(
            "Logging initialized: Console: %s, File: %s, Telemetry: %s",
            settings.logs_console_level,
            settings.logs_file_level,
            otlp_level,
        )


class _ExponentialShadowOnlyReader(PeriodicExportingMetricReader):
    """Drops explicit-bucket histogram data points; keeps everything else.

    The complement of
    ``max.support._telemetry.SkipExponentialHistogramsPrometheusReader``:
    the
    explicit-bucket histograms already reach Prometheus unchanged (above),
    so this reader — used only for the opt-in
    ``MAX_SERVE_OTLP_METRICS_ENDPOINT`` — carries just the exponential
    shadow histograms (plus counters/gauges) to that endpoint, keeping the
    two histogram representations on separate destinations for MXSERV-258
    side-by-side comparison rather than duplicating the explicit-bucket one
    there too. That separation assumes this endpoint differs from the one the
    primary reader resolves: aimed at the same collector, counters and gauges
    arrive twice, cumulative from that reader and delta from this one.
    """

    def _receive_metrics(
        self,
        metrics_data: MetricsData,
        timeout_millis: float = 10_000,
        **kwargs: object,
    ) -> None:
        filtered = drop_metrics_matching(
            metrics_data, lambda m: isinstance(m.data, ExplicitHistogramData)
        )
        if filtered is not None:
            super()._receive_metrics(filtered, timeout_millis, **kwargs)


def _histogram_views(settings: Settings) -> list[View]:
    # One View per histogram, matched by its exact instrument name, so each
    # metric gets bucket boundaries tuned to its actual range (latency in
    # ms, percentages, token/occupancy counts, batch sizes, throughput,
    # ...). Always active, regardless of MAX_SERVE_OTLP_METRICS_ENDPOINT:
    # Prometheus keeps showing exactly these, unchanged.
    #
    # Matching by exact name guarantees every Histogram matches at most one
    # of these Views, so we never get duplicate Prometheus series from
    # overlapping Views. Any histogram missing from the map falls back to
    # the OTEL SDK default buckets; test_all_histograms_have_explicit_buckets
    # guards against that.
    views: list[View] = [
        View(
            instrument_name=name,
            aggregation=ExplicitBucketHistogramAggregation(buckets),
        )
        for name, buckets in HISTOGRAM_BUCKETS_BY_METRIC.items()
    ]

    if settings.otlp_metrics_endpoint:
        # Exponential shadow of every histogram (metrics.HISTOGRAM_SHADOW_SUFFIX),
        # matched by name glob so new histograms are covered automatically.
        # Self-calibrating: no per-metric bucket boundaries to hand-tune, and
        # (unlike the explicit-bucket histograms above) mergeable across
        # replicas, so a Datadog-side p99 across the fleet is a mathematically
        # valid percentile rather than an average/max of independent
        # per-replica estimates. Reaches only the OTLP endpoint
        # (_ExponentialShadowOnlyReader), side by side with the
        # explicit-bucket histograms on Prometheus, for MXSERV-258 comparison.
        views.append(
            View(
                instrument_name="*" + HISTOGRAM_SHADOW_SUFFIX,
                aggregation=ExponentialBucketHistogramAggregation(),
            )
        )

    return views


@contextmanager
def _sdk_disable_masked() -> Iterator[None]:
    """Hides ``OTEL_SDK_DISABLED`` from an SDK provider constructor.

    ``MeterProvider`` would otherwise return no-op instruments, emptying the
    local Prometheus surface and making every measurement log an error.
    Egress is decided by ``_telemetry_disabled`` and the readers below.
    """
    saved = os.environ.pop("OTEL_SDK_DISABLED", None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ["OTEL_SDK_DISABLED"] = saved


def _metric_exporter() -> OTLPMetricExporter:
    """Builds the metric exporter, falling back to Modular's collector."""
    if _operator_set_endpoint("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"):
        return OTLPMetricExporter()
    return OTLPMetricExporter(
        endpoint=otelBaseUrl + "/v1/metrics", headers=_MODULAR_OTLP_HEADERS
    )


def configure_metrics(settings: Settings) -> None:
    egress_enabled = not _telemetry_disabled(settings)
    configure_histogram_shadow_emission(bool(settings.otlp_metrics_endpoint))

    meter_readers: list[MetricReader] = [
        SkipExponentialHistogramsPrometheusReader(True)
    ]
    if egress_enabled:
        meter_readers.append(PeriodicExportingMetricReader(_metric_exporter()))
    if settings.otlp_metrics_endpoint:
        meter_readers.append(
            _ExponentialShadowOnlyReader(
                OTLPMetricExporter(
                    endpoint=settings.otlp_metrics_endpoint,
                    preferred_temporality={
                        Histogram: AggregationTemporality.DELTA,
                        Counter: AggregationTemporality.DELTA,
                    },
                )
            )
        )

    with _sdk_disable_masked():
        provider = MeterProvider(
            metric_readers=meter_readers,
            resource=metrics_resource,
            views=_histogram_views(settings),
        )
    set_meter_provider(provider)

    logger = logging.getLogger()
    if not egress_enabled:
        logger.info("OTLP metrics export disabled.")
    else:
        logger.info("Metrics initialized.")


def _string_attrs(pairs: Mapping[str, object]) -> list[dict[str, object]]:
    """Renders a mapping as OTLP string-valued key/value attributes."""
    return [
        {"key": key, "value": {"stringValue": str(value)}}
        for key, value in pairs.items()
    ]


# Send a simple one-time structured log, avoiding the buggy OTEL SDK
# (see MAXSERV-904)
def send_telemetry_log(model_name: str) -> None:
    attrs = logs_resource.attributes
    body = {
        "resourceLogs": [
            {
                "resource": {
                    "attributes": _string_attrs(
                        {
                            "deployment.model": model_name,
                            "web.user.id": attrs["web.user.id"],
                            "enduser.id": attrs["enduser.id"],
                            "deployment.id": attrs["deployment.id"],
                            "os.type": attrs["os.type"],
                            "os.version": attrs["os.version"],
                            "cpu.description": attrs["cpu.description"],
                            "cpu.arch": attrs["cpu.arch"],
                            "system.cloud": attrs["system.cloud"],
                            "service.name": attrs["service.name"],
                            "telemetry.sdk.language": "python",
                            "telemetry.sdk.version": "0.0.0",
                            "telemetry.sdk.name": "opentelemetry",
                        }
                    )
                },
                "scopeLogs": [
                    {
                        "logRecords": [
                            {
                                "attributes": _string_attrs(
                                    {
                                        "event.domain": "modular",
                                        "event.name": "serve.telemetry.log",
                                    }
                                ),
                                "body": {"stringValue": ""},
                                "observedTimeUnixNano": str(
                                    int(time() * 1_000_000_000)
                                ),
                                "severityNumber": 9,
                                "severityText": "INFO",
                            }
                        ],
                        "scope": {"name": "modular_logger"},
                    }
                ],
            }
        ]
    }

    requests.post(
        otelBaseUrl + "/v1/logs",
        data=json.dumps(body),
        headers={"Content-Type": "application/json"},
        timeout=2,
    )
