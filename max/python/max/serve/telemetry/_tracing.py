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
"""Configures span export and GPU kernel tracing for a process."""

from __future__ import annotations

import atexit
import collections
import logging
import os
import signal
import threading
from collections.abc import Sequence
from types import FrameType
from urllib.parse import urlparse

from max.profiler import set_gpu_profiling_state
from max.serve.config import KernelTraceLevel, Settings
from max.serve.telemetry import _log_context
from max.serve.telemetry.common import _telemetry_disabled, logs_resource
from opentelemetry import trace
from opentelemetry.context import Context as OtelContext
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.sdk.trace import (
    ReadableSpan,
    SpanLimits,
    TracerProvider,
    sampling,
)
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import set_tracer_provider
from opentelemetry.util.types import Attributes

trace_level_header_enabled = False
"""Whether per-request trace levels are honored in this process: the API
server reads the ``x-max-trace-level`` header and the model worker may arm
captures. Set by :func:`configure_tracing` when it installs a provider and
``kernel_trace_headers`` is on. That call rebinds it, so read it through
:func:`trace_level_header_honored`, or through the module per request."""

_HTTP_PROTOBUF = "http/protobuf"
_GRPC = "grpc"


def _traces_protocol() -> str:
    """Returns the OTLP protocol to export spans over.

    Reads OTel's own ``OTEL_EXPORTER_OTLP_PROTOCOL`` convention, so an
    operator configures MAX Serve the way they configure anything else
    speaking OTLP, and the traces-specific variable wins over the generic
    one. Unset leaves MAX Serve on ``http/protobuf``, which is the only
    protocol it has ever spoken. ``http/json`` is in the specification but
    has no exporter in the Python SDK, so it is rejected with everything
    else unrecognised.
    """
    configured = os.environ.get(
        "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"
    ) or os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL")
    protocol = (configured or _HTTP_PROTOBUF).strip().lower()
    if protocol in (_GRPC, _HTTP_PROTOBUF):
        return protocol
    logging.getLogger(__name__).warning(
        "Unsupported OTLP traces protocol %r; exporting over %s.",
        protocol,
        _HTTP_PROTOBUF,
    )
    return _HTTP_PROTOBUF


def _tls_by_omission() -> str | None:
    """Returns the traces endpoint that gRPC will dial over TLS unasked.

    The SDK infers a plaintext channel only from an explicit ``http://``
    scheme, and ``urlparse`` reads a bare ``agent:4317`` as a URL whose
    scheme is the hostname — so the obvious in-cluster spelling is dialled
    over TLS and fails against a plaintext collector. An operator who set
    the insecure variable chose their transport, so leave them alone.
    """
    if os.environ.get("OTEL_EXPORTER_OTLP_TRACES_INSECURE") or os.environ.get(
        "OTEL_EXPORTER_OTLP_INSECURE"
    ):
        return None
    endpoint = os.environ.get(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
    ) or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint or urlparse(endpoint).scheme in ("http", "https"):
        return None
    return endpoint


def _grpc_span_exporter() -> SpanExporter:
    """Builds the gRPC span exporter, tolerating an HTTP-shaped environment.

    The imports are function-local, against the repository's top-level
    import rule, because they cost approximately 0.6s of cold start and 7MB
    of resident memory. Every import of this module pays that, while only a
    deployment that asks for gRPC needs it.
    """
    from grpc import Compression
    from opentelemetry.exporter.otlp.proto.grpc.exporter import (
        InvalidCompressionValueException,
    )
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter as OTLPGrpcSpanExporter,
    )

    logger = logging.getLogger(__name__)
    endpoint = _tls_by_omission()
    if endpoint is not None:
        endpoint = _without_credentials(endpoint)[0]
        logger.warning(
            "OTLP traces endpoint %r has no http:// or https:// scheme, so "
            "spans will be exported over TLS. Write it as http://%s for a "
            "plaintext collector, or set OTEL_EXPORTER_OTLP_TRACES_INSECURE.",
            endpoint,
            endpoint,
        )

    try:
        return OTLPGrpcSpanExporter()
    except InvalidCompressionValueException:
        # gRPC accepts only gzip, where HTTP also takes none and deflate.
        # The generic variable is shared with metrics and logs, which stay
        # on HTTP, so a setting that is valid for them must not take the
        # server down when traces move to gRPC.
        logger.warning(
            "OTLP traces compression %r is unavailable over gRPC, which "
            "accepts only gzip; exporting spans uncompressed.",
            os.environ.get("OTEL_EXPORTER_OTLP_TRACES_COMPRESSION")
            or os.environ.get("OTEL_EXPORTER_OTLP_COMPRESSION"),
        )
        return OTLPGrpcSpanExporter(compression=Compression.NoCompression)


def span_exporter() -> SpanExporter:
    """Builds the span exporter, leaving the endpoint to the SDK."""
    if _traces_protocol() == _GRPC:
        return _grpc_span_exporter()
    return OTLPSpanExporter()


def _without_credentials(endpoint: str) -> tuple[str, list[str]]:
    """Returns the endpoint as logs show it, and the parts it leaves out.

    Userinfo or a query can carry a credential. A gRPC endpoint may have no
    scheme, so one without ``://`` is read as an authority.
    """
    url = urlparse(endpoint if "://" in endpoint else "//" + endpoint)
    userinfo, _, host = url.netloc.rpartition("@")
    shown = url._replace(netloc=host, query="").geturl().removeprefix("//")
    return shown, [s for s in (userinfo, url.query) if s]


# An exporter that returns a failure code logs why only here. MAX sends only
# spans over gRPC, whose signals share one logger.
_EXPORT_ERROR_LOGGERS = (
    "opentelemetry.exporter.otlp.proto.http.trace_exporter",
    "opentelemetry.exporter.otlp.proto.grpc.exporter",
)


class _CountingQueue(collections.deque[ReadableSpan]):
    """The SDK's span queue, counting the spans it drops to make room.

    The SDK drops a full queue's oldest span without a word. The fullness
    check and the append share a lock with the worker's pops, so each drop
    is counted once.
    """

    def __init__(self, maxlen: int) -> None:
        super().__init__([], maxlen)
        self.dropped = 0
        self._lock = threading.Lock()

    def appendleft(self, span: ReadableSpan) -> None:
        with self._lock:
            if len(self) == self.maxlen:
                self.dropped += 1
            super().appendleft(span)

    def pop(self, *args: int) -> ReadableSpan:
        with self._lock:
            return super().pop(*args)

    def clear(self) -> None:
        # The SDK clears the queue in a forked child, where a thread that no
        # longer exists may hold the lock.
        self._lock = threading.Lock()
        super().clear()


class ReportingSpanExporter(SpanExporter):
    """Warns through a ``max.serve`` logger when span export starts failing,
    when it recovers, and once when a full queue has dropped spans.

    ``configure_logging()`` drops the SDK's own ``opentelemetry.*`` records,
    and admitting them would print a metrics-export traceback every interval
    wherever Modular's collector is unreachable.
    """

    def __init__(
        self,
        exporter: SpanExporter,
        what: str = "spans",
        queue_size: int | None = None,
        size_setting: str = "OTEL_BSP_MAX_QUEUE_SIZE",
    ) -> None:
        self._exporter = exporter
        self._what = what
        self._size_setting = size_setting
        self._endpoint, self._secrets = _without_credentials(
            os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"]
        )
        self._failing = False
        self._warned_of_drops = False
        # The replay's exporter logs to the same loggers from its own thread.
        self._thread = threading.local()
        self.queue_size = (
            queue_size or BatchSpanProcessor._default_max_queue_size()
        )
        self.queue = _CountingQueue(self.queue_size)
        for name in _EXPORT_ERROR_LOGGERS:
            logging.getLogger(name).addFilter(self._keep_error)

    def _keep_error(self, record: logging.LogRecord) -> bool:
        self._thread.error = record.getMessage()
        return True

    def _redact(self, text: str | None) -> str | None:
        for secret in self._secrets:
            text = text and text.replace(secret, "<redacted>")
        return text

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self._thread.error = None
        try:
            result = self._exporter.export(spans)
        except Exception as e:
            self._thread.error = f"{type(e).__name__}: {e}"
            result = SpanExportResult.FAILURE
        logger = logging.getLogger(__name__)
        if result is SpanExportResult.SUCCESS:
            if self._failing:
                self._failing = False
                logger.warning(
                    "Exporting %s to %s recovered.", self._what, self._endpoint
                )
        else:
            if not self._failing:
                self._failing = True
                logger.warning(
                    "Exporting %s to %s failed, and spans are lost until an"
                    " export succeeds. Error: %s",
                    self._what,
                    self._endpoint,
                    self._redact(self._thread.error),
                )
        if self.queue.dropped and not self._warned_of_drops:
            self._warned_of_drops = True
            logger.warning(
                "The queue of %s for %s is full and has dropped %d so far;"
                " raise %s (%d) or check that the collector keeps up.",
                self._what,
                self._endpoint,
                self.queue.dropped,
                self._size_setting,
                self.queue_size,
            )
        return result

    def shutdown(self) -> None:
        for name in _EXPORT_ERROR_LOGGERS:
            logging.getLogger(name).removeFilter(self._keep_error)
        self._exporter.shutdown()


class CountingSpanProcessor(BatchSpanProcessor):
    """Queues spans in :class:`ReportingSpanExporter`'s counting queue."""

    def __init__(self, exporter: ReportingSpanExporter) -> None:
        super().__init__(exporter, max_queue_size=exporter.queue_size)
        self._batch_processor._queue = exporter.queue


class _LinkedRootSampler(sampling.Sampler):
    """Samples a root span with a sampled link, such as ``max.batch``, so a
    pass follows its requests, and leaves every other span to ``fallback``.

    Args:
        fallback: The sampler the SDK builds from ``OTEL_TRACES_SAMPLER``.
    """

    def __init__(self, fallback: sampling.Sampler) -> None:
        self._fallback = fallback

    def should_sample(
        self,
        parent_context: OtelContext | None,
        trace_id: int,
        name: str,
        kind: trace.SpanKind | None = None,
        attributes: Attributes = None,
        links: Sequence[trace.Link] | None = None,
        trace_state: trace.TraceState | None = None,
    ) -> sampling.SamplingResult:
        if links and any(link.context.trace_flags.sampled for link in links):
            parent = trace.get_current_span(parent_context).get_span_context()
            if not parent.is_valid:
                return sampling.SamplingResult(
                    sampling.Decision.RECORD_AND_SAMPLE, attributes
                )
        return self._fallback.should_sample(
            parent_context, trace_id, name, kind, attributes, links, trace_state
        )

    def get_description(self) -> str:
        return f"LinkedRootSampler{{{self._fallback.get_description()}}}"


def configure_tracing(settings: Settings, max_links: int | None = None) -> None:
    """Installs the process's tracer provider if it is to export spans.

    Args:
        settings: Server settings.
        max_links: A per-span link limit to use in place of the SDK's.
    """
    global trace_level_header_enabled
    trace_level_header_enabled = False
    # Spans cost work on every request, so only the traces-specific variable
    # turns them on: the generic endpoint may be set for metrics alone.
    telemetry_on = not _telemetry_disabled(settings)
    export_spans = telemetry_on and bool(
        os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    )
    if export_spans:
        provider = TracerProvider(
            sampler=_LinkedRootSampler(sampling._get_from_env_or_default()),
            resource=logs_resource,
            span_limits=(
                SpanLimits(max_links=max_links)
                if max_links is not None
                else None
            ),
        )
        exporter = ReportingSpanExporter(span_exporter())
        provider.add_span_processor(CountingSpanProcessor(exporter))
        set_tracer_provider(provider)
        trace_level_header_enabled = settings.kernel_trace_headers

    logger = logging.getLogger()
    if export_spans:
        logger.info("Tracing initialized.")
    elif not telemetry_on:
        logger.info("Tracing disabled.")
    elif os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        # The generic endpoint turned traces on before, so say why they stop.
        logger.warning(
            "OTEL_EXPORTER_OTLP_ENDPOINT is set but spans need"
            " OTEL_EXPORTER_OTLP_TRACES_ENDPOINT; not exporting traces."
        )
    else:
        logger.info(
            "Tracing off; set OTEL_EXPORTER_OTLP_TRACES_ENDPOINT to export"
            " spans."
        )


_kernel_trace_level = KernelTraceLevel.OFF


def batch_spans_enabled() -> bool:
    """Returns whether ``max.batch`` spans should be emitted; that is, the model
    worker was configured with ``kernel_trace_level`` at ``batch`` or above."""
    return _kernel_trace_level >= KernelTraceLevel.BATCH


def trace_level_header_honored() -> bool:
    """Returns whether per-request trace levels are honored in this process."""
    return trace_level_header_enabled


def _exit_on_sigterm(signum: int, frame: FrameType | None) -> None:
    """Raises ``SystemExit`` so that SIGTERM runs the process's exit hooks.

    The profiler plugin writes the libkineto trace from an exit hook, which
    SIGTERM's default action skips.
    """
    raise SystemExit(128 + signum)


def _flush_spans_on_sigterm(signum: int, frame: FrameType | None) -> None:
    """Exports the queued spans for up to 3 s, then dies of the signal."""
    provider = trace.get_tracer_provider()
    if isinstance(provider, TracerProvider):
        # The SDK's force_flush ignores its timeout, so a thread bounds it.
        flush = threading.Thread(target=provider.force_flush, daemon=True)
        flush.start()
        flush.join(3)
    signal.signal(signum, signal.SIG_DFL)
    signal.raise_signal(signum)


def configure_kernel_tracing(settings: Settings) -> None:
    """Configures GPU kernel-trace capture based on ``kernel_trace_level``.

    Must be called in the model worker process before ``InferenceSession``
    is constructed so that the libkineto auto-start picks up the enabled
    flag. Also records the level read by :func:`batch_spans_enabled`, so it
    must run before the scheduler is built. It may install a SIGTERM
    handler, so it must run on the main thread after :func:`configure_tracing`.

    Args:
        settings: Server settings carrying ``kernel_trace_level``.
    """
    global _kernel_trace_level
    level = settings.kernel_trace_level
    _kernel_trace_level = level
    if _log_context.tracing_enabled() and level < KernelTraceLevel.KERNEL:
        signal.signal(signal.SIGTERM, _flush_spans_on_sigterm)
    if level < KernelTraceLevel.OP:
        return

    if level == KernelTraceLevel.KERNEL:
        # Full libkineto kernel timeline: enable detailed NVTX tracing and
        # the libkineto auto-start that fires on InferenceSession construction.
        set_gpu_profiling_state("detailed")
        os.environ.setdefault("MODULAR_MAX_DEBUG_PROFILING_ENABLED", "true")
        signal.signal(signal.SIGTERM, _exit_on_sigterm)
        # Once exit starts, SIGTERM must not interrupt the trace write.
        # Finalization resets a Python handler to SIG_DFL but keeps SIG_IGN.
        atexit.register(signal.signal, signal.SIGTERM, signal.SIG_IGN)
    else:
        # OP level: op-level NVTX user-annotation ranges only.
        set_gpu_profiling_state("on")

    logging.getLogger("max.serve").info(
        "Kernel tracing initialized: level=%s", level.value
    )
