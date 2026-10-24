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
"""Tests that MAX Serve honours the standard OTel SDK configuration
environment variables.

MAX deliberately supplies no ``service.name``, no sampler settings and, once an
operator has set an endpoint variable, no endpoint — leaving each to the
SDK's own resolution. Those omissions decide the behaviour but are
invisible at the call site, so they are pinned here.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent import futures
from unittest import mock

import grpc
import pytest
from max.serve.config import Settings
from max.serve.telemetry import _tracing, common
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter as OTLPGrpcSpanExporter,
)
from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
    OTLPMetricExporter,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.proto.collector.trace.v1 import (
    trace_service_pb2,
    trace_service_pb2_grpc,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    InMemoryMetricReader,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider, sampling
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

GENERIC = "OTEL_EXPORTER_OTLP_ENDPOINT"
TRACES = "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
METRICS = "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"
PROTOCOL = "OTEL_EXPORTER_OTLP_PROTOCOL"
TRACES_PROTOCOL = "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"


@pytest.fixture(autouse=True)
def clear_otel_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Isolates each test from the environment it inherits.

    The bazel target sets MAX_SERVE_DISABLE_TELEMETRY, which reaches Settings
    by alias and would otherwise decide the outcome of tests about the OTel
    variables.
    """
    for name in [n for n in os.environ if n.startswith("OTEL_")]:
        monkeypatch.delenv(name)
    monkeypatch.delenv("MAX_SERVE_DISABLE_TELEMETRY", raising=False)
    yield


@pytest.mark.parametrize(
    ("build", "signal_var", "path"),
    [
        pytest.param(
            common._metric_exporter, METRICS, "/v1/metrics", id="metrics"
        ),
    ],
)
@pytest.mark.parametrize(
    "headers_var",
    [
        "OTEL_EXPORTER_OTLP_HEADERS",
        "OTEL_EXPORTER_OTLP_METRICS_HEADERS",
        "OTEL_EXPORTER_OTLP_LOGS_HEADERS",
    ],
)
def test_exporter_endpoint_precedence(
    monkeypatch: pytest.MonkeyPatch,
    build: Callable[[], OTLPMetricExporter],
    signal_var: str,
    path: str,
    headers_var: str,
) -> None:
    """The last two endpoint assertions are the ones that matter: reading either
    variable's value and passing it as ``endpoint=`` would let the generic one
    win, inverting the precedence the OTel spec requires."""
    monkeypatch.setenv(headers_var, "dd-api-key=secret")
    assert build()._endpoint == common.otelBaseUrl + path
    assert "dd-api-key" not in build()._session.headers
    assert "dd-api-key" not in common._log_exporter()._session.headers

    monkeypatch.setenv(GENERIC, "http://agent:4318")
    assert build()._endpoint == "http://agent:4318" + path
    if "LOGS" not in headers_var:
        assert "dd-api-key" in build()._session.headers

    monkeypatch.setenv(signal_var, "http://specific:4318/sink")
    assert build()._endpoint == "http://specific:4318/sink"

    monkeypatch.delenv(GENERIC)
    assert build()._endpoint == "http://specific:4318/sink"


def test_traces_protocol_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Signal-specific beats generic, as for the endpoint variables — an
    operator moving only traces onto gRPC must not have to move metrics and
    logs with them."""
    assert _tracing._traces_protocol() == "http/protobuf"

    monkeypatch.setenv(PROTOCOL, "grpc")
    assert _tracing._traces_protocol() == "grpc"

    monkeypatch.setenv(TRACES_PROTOCOL, "http/protobuf")
    assert _tracing._traces_protocol() == "http/protobuf"

    monkeypatch.setenv(PROTOCOL, "http/protobuf")
    monkeypatch.setenv(TRACES_PROTOCOL, "grpc")
    assert _tracing._traces_protocol() == "grpc"


@pytest.mark.parametrize("value", ["http/json", " GRPC "])
def test_traces_protocol_parsing(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """Case and surrounding whitespace are forgiven; anything the SDK has no
    exporter for degrades to HTTP rather than failing startup, because a
    typo in a deploy config should cost spans, not the server."""
    monkeypatch.setenv(PROTOCOL, value)
    expected = "grpc" if value.strip().lower() == "grpc" else "http/protobuf"
    assert _tracing._traces_protocol() == expected


def test_protocol_does_not_reach_metrics_or_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scope wall: this change is traces only. The generic protocol variable
    is deliberately not honoured for the other two signals, which still go to
    Modular's collector over HTTP."""
    monkeypatch.setenv(GENERIC, "http://agent:4317")
    monkeypatch.setenv(PROTOCOL, "grpc")
    assert isinstance(common._metric_exporter(), OTLPMetricExporter)


@pytest.mark.parametrize("value", ["none", "deflate"])
def test_grpc_tolerates_an_http_only_compression(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """gRPC accepts only gzip, where HTTP also takes none and deflate, and
    the generic variable is shared with metrics and logs that stay on HTTP.
    Left alone the SDK raises out of the constructor, so a setting that is
    valid for the other two signals would crash-loop the worker the moment
    traces moved to gRPC."""
    monkeypatch.setenv(TRACES, "http://agent:4317")
    monkeypatch.setenv(PROTOCOL, "grpc")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_COMPRESSION", value)
    assert isinstance(_tracing.span_exporter(), OTLPGrpcSpanExporter)


@pytest.mark.parametrize(
    ("endpoint", "warns"),
    [
        ("http://agent:4317", False),
        ("agent:4317", True),
    ],
)
def test_grpc_warns_when_the_endpoint_omits_its_scheme(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    endpoint: str,
    warns: bool,
) -> None:
    """``urlparse`` reads a bare ``agent:4317`` as a URL whose scheme is the
    hostname, so the SDK does not infer plaintext and dials TLS instead.
    Against a plaintext collector that surfaces only as spans never arriving,
    which is why it is worth a line in the log at startup."""
    monkeypatch.setenv(TRACES, endpoint)
    monkeypatch.setenv(PROTOCOL, "grpc")
    with caplog.at_level("WARNING"):
        _tracing.span_exporter()
    assert ("exported over TLS" in caplog.text) is warns


def test_explicit_insecure_silences_the_scheme_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An operator who set the insecure variable has chosen their transport,
    and the scheme no longer decides it."""
    monkeypatch.setenv(TRACES, "agent:4317")
    assert _tracing._tls_by_omission() == "agent:4317"

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_INSECURE", "true")
    assert _tracing._tls_by_omission() is None


class _RecordingTraceService(trace_service_pb2_grpc.TraceServiceServicer):
    """The receiver half of the gRPC egress test below."""

    def __init__(self) -> None:
        self.requests: list[trace_service_pb2.ExportTraceServiceRequest] = []

    def Export(
        self,
        request: trace_service_pb2.ExportTraceServiceRequest,
        context: grpc.ServicerContext,
    ) -> trace_service_pb2.ExportTraceServiceResponse:
        self.requests.append(request)
        return trace_service_pb2.ExportTraceServiceResponse()


def test_grpc_spans_reach_a_receiver(monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end over a real OTLP/gRPC receiver, because every assertion
    above stops at which exporter was constructed. What this adds is that the
    span leaves: a plaintext channel is inferred from the ``http://`` scheme,
    so an in-cluster collector needs no TLS configuration, and the payload
    carries MAX's resource."""
    receiver = _RecordingTraceService()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    trace_service_pb2_grpc.add_TraceServiceServicer_to_server(receiver, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        monkeypatch.setenv(TRACES, f"http://127.0.0.1:{port}")
        monkeypatch.setenv(PROTOCOL, "grpc")
        provider = TracerProvider(resource=common.logs_resource)
        exporter = _tracing.span_exporter()
        provider.add_span_processor(BatchSpanProcessor(exporter))
        with provider.get_tracer("test").start_as_current_span("max.request"):
            pass
        provider.force_flush(5000)
        provider.shutdown()
    finally:
        server.stop(2).wait(5)

    assert len(receiver.requests) == 1
    spans = receiver.requests[0].resource_spans[0].scope_spans[0].spans
    assert [span.name for span in spans] == ["max.request"]


@pytest.mark.parametrize(
    "attrs",
    [common._LOGS_RESOURCE_ATTRS, common._METRICS_RESOURCE_ATTRS],
    ids=["logs", "metrics"],
)
def test_resource_leaves_service_name_to_the_sdk(
    monkeypatch: pytest.MonkeyPatch, attrs: dict[str, str]
) -> None:
    """``Resource.create`` merges the env-derived resource first and explicit
    attributes last, so a ``service.name`` passed by MAX would beat
    ``OTEL_SERVICE_NAME`` and strip an operator's ability to name the
    service. Rebuilt here under a chosen value: asserting on the imported
    resource would pass against a MAX-hardcoded ``unknown_service`` too."""
    monkeypatch.setenv("OTEL_SERVICE_NAME", "operator-chosen")
    assert (
        Resource.create(attrs).attributes["service.name"] == "operator-chosen"
    )


def test_configure_tracing_defers_to_the_sdk_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Captured at the call site because the global provider is write-once, so
    a test that installed one would couple to test order."""
    captured: list[TracerProvider] = []
    monkeypatch.setattr(_tracing, "set_tracer_provider", captured.append)
    # Only the traces endpoint installs a provider.
    monkeypatch.setenv(TRACES, "http://localhost:4318/v1/traces")
    monkeypatch.setenv("OTEL_TRACES_SAMPLER", "traceidratio")
    monkeypatch.setenv("OTEL_TRACES_SAMPLER_ARG", "0.25")
    _tracing.configure_tracing(Settings(disable_telemetry=False))
    captured[0].shutdown()
    sampler = captured[0].sampler
    assert isinstance(sampler, _tracing._LinkedRootSampler)
    assert isinstance(sampler._fallback, sampling.TraceIdRatioBased)


def _span_context(n: int, sampled: bool) -> trace.SpanContext:
    return trace.SpanContext(n, n, True, trace.TraceFlags(sampled))


@pytest.mark.parametrize(
    ("parent", "link", "sampled"),
    [
        pytest.param(None, None, False, id="root"),
        pytest.param(None, True, True, id="linked"),
        pytest.param(None, False, False, id="unsampled-link"),
        pytest.param(False, True, False, id="unsampled-parent"),
        pytest.param(True, None, True, id="sampled-parent"),
    ],
)
def test_a_root_with_a_sampled_link_is_sampled(
    parent: bool | None, link: bool | None, sampled: bool
) -> None:
    """Every other span gets the fallback's decision."""
    fallback = sampling.ParentBased(sampling.ALWAYS_OFF)
    provider = TracerProvider(sampler=_tracing._LinkedRootSampler(fallback))
    span = provider.get_tracer("test").start_span(
        "max.batch",
        context=None
        if parent is None
        else trace.set_span_in_context(
            trace.NonRecordingSpan(_span_context(1, parent))
        ),
        links=None if link is None else [trace.Link(_span_context(2, link))],
        attributes={"max.batch.id": 1},
    )
    assert span.get_span_context().trace_flags.sampled is sampled
    if sampled:
        # The SDK takes a span's attributes from the sampling result only.
        assert isinstance(span, ReadableSpan)
        assert span.attributes == {"max.batch.id": 1}


@pytest.mark.parametrize(
    ("env", "installed"),
    [
        pytest.param({}, False, id="default"),
        pytest.param({GENERIC: "http://agent:4318"}, False, id="generic"),
        pytest.param(
            {TRACES: "http://localhost:4318/v1/traces"}, True, id="traces"
        ),
        pytest.param(
            {
                TRACES: "http://localhost:4318/v1/traces",
                "OTEL_SDK_DISABLED": "true",
            },
            False,
            id="traces-sdk-disabled",
        ),
    ],
)
def test_only_the_traces_endpoint_turns_tracing_on(
    monkeypatch: pytest.MonkeyPatch, env: Mapping[str, str], installed: bool
) -> None:
    """With no provider installed the global stays OTel's
    ProxyTracerProvider, so spans are no-ops and ``tracing_enabled()``
    skips the scheduler's bookkeeping. The generic endpoint must
    not count: a deployment may set it for metrics."""
    captured: list[TracerProvider] = []
    reporter = mock.Mock(wraps=_tracing.ReportingSpanExporter)
    monkeypatch.setattr(_tracing, "set_tracer_provider", captured.append)
    monkeypatch.setattr(_tracing, "ReportingSpanExporter", reporter)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    _tracing.configure_tracing(Settings(disable_telemetry=False))
    assert len(captured) == int(installed)
    assert reporter.call_count == int(installed)
    for provider in captured:
        provider.shutdown()


def test_configure_tracing_installs_the_grpc_exporter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``configure_tracing`` installs the gRPC exporter when the traces
    protocol is ``grpc``, not only ``span_exporter`` in isolation."""
    reporter = mock.Mock(wraps=_tracing.ReportingSpanExporter)
    captured: list[TracerProvider] = []
    monkeypatch.setattr(_tracing, "set_tracer_provider", captured.append)
    monkeypatch.setattr(_tracing, "ReportingSpanExporter", reporter)
    monkeypatch.setenv(TRACES, "http://agent:4317")
    monkeypatch.setenv(TRACES_PROTOCOL, "grpc")
    _tracing.configure_tracing(Settings(disable_telemetry=False))
    for provider in captured:
        provider.shutdown()

    assert len(captured) == 1
    exporter = reporter.call_args.args[0]
    assert isinstance(exporter, OTLPGrpcSpanExporter)
    assert exporter._endpoint == "agent:4317"


@pytest.mark.parametrize(
    ("env", "level"),
    [
        pytest.param({}, logging.INFO, id="default"),
        pytest.param(
            {GENERIC: "http://agent:4318"}, logging.WARNING, id="generic"
        ),
    ],
)
def test_tracing_off_warns_only_when_the_generic_endpoint_is_set(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    env: Mapping[str, str],
    level: int,
) -> None:
    """The generic endpoint used to turn traces on, so an operator who set
    only it is told why spans stopped."""
    monkeypatch.setattr(_tracing, "set_tracer_provider", lambda _: None)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with caplog.at_level(logging.INFO):
        _tracing.configure_tracing(Settings(disable_telemetry=False))
    assert [r.levelno for r in caplog.records] == [level]


def test_span_exporter_uses_the_traces_endpoint_as_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(GENERIC, "http://agent:4318")
    monkeypatch.setenv(TRACES, "http://specific:4318/sink")
    exporter = _tracing.span_exporter()
    assert isinstance(exporter, OTLPSpanExporter)
    assert exporter._endpoint == "http://specific:4318/sink"


@pytest.mark.parametrize(
    ("value", "disabled"),
    [
        ("true", True),
        ("TRUE", True),
        (" true ", True),
        ("1", False),
    ],
)
def test_otel_sdk_disabled(
    monkeypatch: pytest.MonkeyPatch, value: str, disabled: bool
) -> None:
    """Only the literal ``true`` disables, per the spec, and surrounding
    whitespace is stripped — matching the SDK's own parse so that the two
    cannot disagree about a value and leave telemetry half-off."""
    monkeypatch.setenv("OTEL_SDK_DISABLED", value)
    settings = Settings(disable_telemetry=False)
    assert common._telemetry_disabled(settings) is disabled


def test_sdk_disable_is_masked_from_the_meter_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A MeterProvider that reads OTEL_SDK_DISABLED itself hands back no-op
    instruments, which empties the local Prometheus endpoint and makes every
    measurement log an error. Masking the variable keeps that surface live
    while MAX still drops the OTLP reader."""
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    assert MeterProvider()._disabled is True
    with common._sdk_disable_masked():
        assert "OTEL_SDK_DISABLED" not in os.environ
        assert MeterProvider()._disabled is False
    assert os.environ["OTEL_SDK_DISABLED"] == "true"


def test_configure_metrics_keeps_prometheus_live_when_sdk_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pairing that motivates the mask: OTLP export stops, the local
    Prometheus surface keeps real instruments. Stubbing the Prometheus reader
    because a second real one would clash on the process-global registry."""
    captured: list[MeterProvider] = []
    monkeypatch.setattr(common, "set_meter_provider", captured.append)
    monkeypatch.setattr(
        common,
        "SkipExponentialHistogramsPrometheusReader",
        lambda *a, **k: InMemoryMetricReader(),
    )
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    common.configure_metrics(Settings(disable_telemetry=False))
    provider = captured[0]
    assert provider._disabled is False
    assert not any(
        isinstance(r, PeriodicExportingMetricReader)
        for r in provider._sdk_config.metric_readers
    )


def _install_tracing(monkeypatch: pytest.MonkeyPatch) -> TracerProvider:
    captured: list[TracerProvider] = []
    monkeypatch.setattr(_tracing, "set_tracer_provider", captured.append)
    _tracing.configure_tracing(Settings(disable_telemetry=False))
    [provider] = captured
    return provider


def _export_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage() for r in caplog.records if r.name == _tracing.__name__
    ]


@pytest.mark.parametrize(
    ("protocol", "endpoint", "error"),
    [
        (
            "http/protobuf",
            "http://user:hunter2@127.0.0.1:1?token=s3cr3t",
            "ConnectionError",
        ),
        ("grpc", "http://127.0.0.1:1", "StatusCode."),
    ],
)
def test_a_refused_endpoint_warns_once(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    protocol: str,
    endpoint: str,
    error: str,
) -> None:
    """HTTP raises out of ``export()``; gRPC returns a failure code and names
    the error only in a log record that MAX's log filter drops. Neither
    warning shows the endpoint's credentials, and shutdown unhooks the
    reporter from the SDK's loggers."""
    monkeypatch.setenv(TRACES, endpoint)
    monkeypatch.setenv(TRACES_PROTOCOL, protocol)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_TIMEOUT", "1")
    filters = list(logging.getLogger(OTLPSpanExporter.__module__).filters)
    provider = _install_tracing(monkeypatch)
    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            provider.get_tracer("test").start_span("max.request").end()
            provider.force_flush()
    provider.shutdown()
    assert logging.getLogger(OTLPSpanExporter.__module__).filters == filters

    [failure] = _export_warnings(caplog)
    assert failure.startswith("Exporting spans to http://127.0.0.1:1 failed")
    assert error in failure
    assert "hunter2" not in failure and "s3cr3t" not in failure


def test_a_scheme_less_endpoint_hides_its_userinfo() -> None:
    """A gRPC endpoint may be written without a scheme."""
    assert _tracing._without_credentials("user:hunter2@agent:4317") == (
        "agent:4317",
        ["user:hunter2"],
    )


class _HungExporter(SpanExporter):
    """Holds its second export until released and fails it, as a collector
    that times out would; every other export succeeds."""

    def __init__(self) -> None:
        self.calls = 0
        self.entered = threading.Event()
        self.release = threading.Event()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self.calls += 1
        if self.calls != 2:
            return SpanExportResult.SUCCESS
        self.entered.set()
        self.release.wait(5)
        logger = logging.getLogger(OTLPSpanExporter.__module__)
        logger.error("timed out")
        # The replay's exporter failing on its own thread meanwhile.
        other = threading.Thread(target=logger.error, args=("replay failed",))
        other.start()
        other.join()
        return SpanExportResult.FAILURE


def test_a_full_queue_and_a_recovery_each_warn_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A full queue drops its oldest spans without blocking or logging."""
    exporter = _HungExporter()
    monkeypatch.setattr(_tracing, "span_exporter", lambda: exporter)
    monkeypatch.setenv(TRACES, "http://agent:4318/v1/traces")
    monkeypatch.setenv("OTEL_BSP_MAX_QUEUE_SIZE", "4")
    monkeypatch.setenv("OTEL_BSP_MAX_EXPORT_BATCH_SIZE", "2")
    provider = _install_tracing(monkeypatch)
    tracer = provider.get_tracer("test")
    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            tracer.start_span("max.batch").end()
        provider.force_flush()
        # The next two wake the worker, which takes both and hangs.
        for _ in range(2):
            tracer.start_span("max.batch").end()
        assert exporter.entered.wait(5)
        # The queue keeps the last four of these.
        for _ in range(20):
            tracer.start_span("max.batch").end()
        exporter.release.set()
        provider.shutdown()

    endpoint = "http://agent:4318/v1/traces"
    assert _export_warnings(caplog) == [
        f"Exporting spans to {endpoint} failed, and spans are lost until an"
        " export succeeds. Error: timed out",
        f"The queue of spans for {endpoint} is full and has dropped 16 so"
        " far; raise OTEL_BSP_MAX_QUEUE_SIZE (4) or check that the"
        " collector keeps up.",
        f"Exporting spans to {endpoint} recovered.",
    ]


def test_a_burst_before_any_export_warns_of_drops(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Holding the export lock keeps the worker from taking any span."""
    monkeypatch.setattr(_tracing, "span_exporter", InMemorySpanExporter)
    monkeypatch.setenv(TRACES, "http://agent:4318/v1/traces")
    monkeypatch.setenv("OTEL_BSP_MAX_QUEUE_SIZE", "4")
    monkeypatch.setenv("OTEL_BSP_MAX_EXPORT_BATCH_SIZE", "4")
    provider = _install_tracing(monkeypatch)
    [processor] = provider._active_span_processor._span_processors
    assert isinstance(processor, _tracing.CountingSpanProcessor)
    with caplog.at_level(logging.WARNING):
        with processor._batch_processor._export_lock:
            for _ in range(5):
                provider.get_tracer("test").start_span("max.batch").end()
        provider.shutdown()

    [dropped] = _export_warnings(caplog)
    assert "has dropped 1 so far" in dropped
