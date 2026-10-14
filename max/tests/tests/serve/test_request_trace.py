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
"""With tracing on, a request's logs and spans share its server span's trace."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Iterator, Mapping
from pathlib import Path
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from max.pipelines.context import GenerationStatus, TextContext, TokenBuffer
from max.pipelines.lib import PipelineConfig
from max.pipelines.modeling.types import RequestID
from max.serve import request as request_module
from max.serve.api_server import ServingTokenGeneratorSettings, fastapi_app
from max.serve.config import APIType, Settings
from max.serve.pipelines.echo_gen import (
    EchoPipelineTokenizer,
    EchoTokenGenerator,
)
from max.serve.pipelines.llm import TokenGeneratorOutput
from max.serve.request import register_request
from max.serve.router import openai_routes
from max.serve.telemetry import _trace_context, common
from max.serve.telemetry._trace_context import inject_trace_carrier
from max.serve.telemetry.common import configure_logging
from opentelemetry import propagate as otel_propagate
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF

_TRACEPARENT = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
_TRACE_ID_HEX = "4bf92f3577b34da6a3ce929d0e0e4736"


@pytest.fixture
def structured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """Structured file logging with tracing on, undone afterwards."""
    monkeypatch.setattr(common, "_tracing_enabled", lambda: True)
    root = logging.getLogger()
    saved = (list(root.handlers), root.level)
    log_path = tmp_path / "max-serve.json"
    configure_logging(
        Settings(
            logs_console_level=None,
            logs_file_level="WARNING",
            logs_file_path=str(log_path),
            structured_logging=True,
            disable_telemetry=True,
        )
    )
    try:
        yield log_path
    finally:
        handlers, level = saved
        for handler in list(root.handlers):
            if handler not in handlers:
                handler.close()
        root.handlers[:] = handlers
        root.setLevel(level)


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
    """Records the server and ``max.request`` spans from a local SDK provider.

    The global provider is set-once and would leak across the target.
    """
    finished = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(finished))
    tracer = provider.get_tracer("test")
    monkeypatch.setattr(_trace_context, "_tracer", tracer)
    monkeypatch.setattr(request_module, "_tracing_enabled", lambda: True)
    monkeypatch.setattr(openai_routes, "_tracer", tracer)
    return finished


def _server_spans(
    spans: InMemorySpanExporter, paths: list[str]
) -> list[ReadableSpan]:
    """Requests ``paths`` through the middleware alone; returns server spans."""
    app = FastAPI()
    register_request(app)

    @app.get("/work")
    async def work() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/crash")
    async def crash() -> dict[str, bool]:
        raise RuntimeError("unhandled")

    @app.get("/health")
    @app.get("/v1/health")
    @app.get("/v2/health/live")
    @app.get("/v2/health/ready")
    async def health() -> dict[str, bool]:
        return {"ok": True}

    app.mount("/metrics", FastAPI())
    with TestClient(app) as client:
        for path in paths:
            client.get(path)
    return [
        s for s in spans.get_finished_spans() if s.kind == trace.SpanKind.SERVER
    ]


def test_no_server_span_with_tracing_off(
    structured: Path,
    spans: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With tracing off, structured logging sets only the request ID: no
    # trace capture and no server span, not even a no-op one.
    monkeypatch.setattr(request_module, "_tracing_enabled", lambda: False)
    app = FastAPI()
    register_request(app, structured_logging=True)
    seen: list[object] = []

    @app.get("/work")
    async def work() -> dict[str, bool]:
        seen.extend(
            [common._request_id_ctx.get(), common.request_trace_ctx.get()]
        )
        return {"ok": True}

    with TestClient(app) as client:
        response = client.get("/work", headers={"traceparent": _TRACEPARENT})

    assert seen == [response.headers["X-Request-ID"], None]
    assert _server_spans(spans, ["/work"]) == []


def test_probes_get_no_server_span(spans: InMemorySpanExporter) -> None:
    # Health checks and metric scrapes would bury real requests in spans.
    (server,) = _server_spans(
        spans,
        [
            "/health",
            "/v1/health",
            "/v2/health/live",
            "/v2/health/ready",
            "/metrics/",
            "/work",
        ],
    )

    assert server.name == "GET /work"


def test_server_span_records_a_failed_request(
    spans: InMemorySpanExporter,
) -> None:
    # Datadog's error rate by route reads the status and these attributes.
    (server,) = _server_spans(spans, ["/crash"])

    assert server.name == "GET /crash"
    assert server.status.status_code == trace.StatusCode.ERROR
    assert server.attributes is not None
    assert server.attributes["http.response.status_code"] == 500
    assert server.attributes["http.route"] == "/crash"
    assert server.attributes["url.scheme"] == "http"


def test_server_span_ends_after_the_last_chunk(
    spans: InMemorySpanExporter,
) -> None:
    # Ending with the headers would make a stream's span cover only the time
    # to its first byte.
    app = FastAPI()
    register_request(app)
    recording: list[bool] = []

    @app.get("/stream")
    async def stream() -> StreamingResponse:
        span = trace.get_current_span(common.request_trace_ctx.get())

        async def body() -> AsyncIterator[bytes]:
            yield b"a"
            yield b"b"
            recording.append(span.is_recording())

        return StreamingResponse(body())

    with TestClient(app) as client:
        response = client.get("/stream")

    assert response.content == b"ab"
    assert recording == [True]
    (server,) = [
        s for s in spans.get_finished_spans() if s.kind == trace.SpanKind.SERVER
    ]
    assert server.name == "GET /stream"
    assert server.attributes is not None
    assert server.attributes["http.response.status_code"] == 200


def test_server_span_ends_for_a_plain_response(
    spans: InMemorySpanExporter,
) -> None:
    (server,) = _server_spans(spans, ["/work"])

    assert server.status.status_code == trace.StatusCode.UNSET
    assert server.attributes is not None
    assert server.attributes["http.response.status_code"] == 200


@pytest.mark.asyncio
async def test_client_disconnect_mid_stream_ends_the_server_span(
    spans: InMemorySpanExporter,
) -> None:
    # The body never finishes on its own, so only the disconnect can end it.
    app = FastAPI()
    register_request(app)
    first_chunk_sent = asyncio.Event()

    @app.get("/stream")
    async def stream() -> StreamingResponse:
        async def body() -> AsyncIterator[bytes]:
            yield b"a"
            await asyncio.Event().wait()
            yield b"never"

        return StreamingResponse(body())

    requested = False

    async def receive() -> dict[str, object]:
        nonlocal requested
        if not requested:
            requested = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await first_chunk_sent.wait()
        return {"type": "http.disconnect"}

    async def send(message: Mapping[str, object]) -> None:
        if message["type"] == "http.response.body" and message.get("body"):
            first_chunk_sent.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/stream",
        "raw_path": b"/stream",
        "query_string": b"",
        "root_path": "",
        "headers": [],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }
    await asyncio.wait_for(app(scope, receive, send), timeout=30)

    (server,) = [
        s for s in spans.get_finished_spans() if s.kind == trace.SpanKind.SERVER
    ]
    assert server.name == "GET /stream"


def test_unsampled_server_span_still_parents_the_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A sampled-out server span must still parent max.request and the
    # worker's spans, so parent-based sampling drops them with it.
    tracer = TracerProvider(sampler=ALWAYS_OFF).get_tracer("test")
    monkeypatch.setattr(_trace_context, "_tracer", tracer)
    monkeypatch.setattr(request_module, "_tracing_enabled", lambda: True)
    app = FastAPI()
    register_request(app)
    seen: list[trace.Span] = []

    @app.get("/work")
    async def work() -> dict[str, bool]:
        seen.append(trace.get_current_span(common.request_trace_ctx.get()))
        return {"ok": True}

    with TestClient(app) as client:
        client.get("/work")

    (span,) = seen
    assert not span.is_recording()
    assert span.get_span_context().is_valid


def _mock_chat_request() -> Mock:
    request = Mock()
    request.request_id = RequestID("test")
    request.model_name = "test-model"
    request.tools = None
    request.response_format = None
    request.timestamp_ns = 1
    request.request_path = "/v1/chat/completions"
    request.sampling_params = Mock()
    request.sampling_params.stop = []
    request.messages = []
    return request


@pytest.mark.parametrize(
    "headers",
    [{"traceparent": _TRACEPARENT}, {}],
    ids=["traceparent", "no-traceparent"],
)
def test_logs_and_spans_share_the_server_spans_trace(
    structured: Path,
    spans: InMemorySpanExporter,
    mock_pipeline_config: PipelineConfig,
    monkeypatch: pytest.MonkeyPatch,
    headers: dict[str, str],
) -> None:
    # Through the real app and a real chat handler: with or without a
    # traceparent, the logs, max.request and the worker's carrier share the
    # server span's trace.
    carriers: list[dict[str, str]] = []

    async def all_tokens(request: object) -> list[TokenGeneratorOutput]:
        # Never current, so the OTLP log handler keeps stamping zero IDs.
        assert not trace.get_current_span().get_span_context().is_valid
        logging.getLogger("max.serve").warning("generating")
        context = TextContext(
            request_id=RequestID(),
            max_length=16,
            tokens=TokenBuffer(np.ones(4, dtype=np.int64)),
        )
        inject_trace_carrier(context)
        carriers.append(context.trace_carrier or {})
        return [
            TokenGeneratorOutput(
                status=GenerationStatus.END_OF_SEQUENCE,
                decoded_tokens="hi",
                prompt_token_count=4,
            )
        ]

    pipeline = Mock()
    pipeline.model_name = "test-model"
    pipeline.all_tokens = all_tokens

    async def chat() -> dict[str, bool]:
        await openai_routes.OpenAIChatResponseGenerator(pipeline).complete(
            [_mock_chat_request()]
        )
        return {"ok": True}

    app = fastapi_app(
        Settings(api_types=[APIType.KSERVE], use_heartbeat=False),
        ServingTokenGeneratorSettings(
            model_factory=EchoTokenGenerator,
            pipeline_config=mock_pipeline_config,
            tokenizer=EchoPipelineTokenizer(),
        ),
    )
    app.add_api_route("/work", chat)
    monkeypatch.setattr(openai_routes, "METRICS", MagicMock())
    monkeypatch.setattr(openai_routes, "record_request_start", Mock())
    monkeypatch.setattr(openai_routes, "record_request_end", Mock())
    response = TestClient(app).get("/work", headers=headers)

    assert response.status_code == 200
    finished = spans.get_finished_spans()
    (server,) = [s for s in finished if s.kind == trace.SpanKind.SERVER]
    (request_span,) = [s for s in finished if s.name == "max.request"]
    assert server.name == "GET /work"
    assert server.attributes is not None
    assert server.attributes["http.response.status_code"] == 200
    assert server.context is not None and request_span.context is not None
    trace_id = server.context.trace_id
    if headers:
        assert trace.format_trace_id(trace_id) == _TRACE_ID_HEX
    assert request_span.parent is not None
    assert request_span.parent.span_id == server.context.span_id
    assert request_span.context.trace_id == trace_id
    # The carrier parents the worker's phase spans.
    (carrier,) = carriers
    carried = trace.get_current_span(otel_propagate.extract(carrier))
    assert carried.get_span_context().trace_id == trace_id
    (line,) = [
        line
        for line in structured.read_text().splitlines()
        if "generating" in line
    ]
    assert json.loads(line)["dd.trace_id"] == trace.format_trace_id(trace_id)
