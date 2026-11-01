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
"""Tests for cascade's API metrics.

Most of these assert what the middleware decided to record, by giving it
recording instruments in place of the real ones: the decisions are cascade's,
while aggregating and rendering them belong to OpenTelemetry and
``prometheus_client``. The few that do read rendered Prometheus text are the
ones where reaching that text is the thing under test.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from httpx import ASGITransport, AsyncClient
from max.experimental.cascade import telemetry
from max.experimental.cascade.serve import cascade_fastapi
from max.experimental.cascade.serve.cascade_fastapi import (
    CascadeFastAPI,
    _HttpMetricsMiddleware,
)
from prometheus_client import generate_latest
from prometheus_client.parser import text_string_to_metric_families
from starlette.types import ASGIApp, Message, Receive, Scope, Send


@pytest.fixture(scope="module", autouse=True)
def _telemetry() -> None:
    """Configure the process MeterProvider once for the whole module.

    ``configure_telemetry`` expects to be called once per process, so this
    mirrors how the frontend is set up: Views are fixed at construction, so
    every ladder cascade defines is registered here.
    """
    telemetry.configure_telemetry("cascade.test", "test-endpoint")


@pytest.fixture
def app() -> CascadeFastAPI:
    """An app exercising every response shape the middleware must handle."""
    built = CascadeFastAPI()

    @built.get("/ok")
    @telemetry.measured
    async def ok() -> str:
        return "ok"

    @built.get("/boom")
    @telemetry.measured
    async def boom() -> str:
        raise RuntimeError("handler failed")

    @built.get("/refused")
    @telemetry.measured
    async def refused() -> str:
        raise HTTPException(status_code=422, detail="bad request")

    @built.get("/stream")
    @telemetry.measured
    async def stream() -> StreamingResponse:
        async def body() -> AsyncIterator[bytes]:
            for i in range(4):
                yield f"chunk {i}\n".encode()

        return StreamingResponse(body(), media_type="text/plain")

    @built.get("/stream-boom")
    @telemetry.measured
    async def stream_boom() -> StreamingResponse:
        async def body() -> AsyncIterator[bytes]:
            yield b"first\n"
            raise RuntimeError("failed mid-stream")

        return StreamingResponse(body(), media_type="text/plain")

    return built


@dataclasses.dataclass
class _Recorded:
    """One ``record()`` call the middleware made."""

    instrument: str
    value: float
    attributes: dict[str, object]


class _Recorder:
    """Stands in for the histograms the middleware records into."""

    def __init__(self) -> None:
        self.calls: list[_Recorded] = []

    def ladder(self, instrument: str) -> SimpleNamespace:
        def record(value: float, attributes: dict[str, object]) -> None:
            self.calls.append(_Recorded(instrument, value, attributes))

        return SimpleNamespace(histogram=SimpleNamespace(record=record))

    def of(self, instrument: str) -> list[_Recorded]:
        """Every call against one instrument, in order."""
        return [c for c in self.calls if c.instrument == instrument]


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    """Record what the middleware would have written."""
    spy = _Recorder()
    for name in ("REQUEST_TIME", "TIME_TO_FIRST_CHUNK", "INTER_CHUNK_TIME"):
        monkeypatch.setattr(cascade_fastapi, name, spy.ladder(name.lower()))
    return spy


def _value(name: str, **labels: str) -> float:
    """Sums the rendered samples of ``name`` matching every given label."""
    return sum(
        sample.value
        for family in text_string_to_metric_families(
            generate_latest().decode("utf-8")
        )
        for sample in family.samples
        if sample.name == name
        and all(sample.labels.get(k) == v for k, v in labels.items())
    )


async def _get(app: CascadeFastAPI, path: str) -> int:
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        response = await client.get(path)
        return response.status_code


@pytest.mark.asyncio
async def test_every_histogram_renders_its_declared_ladder(
    app: CascadeFastAPI,
) -> None:
    """A ladder a route declares is the one Prometheus renders it with.

    A histogram whose ladder never reached the provider is silent: it renders
    with the OTEL SDK defaults, which are milliseconds, so every cascade
    latency lands in one bucket and no percentile means anything.
    """
    assert await _get(app, "/ok") == 200
    assert await _get(app, "/stream") == 200

    rendered = {
        family.name: sorted(
            {
                float(sample.labels["le"])
                for sample in family.samples
                if sample.name.endswith("_bucket")
                and sample.labels["le"] != "+Inf"
            }
        )
        for family in text_string_to_metric_families(
            generate_latest().decode("utf-8")
        )
        if family.name.startswith("cascade_")
    }
    declared = {
        ladder.name.replace(".", "_") + "_seconds": sorted(ladder.buckets)
        for ladder in telemetry.LADDERS
    }

    histograms = {name: edges for name, edges in rendered.items() if edges}
    assert histograms, "no cascade histogram reached the exposition"
    for name, edges in histograms.items():
        assert name in declared, f"{name} declares no buckets"
        assert edges == declared[name], name


@pytest.mark.asyncio
async def test_only_a_marked_endpoint_is_timed(
    app: CascadeFastAPI, recorder: _Recorder
) -> None:
    """A route with no mark is timed by nothing rather than by a catch-all.

    Both ways of having no mark are covered: a routed handler that was never
    decorated, and a 404 that routed to nothing at all. A health probe or a
    scanner answers on its own scale, and mixing either into an inference
    route's distribution gives a p50 belonging to neither.
    """

    @app.get("/health")
    async def health() -> str:
        return "OK"

    assert await _get(app, "/health") == 200
    for path in ("/nope", "/also-nope"):
        assert await _get(app, path) == 404

    assert recorder.calls == []


@pytest.mark.asyncio
async def test_a_request_is_timed_against_its_route_and_outcome(
    app: CascadeFastAPI, recorder: _Recorder
) -> None:
    assert await _get(app, "/ok") == 200

    (call,) = recorder.of("request_time")
    assert call.attributes == {
        "route": "/ok",
        "outcome": "ok",
        "streaming": False,
    }
    assert call.value > 0


@pytest.mark.asyncio
async def test_client_and_server_errors_are_told_apart(
    app: CascadeFastAPI, recorder: _Recorder
) -> None:
    """A rising server_error rate is ours; a rising client_error rate is the
    caller's, and an error-rate alert usually wants only the first."""
    await _get(app, "/boom")
    assert await _get(app, "/refused") == 422

    assert [
        (c.attributes["route"], c.attributes["outcome"])
        for c in recorder.of("request_time")
    ] == [("/boom", "server_error"), ("/refused", "client_error")]


@pytest.mark.asyncio
async def test_a_stream_records_one_first_chunk_and_a_gap_per_chunk_after(
    app: CascadeFastAPI, recorder: _Recorder
) -> None:
    """/stream yields four chunks, so the first is the time to first chunk and
    the three that follow another are gaps, all on request_time's clock."""
    assert await _get(app, "/stream") == 200

    (timed,) = recorder.of("request_time")
    assert timed.attributes["streaming"] is True
    assert len(recorder.of("time_to_first_chunk")) == 1
    assert len(recorder.of("inter_chunk_time")) == 3
    assert all(
        c.attributes == {"route": "/stream"}
        for c in recorder.of("inter_chunk_time")
    )


@pytest.mark.asyncio
async def test_a_single_shot_response_has_a_duration_and_nothing_else(
    app: CascadeFastAPI, recorder: _Recorder
) -> None:
    """A response sent in one piece records no TTFC or ICT.

    Those distributions answer "how long until the caller saw something" and
    "how steadily did it arrive"; a whole request's duration answers neither.
    """
    assert await _get(app, "/ok") == 200

    assert len(recorder.of("request_time")) == 1
    assert recorder.of("time_to_first_chunk") == []
    assert recorder.of("inter_chunk_time") == []


@pytest.mark.asyncio
async def test_failure_after_the_first_chunk_is_an_error_not_a_success(
    app: CascadeFastAPI, recorder: _Recorder
) -> None:
    """A streamed response is committed to 200 before its body is produced.

    The status code therefore cannot say whether the request succeeded, which
    is why ``outcome`` is recorded from the response lifecycle instead.
    """
    await _get(app, "/stream-boom")

    (call,) = recorder.of("request_time")
    assert call.attributes["route"] == "/stream-boom"
    assert call.attributes["outcome"] == "server_error"


@pytest.mark.asyncio
async def test_a_route_behind_a_router_prefix_is_labelled() -> None:
    """The responses route is registered on a router carrying ``prefix="/v1"``.

    The mark rides on the endpoint, so a prefix cannot separate a route from
    its instruments; what the prefix does reach is the ``route`` label, which
    is the full template.
    """
    built = CascadeFastAPI()
    router = APIRouter(prefix="/v1")

    @router.get("/responses")
    @telemetry.measured
    async def responses() -> str:
        return "ok"

    built.include_router(router)

    name = "cascade_http_request_time_seconds_count"
    before = _value(name, route="/v1/responses", outcome="ok")
    assert await _get(built, "/v1/responses") == 200

    assert _value(name, route="/v1/responses", outcome="ok") == before + 1


async def _drive(app: ASGIApp) -> None:
    """Run ``app`` through the middleware with a marked, routed scope."""
    route = "/v1/chat/completions"

    # Placeholder: the middleware passes receive straight through and these
    # stub apps never read a body, but the ASGI signature requires it.
    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        return None

    await _HttpMetricsMiddleware(app)(
        {
            "type": "http",
            "path": route,
            "method": "GET",
            "headers": [],
            # The stub app never routes, so stand in for what starlette sets.
            "route": SimpleNamespace(path=route),
            "endpoint": telemetry.measured(lambda: None),
        },
        receive,
        send,
    )


@pytest.mark.asyncio
async def test_a_body_that_never_terminates_is_cancelled_not_ok(
    recorder: _Recorder,
) -> None:
    """A client that goes away mid-stream must not read as success.

    Asserted on the ASGI state a disconnect leaves rather than by racing a
    real one: starlette runs a streaming response against the disconnect and
    cancels the stream from under it, then returns normally, so the app call
    completes with the body unterminated and nothing raised.
    """

    async def truncated(scope: Scope, receive: Receive, send: Send) -> None:
        await send(
            {"type": "http.response.start", "status": 200, "headers": []}
        )
        await send(
            {
                "type": "http.response.body",
                "body": b"partial",
                "more_body": True,
            }
        )

    await _drive(truncated)

    (call,) = recorder.of("request_time")
    assert call.attributes["outcome"] == "cancelled"


@pytest.mark.asyncio
async def test_a_content_bearing_last_message_is_a_gap(
    recorder: _Recorder,
) -> None:
    """A stream whose final message carries content still yields its gap.

    Starlette closes a stream with an empty body message, but an ASGI app is
    free to put the last bytes in the message that clears ``more_body``.
    """

    async def two_chunks(scope: Scope, receive: Receive, send: Send) -> None:
        await send(
            {"type": "http.response.start", "status": 200, "headers": []}
        )
        await send(
            {"type": "http.response.body", "body": b"a", "more_body": True}
        )
        await send({"type": "http.response.body", "body": b"b"})

    await _drive(two_chunks)

    assert len(recorder.of("time_to_first_chunk")) == 1
    assert len(recorder.of("inter_chunk_time")) == 1


@pytest.mark.asyncio
async def test_sse_keep_alive_frames_are_not_chunks(
    recorder: _Recorder,
) -> None:
    async def pinged(scope: Scope, receive: Receive, send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream")],
            }
        )
        for body in (b": ping\r\n\r\n", b"data: a\r\n\r\n", b": ping\r\n\r\n"):
            await send(
                {"type": "http.response.body", "body": body, "more_body": True}
            )
        await send({"type": "http.response.body", "body": b"data: b\r\n\r\n"})

    await _drive(pinged)

    assert len(recorder.of("time_to_first_chunk")) == 1
    assert len(recorder.of("inter_chunk_time")) == 1


@pytest.mark.asyncio
async def test_only_a_marked_endpoint_moves_the_inflight_gauge(
    app: CascadeFastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    adds: list[int] = []
    monkeypatch.setattr(
        cascade_fastapi,
        "HTTP_INFLIGHT",
        SimpleNamespace(add=adds.append),
    )

    @app.get("/unmarked")
    async def unmarked() -> str:
        return "ok"

    assert await _get(app, "/unmarked") == 200
    assert await _get(app, "/missing") == 404
    assert adds == []

    assert await _get(app, "/ok") == 200
    assert await _get(app, "/stream") == 200
    assert adds == [1, -1, 1, -1]


@pytest.mark.asyncio
async def test_a_request_is_inflight_while_its_handler_runs(
    app: CascadeFastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    adds: list[int] = []
    monkeypatch.setattr(
        cascade_fastapi, "HTTP_INFLIGHT", SimpleNamespace(add=adds.append)
    )
    entered, release = asyncio.Event(), asyncio.Event()

    @app.post("/slow")
    @telemetry.measured
    async def slow(body: dict[str, str]) -> str:
        entered.set()
        await release.wait()
        return "ok"

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        request = asyncio.create_task(client.post("/slow", json={}))
        await entered.wait()
        assert sum(adds) == 1
        release.set()
        assert (await request).status_code == 200
    assert sum(adds) == 0
