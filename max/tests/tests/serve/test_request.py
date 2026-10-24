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
"""Tests for the request-session middleware, especially that the
``maxserve.request_count`` metric is labeled with the real HTTP status code."""

import asyncio
import os
import socket
import subprocess
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest import mock

import pytest
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.testclient import TestClient
from max.serve import request as request_module
from max.serve.request import _should_count_request, register_request
from max.serve.router._disconnect import ClientDisconnected
from sse_starlette.sse import AppStatus, EventSourceResponse
from tests.serve.asgi_disconnect import disconnect_after_first_chunk


def _make_app() -> FastAPI:
    app = FastAPI()

    @app.get("/v1/ok")
    async def ok() -> JSONResponse:
        return JSONResponse(status_code=200, content={"ok": True})

    @app.get("/v1/bad-image")
    async def bad_image() -> Response:
        # Mirrors an image-fetch failure surfaced as an HTTP 400 by the route.
        raise HTTPException(status_code=400, detail="Failed to fetch image")

    @app.get("/v1/crash")
    async def crash() -> Response:
        raise RuntimeError("unhandled")

    @app.get("/v1/hung-up")
    async def hung_up() -> Response:
        raise ClientDisconnected

    @app.get("/v1/stream")
    async def stream() -> StreamingResponse:
        async def body() -> AsyncIterator[bytes]:
            yield b"a"
            yield b"b"

        return StreamingResponse(body())

    @app.get("/health")
    async def health() -> Response:
        return Response(status_code=200)

    register_request(app)
    return app


@pytest.fixture
def request_count(monkeypatch: pytest.MonkeyPatch) -> mock.Mock:
    counter = mock.Mock()
    monkeypatch.setattr(request_module.METRICS, "request_count", counter)
    return counter


@pytest.fixture
def disconnect_count(monkeypatch: pytest.MonkeyPatch) -> mock.Mock:
    counter = mock.Mock()
    monkeypatch.setattr(
        request_module.METRICS, "client_disconnect_cancellation", counter
    )
    return counter


def test_records_success_code(request_count: mock.Mock) -> None:
    with TestClient(_make_app()) as client:
        response = client.get("/v1/ok")
    assert response.status_code == 200
    request_count.assert_called_once_with(200, "/v1/ok")


def test_records_error_code(request_count: mock.Mock) -> None:
    # A non-200 response (e.g. a failed image fetch) must be counted with its
    # real status code, not 200.
    with TestClient(_make_app()) as client:
        response = client.get("/v1/bad-image")
    assert response.status_code == 400
    request_count.assert_called_once_with(400, "/v1/bad-image")


def test_records_500_for_unhandled_exception(request_count: mock.Mock) -> None:
    # ``raise_server_exceptions`` stays at its default: the middleware answers
    # the request itself, so the exception never reaches the transport.
    with TestClient(_make_app()) as client:
        response = client.get("/v1/crash")
    assert response.status_code == 500
    request_count.assert_called_once_with(500, "/v1/crash")


def test_unhandled_exception_returns_openai_error_envelope() -> None:
    """An unhandled route exception is answered in the OpenAI error shape.

    Raising from this middleware instead bypasses the app's ``HTTPException``
    handler -- it runs outside Starlette's exception middleware -- and reaches
    ``ServerErrorMiddleware``, which replies in ``text/plain`` and re-raises,
    closing the connection under a client that was owed a response.
    """
    with TestClient(_make_app()) as client:
        response = client.get("/v1/crash")

    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/json")
    assert response.headers["X-Request-ID"]
    assert response.json() == {
        "error": {
            "code": "500",
            "message": "Internal server error.",
            "param": "",
            "type": "api_error",
        }
    }


def test_skips_probe_endpoints(request_count: mock.Mock) -> None:
    with TestClient(_make_app()) as client:
        assert client.get("/health").status_code == 200
    request_count.assert_not_called()


def test_counts_disconnect_before_response(
    request_count: mock.Mock, disconnect_count: mock.Mock
) -> None:
    with TestClient(_make_app()) as client:
        response = client.get("/v1/hung-up")
    assert response.status_code == 499
    request_count.assert_called_once_with(499, "/v1/hung-up")
    disconnect_count.assert_called_once_with("before_response")


def test_completed_stream_is_not_a_disconnect(
    request_count: mock.Mock, disconnect_count: mock.Mock
) -> None:
    with TestClient(_make_app()) as client:
        response = client.get("/v1/stream")
    assert response.content == b"ab"
    request_count.assert_called_once_with(200, "/v1/stream")
    disconnect_count.assert_not_called()


@pytest.mark.asyncio
async def test_probe_endpoints_are_not_disconnects(
    request_count: mock.Mock, disconnect_count: mock.Mock
) -> None:
    app = FastAPI()
    register_request(app)

    @app.get("/health")
    async def health() -> StreamingResponse:
        async def body() -> AsyncIterator[bytes]:
            yield b"a"
            await asyncio.Event().wait()

        return StreamingResponse(body())

    await disconnect_after_first_chunk(app, "/health")

    disconnect_count.assert_not_called()


@pytest.mark.asyncio
async def test_counts_disconnect_during_response(
    request_count: mock.Mock, disconnect_count: mock.Mock
) -> None:
    # The body never finishes on its own, so only the disconnect can end it.
    app = FastAPI()
    register_request(app)

    @app.get("/v1/stream")
    async def stream() -> StreamingResponse:
        async def body() -> AsyncIterator[bytes]:
            yield b"a"
            await asyncio.Event().wait()
            yield b"never"

        return StreamingResponse(body())

    await disconnect_after_first_chunk(app, "/v1/stream")

    request_count.assert_called_once_with(200, "/v1/stream")
    disconnect_count.assert_called_once_with("during_response")


def _make_streaming_app() -> FastAPI:
    app = FastAPI()
    register_request(app)

    @app.get("/v1/endless")
    async def endless() -> StreamingResponse:
        async def body() -> AsyncIterator[bytes]:
            yield b"a"
            await asyncio.Event().wait()

        return StreamingResponse(body())

    @app.get("/v1/endless-sse")
    async def endless_sse() -> EventSourceResponse:
        async def events() -> AsyncIterator[str]:
            yield "a"
            await asyncio.Event().wait()

        return EventSourceResponse(events())

    @app.get("/v1/finite")
    async def finite() -> StreamingResponse:
        async def body() -> AsyncIterator[bytes]:
            yield b"a"
            yield b"b"

        return StreamingResponse(body())

    @app.get("/v1/finite-sse")
    async def finite_sse() -> EventSourceResponse:
        async def events() -> AsyncIterator[str]:
            yield "a"
            yield "[DONE]"

        return EventSourceResponse(events())

    return app


@pytest.fixture(autouse=True)
def fresh_sse_exit_event(monkeypatch: pytest.MonkeyPatch) -> None:
    # sse-starlette binds a module-global Event to the first loop that streams;
    # each asyncio test runs on its own loop.
    monkeypatch.setattr(AppStatus, "should_exit_event", None)


@asynccontextmanager
async def _serving(app: FastAPI) -> AsyncIterator[int]:
    """Runs ``app`` under uvicorn on a free port and yields the port."""
    server = uvicorn.Server(
        uvicorn.Config(
            app, host="127.0.0.1", port=0, lifespan="off", log_level="warning"
        )
    )
    serve_task = asyncio.create_task(server.serve())

    async def wait_until_started() -> None:
        while not server.started:
            assert not serve_task.done(), "uvicorn exited during startup"
            await asyncio.sleep(0.01)

    await asyncio.wait_for(wait_until_started(), timeout=30)
    try:
        yield server.servers[0].sockets[0].getsockname()[1]
    finally:
        server.should_exit = True
        await asyncio.wait_for(serve_task, timeout=30)


def _get_blocking(port: int, path: str, until: bytes) -> None:
    with socket.create_connection(("127.0.0.1", port), timeout=30) as client:
        client.sendall(f"GET {path} HTTP/1.1\r\nHost: test\r\n\r\n".encode())
        received = b""
        while until not in received:
            chunk = client.recv(4096)
            assert chunk, f"connection closed before {until!r} arrived"
            received += chunk


async def _get(port: int, path: str, *, until: bytes) -> None:
    """Sends a GET, reads until ``until`` arrives, then closes the socket.

    The client runs on its own thread so its close races the server the way
    a remote client's does, rather than waiting for the server's loop to
    yield.
    """
    await asyncio.to_thread(_get_blocking, port, path, until)


async def _wait_for_call(counter: mock.Mock) -> None:
    async def poll() -> None:
        while not counter.called:
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout=30)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,first_chunk",
    [("/v1/endless", b"\r\na\r\n"), ("/v1/endless-sse", b"data: a")],
)
async def test_uvicorn_counts_disconnect_during_response(
    request_count: mock.Mock,
    disconnect_count: mock.Mock,
    path: str,
    first_chunk: bytes,
) -> None:
    # Drives a real server: the detection relies on how uvicorn reports a
    # disconnect, which a hand-written receive only imitates.
    async with _serving(_make_streaming_app()) as port:
        await _get(port, path, until=first_chunk)
        await _wait_for_call(disconnect_count)
    request_count.assert_called_once_with(200, path)
    disconnect_count.assert_called_once_with("during_response")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,last_chunk",
    [("/v1/finite", b"\r\nb\r\n"), ("/v1/finite-sse", b"data: [DONE]")],
)
async def test_uvicorn_close_on_last_chunk_is_not_a_disconnect(
    request_count: mock.Mock,
    disconnect_count: mock.Mock,
    path: str,
    last_chunk: bytes,
) -> None:
    # Clients commonly close on the last event, before the terminating
    # chunk arrives; every byte the route sent was still delivered. Repeated
    # because the close races the server finishing the response.
    async with _serving(_make_streaming_app()) as port:
        for _ in range(10):
            await _get(port, path, until=last_chunk)
    # Read after shutdown, which waits for every response to finish.
    disconnect_count.assert_not_called()


@pytest.mark.asyncio
async def test_uvicorn_completed_stream_is_not_a_disconnect(
    request_count: mock.Mock, disconnect_count: mock.Mock
) -> None:
    async with _serving(_make_streaming_app()) as port:
        await _get(port, "/v1/finite", until=b"0\r\n\r\n")
    request_count.assert_called_once_with(200, "/v1/finite")
    disconnect_count.assert_not_called()


@pytest.mark.parametrize(
    "path,expected",
    [
        ("/v1/chat/completions", True),
        ("/v1/embeddings", True),
        ("/invocations", True),
        ("/health", False),
        ("/version", False),
        ("/ping", False),
        ("/metrics", False),
        ("/metrics/foo", False),
    ],
)
def test_should_count_request(path: str, expected: bool) -> None:
    assert _should_count_request(path) is expected


def test_telemetry_common_does_not_import_propagate() -> None:
    # The metrics worker imports common.py but not the routes, and importing
    # opentelemetry.propagate scans entry points, so the extract lives in
    # _trace_context, which the metrics worker doesn't import.
    code = (
        "import sys, max.serve.telemetry._tracing; "
        "assert 'opentelemetry.propagate' not in sys.modules"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    subprocess.run([sys.executable, "-c", code], env=env, check=True)
