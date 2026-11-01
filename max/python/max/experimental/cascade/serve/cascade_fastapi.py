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
"""FastAPI wrapper used by cascade route adapters."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import (
    AsyncIterable,
    Iterable,
)

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response, StreamingResponse
from max.experimental.cascade.telemetry import (
    HTTP_INFLIGHT,
    INTER_CHUNK_TIME,
    METRICS_PATH,
    REQUEST_TIME,
    TIME_TO_FIRST_CHUNK,
    Outcome,
    is_measured,
)
from starlette.requests import ClientDisconnect
from starlette.types import ASGIApp, Message, Receive, Scope, Send

_logger = logging.getLogger(__name__)


class CascadeFastAPI(FastAPI):
    """FastAPI subclass that centralizes the current web-layer integration.

    This consolidates direct usage of both FastAPI and uvicorn in one place
    so the cascade serving layer can be swapped to another framework later
    without rewriting the pipeline route adapters and server entrypoints.
    """

    def __init__(self) -> None:
        super().__init__()
        self.add_exception_handler(
            RequestValidationError, self._log_validation_error
        )
        self.add_middleware(_HttpMetricsMiddleware)

    def streaming_response(
        self,
        content: (
            AsyncIterable[str | bytes | memoryview[int]]
            | Iterable[str | bytes | memoryview[int]]
        ),
        *,
        media_type: str | None = None,
    ) -> StreamingResponse:
        """Create a FastAPI-compatible streaming response."""
        return StreamingResponse(content, media_type=media_type)

    async def _log_validation_error(
        self,
        request: Request,
        exc: Exception,
    ) -> Response:
        """Log request bodies that fail FastAPI validation."""
        assert isinstance(exc, RequestValidationError)
        body_bytes = await request.body()
        body_text = body_bytes.decode("utf-8", errors="replace")
        _logger.warning(
            "Request validation failed for %s %s: errors=%s payload=%s",
            request.method,
            request.url.path,
            exc.errors(),
            body_text,
        )
        return await request_validation_exception_handler(request, exc)

    async def serve(self, host: str, port: int) -> None:
        """Serve this FastAPI app."""
        config = uvicorn.Config(
            self,
            host=host,
            port=port,
            lifespan="off",
            access_log=False,
        )
        server = uvicorn.Server(config)
        await server.serve()


class _HttpMetricsMiddleware:
    """Records timing, size and outcome for every HTTP request.

    The outermost measuring layer, and blind to meaning with one exception: a
    chunk here is one ``http.response.body`` message, except an SSE comment
    frame (a keep-alive ping), which carries nothing a caller sees. Everything
    is measured on one clock, so the latencies subtract.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(
        self, scope: Scope, receive: Receive, send: Send
    ) -> None:
        if scope["type"] != "http" or scope["path"] == METRICS_PATH:
            await self.app(scope, receive, send)
            return

        started = time.monotonic()
        outcome = Outcome.OK
        status_code = 0
        inflight = False
        sse = False
        body_finished = False
        streamed = False
        # Time to each body message from the one before it
        gaps: list[float] = []
        last = started

        def mark_inflight() -> None:
            nonlocal inflight
            if not inflight and is_measured(scope.get("endpoint")):
                HTTP_INFLIGHT.add(1)
                inflight = True

        async def receive_wrapper() -> Message:
            mark_inflight()
            return await receive()

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code, body_finished, last, streamed, sse
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                sse = any(
                    name.lower() == b"content-type"
                    and value.startswith(b"text/event-stream")
                    for name, value in message.get("headers", [])
                )
                mark_inflight()
            elif message["type"] == "http.response.body":
                streaming = message.get("more_body", False)
                if streaming:
                    streamed = True
                else:
                    body_finished = True
                # ``more_body`` is unset both on (1) a single-shot body, whose
                # timing is the whole response and belongs in no distribution
                # here, and (2) on a stream's last message. A preceding chunk
                # distinguishes these two.
                body = message.get("body", b"")
                if (
                    body
                    and not (sse and body.startswith(b":"))
                    and (streaming or gaps)
                ):
                    now = time.monotonic()
                    gaps.append(now - last)
                    last = now
            await send(message)

        try:
            await self.app(scope, receive_wrapper, send_wrapper)
            if not body_finished:
                # Nothing raised, yet the body never terminated: starlette
                # cancels a streaming response on disconnect and returns
                # normally, so exceptions alone miss the common case.
                outcome = Outcome.CANCELLED
            elif status_code >= 500:
                outcome = Outcome.SERVER_ERROR
            elif status_code >= 400:
                outcome = Outcome.CLIENT_ERROR
        except (asyncio.CancelledError, GeneratorExit, ClientDisconnect):
            # Newer ASGI versions surface a disconnect as ClientDisconnect
            # rather than by cancelling the task.
            outcome = Outcome.CANCELLED
            raise
        except Exception:
            outcome = Outcome.SERVER_ERROR
            raise
        finally:
            if is_measured(scope.get("endpoint")):
                route = str(getattr(scope.get("route"), "path", ""))
                REQUEST_TIME.histogram.record(
                    time.monotonic() - started,
                    {
                        "route": route,
                        "outcome": outcome.value,
                        "streaming": streamed,
                    },
                )
                # Only a stream reaches here with gaps, so these two carry no
                # ``streaming`` attribute of their own.
                if gaps:
                    TIME_TO_FIRST_CHUNK.histogram.record(
                        gaps[0], {"route": route}
                    )
                    for gap in gaps[1:]:
                        INTER_CHUNK_TIME.histogram.record(gap, {"route": route})
            if inflight:
                HTTP_INFLIGHT.add(-1)
