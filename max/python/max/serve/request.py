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


import logging
import re
import uuid
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from max.serve._error_envelope import openai_error_body
from max.serve.router._disconnect import (
    CLIENT_CLOSED_REQUEST,
    ClientDisconnected,
)
from max.serve.telemetry import _log_context, _tracing
from max.serve.telemetry._log_context import request_id_ctx
from max.serve.telemetry._trace_context import (
    end_span_after,
    read_trace_level_header,
    record_server_span_status,
    start_server_span,
)
from max.serve.telemetry.metrics import METRICS
from max.serve.telemetry.stopwatch import StopWatch
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger("max.serve")

# Liveness/observability endpoints are hit by periodic probes and scrapers.
# Counting them would swamp the ``maxserve.request_count`` metric (which tracks
# API request volume), so they are excluded. ``/metrics`` is a mounted sub-app,
# so its subpaths are excluded too.
_UNCOUNTED_PATH_RE = re.compile(r"/(?:health|version|ping|metrics(?:/.*)?)")


def _should_count_request(path: str) -> bool:
    return _UNCOUNTED_PATH_RE.fullmatch(path) is None


class _MidResponseDisconnectCounter:
    """Counts a disconnect between a route's response start and its last chunk."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(
        self, scope: Scope, receive: Receive, send: Send
    ) -> None:
        if scope["type"] != "http" or not _should_count_request(scope["path"]):
            await self.app(scope, receive, send)
            return
        has_started = False
        is_complete = False
        is_counted = False

        async def watched_send(message: Message) -> None:
            nonlocal has_started, is_complete
            if message["type"] == "http.response.start":
                has_started = True
            elif message["type"] == "http.response.body" and not message.get(
                "more_body", False
            ):
                is_complete = True
            await send(message)

        async def watched_receive() -> Message:
            nonlocal is_counted
            message = await receive()
            if (
                message["type"] == "http.disconnect"
                and has_started
                and not is_complete
                and not is_counted
            ):
                is_counted = True
                METRICS.client_disconnect_cancellation("during_response")
            return message

        await self.app(scope, watched_receive, watched_send)


def register_request(app: FastAPI, *, structured_logging: bool = False) -> None:
    # Read once: the server configures tracing before it builds the app.
    # Structured logs read the request ID. Only spans, directly or through
    # the worker's trace carrier, and dd.trace_id read the trace context.
    tracing = _log_context.tracing_enabled()
    read_trace_levels = _tracing.trace_level_header_honored()
    # Added before request_session so it sits inside it, where it sees the
    # route's messages rather than the middleware's re-sent copies.
    app.add_middleware(_MidResponseDisconnectCounter)

    @app.middleware("http")
    async def request_session(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = uuid.uuid4().hex
        request.state.request_id = request_id
        request.state.request_timer = StopWatch()
        server_span = None
        if tracing:
            server_span = start_server_span(
                request_id,
                method=request.method,
                path=request.url.path,
                scheme=request.url.scheme,
                headers=request.headers,
            )
            if read_trace_levels:
                read_trace_level_header(request.headers)
        elif structured_logging:
            request_id_ctx.set(request_id)
        # Record the request against the final HTTP status code. This is the
        # authoritative place to label ``maxserve.request_count`` with the
        # return code: it sees the status of every request, including failures
        # (e.g. a bad image URL) that are rejected before reaching the response
        # generator, and it reflects the code actually sent to the client
        # rather than a value guessed mid-stream.
        status_code = 500
        span_ends_with_body = False
        should_count = _should_count_request(request.url.path)
        try:
            response: Response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            # Ending with the headers would make a stream's span cover only
            # the time to its first byte.
            if server_span is not None and hasattr(response, "body_iterator"):
                response.body_iterator = end_span_after(
                    response.body_iterator, server_span
                )
                span_ends_with_body = True
            return response
        except HTTPException as e:
            status_code = e.status_code
            raise  # already wrapped
        except ClientDisconnected:
            status_code = CLIENT_CLOSED_REQUEST
            if should_count:
                METRICS.client_disconnect_cancellation("before_response")
            logger.info("Client disconnected; cancelled request %s", request_id)
            return Response(
                status_code=CLIENT_CLOSED_REQUEST,
                headers={"X-Request-ID": request_id},
            )
        except Exception:
            # Returned rather than raised. This middleware runs outside
            # Starlette's ExceptionMiddleware, so a raise here never reaches the
            # app's HTTPException handler; it lands in ServerErrorMiddleware,
            # which answers with a bare text/plain 500 and then re-raises,
            # prompting uvicorn to close the connection out from under a client
            # that was owed a response. ``status_code`` is already 500.
            logger.exception("Exception in request session : %s", request_id)
            return JSONResponse(
                status_code=500,
                content=openai_error_body(500, "Internal server error."),
                headers={"X-Request-ID": request_id},
            )
        finally:
            if should_count:
                METRICS.request_count(status_code, request.url.path)
            if server_span is not None:
                record_server_span_status(
                    server_span,
                    request.method,
                    request.scope.get("route"),
                    status_code,
                )
                if not span_ends_with_body:
                    server_span.end()
                logger.info(
                    "Request %s returned status %d", request_id, status_code
                )
