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
"""Starts a request's server span and carries its trace to the model worker."""

from __future__ import annotations

import functools
import re
from collections.abc import AsyncIterable, AsyncIterator, Mapping
from contextvars import ContextVar
from enum import Enum
from typing import TypeVar

from max.pipelines.context import BaseContextType, TextContext
from max.serve.telemetry import common as telemetry
from max.serve.telemetry.common import (
    _capture_request_context,
    request_trace_ctx,
)
from opentelemetry import propagate as otel_propagate
from opentelemetry import trace as otel_trace
from opentelemetry.context import Context as OtelContext

_tracer = otel_trace.get_tracer("max.serve")

# The probes ``request._UNCOUNTED_PATH_RE`` leaves out of the request count,
# plus the routers' health routes that Kubernetes probes hit; tracing them
# would bury real requests in server spans.
_UNTRACED_PATH_RE = re.compile(
    r"/(?:health|version|ping|metrics(?:/.*)?"
    r"|v1/health|v2/health/(?:live|ready))"
)

_T = TypeVar("_T")

_phase_parent_ctx: ContextVar[OtelContext | None] = ContextVar(
    "max.serve.phase_parent_ctx", default=None
)
"""The OTel context whose span parents the worker's phase spans."""

TRACE_LEVEL_HEADER = "x-max-trace-level"
"""The request header naming a :class:`RequestTraceLevel`, and the
``trace_carrier`` key that carries it to the model worker."""


@functools.total_ordering
class RequestTraceLevel(Enum):
    """GPU trace depth one request asks for with the ``x-max-trace-level``
    header, honored only when ``kernel_trace_headers`` is on.

    Members compare in declaration order. A forward pass is traced at the
    highest level among its traced requests.
    """

    OFF = "off"
    """Not traced."""

    BATCH = "batch"
    """``max.batch`` and ``max.batch.gpu`` spans only."""

    OP = "op"
    """``batch``, plus one aggregate span per kernel name. Unlike
    ``KernelTraceLevel.OP``, no NVTX ranges."""

    KERNEL_SAMPLED = "kernel-sampled"
    """``batch``, plus kernel spans on 1 traced pass in 8."""

    KERNEL = "kernel"
    """``batch``, plus kernel spans on every traced pass."""

    FULL = "full"
    """``kernel``, plus register, occupancy and shared memory attributes."""

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, RequestTraceLevel):
            return NotImplemented
        members = list(RequestTraceLevel)
        return members.index(self) < members.index(other)

    @classmethod
    def parse(cls, value: str) -> RequestTraceLevel | None:
        """Parses a header value case-insensitively, treating ``_`` as ``-``.

        Returns None for an unknown value, which callers ignore.
        """
        try:
            return cls(value.strip().lower().replace("_", "-"))
        except ValueError:
            return None


request_trace_level: ContextVar[RequestTraceLevel | None] = ContextVar(
    "max.serve.request_trace_level", default=None
)
"""The current request's :data:`TRACE_LEVEL_HEADER` level, or None."""


def read_trace_level_header(headers: Mapping[str, str]) -> None:
    """Records the request's trace level header in :data:`request_trace_level`."""
    value = headers.get(TRACE_LEVEL_HEADER)
    request_trace_level.set(RequestTraceLevel.parse(value) if value else None)


def extract_inbound_context(headers: Mapping[str, str]) -> OtelContext:
    """Returns the trace context the client sent in ``headers``."""
    return otel_propagate.extract(headers)


def start_server_span(
    request_id: str,
    *,
    method: str,
    path: str,
    scheme: str,
    headers: Mapping[str, str],
) -> otel_trace.Span | None:
    """Starts a request's HTTP server span and captures its trace context.

    The span is named and attributed per the OTel HTTP server span
    conventions. Probes get none, so their trace context is the inbound one.
    """
    inbound = extract_inbound_context(headers)
    server_span = None
    if _UNTRACED_PATH_RE.fullmatch(path) is None:
        server_span = _tracer.start_span(
            method,
            context=inbound,
            kind=otel_trace.SpanKind.SERVER,
            attributes={
                "http.request.method": method,
                "url.path": path,
                "url.scheme": scheme,
            },
        )
    _capture_request_context(request_id, inbound, server_span)
    return server_span


async def end_span_after(
    body: AsyncIterable[_T], span: otel_trace.Span
) -> AsyncIterator[_T]:
    """Yields ``body``, then ends ``span``, also if the stream stops early."""
    try:
        async for chunk in body:
            yield chunk
    finally:
        span.end()


def record_server_span_status(
    span: otel_trace.Span, method: str, route: object, status_code: int
) -> None:
    """Names the server ``span`` after the matched ``route``, records status."""
    route_path = getattr(route, "path", None)
    if isinstance(route_path, str):
        span.update_name(f"{method} {route_path}")
        span.set_attribute("http.route", route_path)
    span.set_attribute("http.response.status_code", status_code)
    if status_code >= 500:
        span.set_status(otel_trace.StatusCode.ERROR)


def set_phase_parent(span: otel_trace.Span) -> None:
    """Makes ``span`` the parent of the phase spans the worker records.

    Call it in the handler's task before the pipeline runs: tasks the pipeline
    fans out to copy the context when they are created. Does nothing for the
    span the no-op tracer returns when tracing is off.
    """
    # Gates on validity, not recording: a sampled-out span is non-recording
    # but must still parent the phase spans.
    span_context = span.get_span_context()
    if not span_context.is_valid:
        return
    inbound = request_trace_ctx.get()
    # From OpenTelemetry 1.40 the no-op tracer returns the parent's span, which
    # the carrier falls back to anyway.
    if span_context == otel_trace.get_current_span(inbound).get_span_context():
        return
    _phase_parent_ctx.set(otel_trace.set_span_in_context(span, inbound))


def inject_trace_carrier(context: BaseContextType) -> None:
    """Serializes the parent of the worker's phase spans onto ``context``.

    ``context`` crosses into the model-worker process by value (pickled onto
    the request queue), so neither ContextVar can follow it. It injects
    ``_phase_parent_ctx`` into a string-dict carrier instead, which the
    scheduler re-``extract``s, falling back to ``request_trace_ctx`` when no
    handler set a phase parent. When per-request trace levels are permitted,
    it also records the request's level under :data:`TRACE_LEVEL_HEADER`. A
    no-op for non-``TextContext`` contexts, and when the chosen context has
    nothing to propagate and there is no level.
    """
    if not isinstance(context, TextContext):
        return
    parent = _phase_parent_ctx.get()
    if parent is None:
        parent = request_trace_ctx.get()
    carrier: dict[str, str] = {}
    otel_propagate.inject(carrier, context=parent)
    if telemetry._trace_level_header_enabled:
        level = request_trace_level.get()
        if level is not None:
            carrier[TRACE_LEVEL_HEADER] = level.value
    if carrier:
        context.trace_carrier = carrier
