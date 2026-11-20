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
"""Holds the correlation IDs that structured logs and spans read."""

from __future__ import annotations

from contextvars import ContextVar

from opentelemetry import trace
from opentelemetry.context import Context as OtelContext

request_trace_ctx: ContextVar[OtelContext | None] = ContextVar(
    "max.serve.request_trace_ctx", default=None
)
"""The OTel context holding the current request's HTTP server span (or None),
which continues the inbound W3C traceparent where there is one. For a probe it
is the inbound context itself, and with tracing off it is None. Set by the
``request_session`` middleware, so it is already populated by the time
TextContext is constructed in llm.py — ``call_next`` runs the route in a child
task, which inherits a copy of the context set here."""

request_id_ctx: ContextVar[str | None] = ContextVar(
    "max.serve.request_id_ctx", default=None
)
"""The HTTP request's ID. ``/v1/completions`` and ``/v1/embeddings`` take one
pipeline request per prompt, whose ``RequestID`` and ``max.request.id`` span
attribute append ``_<index>`` to this."""

batch_id_ctx: ContextVar[int | None] = ContextVar(
    "max.serve.batch_id_ctx", default=None
)
"""The ID of the forward pass the text-generation scheduler is executing or
logging batch_metrics for, or None. It is set only when tracing is enabled; the
API process and the disaggregated workers never set it."""


def tracing_enabled() -> bool:
    """Whether a real TracerProvider is installed, vs. the OTel no-op default."""
    return not isinstance(
        trace.get_tracer_provider(), trace.ProxyTracerProvider
    )


def capture_request_context(
    request_id: str, inbound: OtelContext, server_span: trace.Span | None
) -> None:
    """Sets the ambient correlation IDs for the request being handled.

    ``request_session`` calls this, through ``start_server_span``, before
    ``call_next``, so the downstream task's context copy carries them onto
    every route's records. The span is not made current: the OTLP log handler
    would stamp its trace ID, the caller's where a traceparent arrived, onto
    records sent to Modular.
    """
    request_id_ctx.set(request_id)
    request_trace_ctx.set(
        trace.set_span_in_context(server_span, inbound)
        if server_span is not None
        else inbound
    )


def correlation_fields() -> dict[str, str | int]:
    """Returns the ambient request and batch correlation IDs for a log line.

    ``dd.trace_id`` is the key Datadog's correlator reads, as 32 hex digits.
    No span ID: for a probe, ``request_trace_ctx`` holds the caller's context,
    so one would attach MAX's logs to the caller's span.
    """
    fields: dict[str, str | int] = {}
    request_id = request_id_ctx.get()
    if request_id is not None:
        fields["request_id"] = request_id
    batch_id = batch_id_ctx.get()
    if batch_id is not None:
        fields["batch_id"] = batch_id
    context = request_trace_ctx.get()
    if context is not None:
        span_context = trace.get_current_span(context).get_span_context()
        if span_context.is_valid:
            fields["dd.trace_id"] = trace.format_trace_id(span_context.trace_id)
    return fields
