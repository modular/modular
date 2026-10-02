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
"""Carries a request's trace context across to the model worker."""

from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar

from max.pipelines.context import BaseContextType, TextContext
from max.serve.telemetry.common import request_trace_ctx
from opentelemetry import propagate as otel_propagate
from opentelemetry import trace as otel_trace
from opentelemetry.context import Context as OtelContext

_phase_parent_ctx: ContextVar[OtelContext | None] = ContextVar(
    "max.serve.phase_parent_ctx", default=None
)
"""The OTel context whose span parents the worker's phase spans."""


def extract_inbound_context(headers: Mapping[str, str]) -> OtelContext:
    """Returns the trace context the client sent in ``headers``."""
    return otel_propagate.extract(headers)


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
    handler set a phase parent. A no-op for non-``TextContext`` contexts, and
    when the chosen context has nothing to propagate.
    """
    if not isinstance(context, TextContext):
        return
    parent = _phase_parent_ctx.get()
    if parent is None:
        parent = request_trace_ctx.get()
    carrier: dict[str, str] = {}
    otel_propagate.inject(carrier, context=parent)
    if carrier:
        context.trace_carrier = carrier
