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
"""The scheduler's phase spans must parent under ``max.request``.

The two halves live in different processes, so they are asserted separately
and composed.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import nullcontext
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import opentelemetry.trace as otel_trace
import pytest
from max.pipelines.context import GenerationStatus, TextContext, TokenBuffer
from max.pipelines.modeling.types import RequestID
from max.serve.pipelines.llm import TokenGeneratorOutput
from max.serve.router.openai_routes import (
    OpenAIChatResponseGenerator,
    OpenAICompletionResponseGenerator,
    _tracer,
)
from max.serve.scheduler.text_generation_scheduler import (
    _parent_trace_context,
)
from max.serve.telemetry._log_context import request_trace_ctx
from max.serve.telemetry._trace_context import (
    _phase_parent_ctx,
    inject_trace_carrier,
    set_phase_parent,
)
from opentelemetry import propagate as otel_propagate
from opentelemetry.context import Context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF


@pytest.fixture
def finished() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def tracer(finished: InMemorySpanExporter) -> otel_trace.Tracer:
    """A tracer from a local SDK provider, so spans are recording.

    The global provider is set-once and would leak across the target.
    """
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(finished))
    return provider.get_tracer("test")


@pytest.fixture(autouse=True)
def reset_request_context() -> Iterator[None]:
    request_token = request_trace_ctx.set(None)
    parent_token = _phase_parent_ctx.set(None)
    try:
        yield
    finally:
        request_trace_ctx.reset(request_token)
        _phase_parent_ctx.reset(parent_token)


def _mock_request() -> Mock:
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


def _text_context() -> TextContext:
    return TextContext(
        request_id=RequestID(),
        max_length=16,
        tokens=TokenBuffer(np.ones(4, dtype=np.int64)),
    )


def test_carrier_falls_back_to_the_request_trace_context(
    tracer: otel_trace.Tracer,
) -> None:
    span = tracer.start_span("caller")
    request_trace_ctx.set(otel_trace.set_span_in_context(span))

    context = _text_context()
    inject_trace_carrier(context)

    assert context.trace_carrier is not None
    carried = otel_trace.get_current_span(
        _parent_trace_context(context)
    ).get_span_context()
    assert carried.span_id == span.get_span_context().span_id
    assert carried.trace_id == span.get_span_context().trace_id


def test_a_sampled_out_request_span_still_parents_the_phase_spans() -> None:
    provider = TracerProvider(sampler=ALWAYS_OFF)
    request_span = provider.get_tracer("test").start_span("max.request")
    assert not request_span.is_recording()
    set_phase_parent(request_span)

    context = _text_context()
    inject_trace_carrier(context)

    carried = otel_trace.get_current_span(
        _parent_trace_context(context)
    ).get_span_context()
    assert carried.span_id == request_span.get_span_context().span_id


@pytest.mark.parametrize(
    "ambient", [None, Context()], ids=["no-handler", "no-traceparent"]
)
def test_no_ambient_span_leaves_the_carrier_unset(
    ambient: Context | None,
) -> None:
    # No route handler (None), and headers with no traceparent (an empty
    # Context). Neither may put an invalid span id on the wire.
    request_trace_ctx.set(ambient)
    context = _text_context()
    inject_trace_carrier(context)

    assert context.trace_carrier is None
    assert _parent_trace_context(context) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "generator_type",
    [OpenAIChatResponseGenerator, OpenAICompletionResponseGenerator],
    ids=["chat", "completions"],
)
async def test_the_carrier_reaching_the_pipeline_names_the_request_span(
    tracer: otel_trace.Tracer,
    finished: InMemorySpanExporter,
    generator_type: type[
        OpenAIChatResponseGenerator | OpenAICompletionResponseGenerator
    ],
) -> None:
    # Injects from inside the pipeline call, where the handler's context is
    # live. Covers completions, whose gather() fans the context to child
    # tasks, so the set has to precede it. A caller's context is set too,
    # and must not displace max.request.
    caller = tracer.start_span("caller")
    request_trace_ctx.set(otel_trace.set_span_in_context(caller))
    seen: list[otel_trace.Span] = []

    async def all_tokens(
        request: object, *, parse_reasoning: bool = True
    ) -> list[TokenGeneratorOutput]:
        context = _text_context()
        inject_trace_carrier(context)
        parent = _parent_trace_context(context)
        assert parent is not None
        seen.append(otel_trace.get_current_span(parent))
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

    with (
        patch("max.serve.router.openai_routes._tracer", tracer),
        patch("max.serve.router.openai_routes.METRICS", MagicMock()),
        patch("max.serve.router.openai_routes.record_request_start"),
        patch("max.serve.router.openai_routes.record_request_end"),
    ):
        await generator_type(pipeline).complete([_mock_request()])

    # The carrier rebuilds a NonRecordingSpan, which has ids but no name.
    (request_span,) = [
        span
        for span in finished.get_finished_spans()
        if span.name == "max.request"
    ]
    assert len(seen) == 1
    assert (
        seen[0].get_span_context().span_id
        == request_span.get_span_context().span_id
    )


@pytest.mark.asyncio
async def test_complete_leaves_the_request_trace_context_alone(
    tracer: otel_trace.Tracer,
) -> None:
    # The request's trace context has to survive the call, for anything that
    # reads it later in the request.
    caller = tracer.start_span("caller")
    inbound = otel_trace.set_span_in_context(caller)
    request_trace_ctx.set(inbound)

    pipeline = Mock()
    pipeline.model_name = "test-model"
    pipeline.all_tokens = AsyncMock(
        return_value=[
            TokenGeneratorOutput(
                status=GenerationStatus.END_OF_SEQUENCE,
                decoded_tokens="hi",
                prompt_token_count=4,
            )
        ]
    )

    with (
        patch("max.serve.router.openai_routes._tracer", tracer),
        patch("max.serve.router.openai_routes.METRICS", MagicMock()),
        patch("max.serve.router.openai_routes.record_request_start"),
        patch("max.serve.router.openai_routes.record_request_end"),
    ):
        await OpenAIChatResponseGenerator(pipeline).complete([_mock_request()])

    assert request_trace_ctx.get() is inbound


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "generator_type",
    [OpenAIChatResponseGenerator, OpenAICompletionResponseGenerator],
    ids=["chat", "completions"],
)
async def test_tracing_off_leaves_the_phase_parent_unset(
    generator_type: type[
        OpenAIChatResponseGenerator | OpenAICompletionResponseGenerator
    ],
) -> None:
    # No provider is installed, so the handler's span is a no-op, and the
    # middleware sets no request trace context to fall back to.
    seen: list[tuple[Context | None, int]] = []

    async def all_tokens(
        request: object, *, parse_reasoning: bool = True
    ) -> list[TokenGeneratorOutput]:
        context = _text_context()
        inject_trace_carrier(context)
        carried = otel_trace.get_current_span(_parent_trace_context(context))
        seen.append(
            (_phase_parent_ctx.get(), carried.get_span_context().span_id)
        )
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

    with (
        patch("max.serve.router.openai_routes.METRICS", MagicMock()),
        patch("max.serve.router.openai_routes.record_request_start"),
        patch("max.serve.router.openai_routes.record_request_end"),
        patch.object(otel_propagate, "inject", side_effect=AssertionError),
    ):
        await generator_type(pipeline).complete([_mock_request()])

    assert seen == [(None, 0)]


def test_tracing_off_skips_the_span_context_read() -> None:
    span = _tracer.start_span("max.request")
    assert span is otel_trace.INVALID_SPAN
    with patch.object(span, "get_span_context") as get_span_context:
        set_phase_parent(span)
    get_span_context.assert_not_called()


def test_a_no_op_parent_span_leaves_the_phase_parent_unset() -> None:
    # From OpenTelemetry 1.40, the no-op tracer returns the parent's span
    # instead of an invalid one.
    inbound = otel_propagate.extract(
        {"traceparent": f"00-{'1' * 32}-{'2' * 16}-01"}
    )
    request_trace_ctx.set(inbound)
    set_phase_parent(otel_trace.get_current_span(inbound))

    assert _phase_parent_ctx.get() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True], ids=["ok", "failed"])
@pytest.mark.parametrize(
    "generator_type",
    [OpenAIChatResponseGenerator, OpenAICompletionResponseGenerator],
    ids=["chat", "completions"],
)
async def test_a_streamed_request_span_parents_the_phases(
    tracer: otel_trace.Tracer,
    finished: InMemorySpanExporter,
    generator_type: type[
        OpenAIChatResponseGenerator | OpenAICompletionResponseGenerator
    ],
    fails: bool,
) -> None:
    seen: list[otel_trace.Span] = []

    async def next_token_chunk(request: object, **_: object) -> object:
        seen.append(otel_trace.get_current_span(_phase_parent_ctx.get()))
        if fails:
            raise RuntimeError("the model worker is gone")
        return MagicMock()

    pipeline = Mock(next_token_chunk=next_token_chunk)

    with (
        patch("max.serve.router.openai_routes._tracer", tracer),
        pytest.raises(RuntimeError) if fails else nullcontext(),
    ):
        stream = await generator_type(pipeline).stream(_mock_request())
        assert not finished.get_finished_spans()
        async for _ in stream:
            pass

    (request_span,) = finished.get_finished_spans()
    assert seen[0].get_span_context().span_id == request_span.context.span_id
