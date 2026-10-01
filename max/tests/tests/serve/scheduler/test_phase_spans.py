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
"""The scheduler's phase spans must follow each request from prefill to decode."""

from __future__ import annotations

import pytest
from max.pipelines.context import TextContext
from max.pipelines.modeling.types import RequestID
from max.serve.scheduler.base import SchedulerProgress
from max.serve.scheduler.text_generation_scheduler import (
    TokenGenerationScheduler,
)
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from tests.serve.scheduler.common import (
    create_paged_scheduler,
    create_text_context,
)

_SCHEDULER_MODULE = "max.serve.scheduler.text_generation_scheduler"


@pytest.fixture
def finished(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
    """Records the scheduler's spans through a local SDK provider.

    The global provider is set-once and would leak across the target, so the
    scheduler's tracer and its tracing check are patched instead.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        f"{_SCHEDULER_MODULE}._tracer", provider.get_tracer("test")
    )
    monkeypatch.setattr(f"{_SCHEDULER_MODULE}._tracing_enabled", lambda: True)
    return exporter


def _spans(
    exporter: InMemorySpanExporter, name: str, request_id: RequestID
) -> list[ReadableSpan]:
    return [
        span
        for span in exporter.get_finished_spans()
        if span.name == name
        and (span.attributes or {}).get("max.request_id") == str(request_id)
    ]


def _run_until_idle(scheduler: TokenGenerationScheduler) -> None:
    for _ in range(50):
        if scheduler.run_iteration() == SchedulerProgress.NO_PROGRESS:
            return
    raise AssertionError("scheduler still had work after 50 iterations")


def _assert_decoding(
    scheduler: TokenGenerationScheduler,
    finished: InMemorySpanExporter,
    context: TextContext,
) -> None:
    request_id = context.request_id
    assert request_id in scheduler.batch_constructor.all_tg_reqs
    assert len(_spans(finished, "max.phase.prefill", request_id)) == 1
    assert request_id not in scheduler._prefill_spans
    decode = scheduler._decode_spans[request_id]
    assert isinstance(decode, ReadableSpan)
    assert decode.name == "max.phase.decode"


@pytest.mark.parametrize("dp", [1, 2])
def test_prefill_span_ends_when_the_request_starts_decoding(
    finished: InMemorySpanExporter, dp: int
) -> None:
    scheduler, request_queue = create_paged_scheduler(
        max_seq_len=128, num_blocks=64, page_size=8, dp=dp
    )
    contexts = [create_text_context(16, max_seq_len=20) for _ in range(dp)]
    for context in contexts:
        request_queue.put_nowait(context)

    scheduler.run_iteration()

    # One request per replica, so every replica's batch is covered.
    assert [
        len(replica.tg_reqs) for replica in scheduler.batch_constructor.replicas
    ] == [1] * dp
    for context in contexts:
        _assert_decoding(scheduler, finished, context)

    _run_until_idle(scheduler)

    for context in contexts:
        assert (
            len(_spans(finished, "max.phase.decode", context.request_id)) == 1
        )
    assert not scheduler._prefill_spans
    assert not scheduler._decode_spans


def test_prefill_span_stays_open_between_chunks(
    finished: InMemorySpanExporter,
) -> None:
    scheduler, request_queue = create_paged_scheduler(
        max_seq_len=128,
        num_blocks=64,
        max_batch_size=8,
        page_size=8,
        target_tokens_per_batch_ce=64,
        enable_chunked_prefill=True,
    )
    context = create_text_context(100, max_seq_len=104)
    request_queue.put_nowait(context)

    scheduler.run_iteration()

    request_id = context.request_id
    assert request_id in scheduler.batch_constructor.all_ce_reqs
    assert request_id in scheduler._prefill_spans
    assert not _spans(finished, "max.phase.prefill", request_id)
    assert request_id not in scheduler._decode_spans

    scheduler.run_iteration()

    _assert_decoding(scheduler, finished, context)


def test_preempted_request_restarts_its_decode_span(
    finished: InMemorySpanExporter,
) -> None:
    # Blocks for exactly one request at full length, so the second is
    # preempted back to CE and prefills again.
    scheduler, request_queue = create_paged_scheduler(
        max_seq_len=110,
        num_blocks=5,
        max_batch_size=999,
        page_size=2,
        enable_chunked_prefill=False,
    )
    contexts = [create_text_context(3, max_seq_len=10) for _ in range(2)]
    for context in contexts:
        request_queue.put_nowait(context)

    _run_until_idle(scheduler)

    assert scheduler.batch_constructor.total_preemption_count == 1
    decode_spans = sorted(
        (
            _spans(finished, "max.phase.decode", context.request_id)
            for context in contexts
        ),
        key=len,
    )
    assert [len(spans) for spans in decode_spans] == [1, 2]
    before, after = decode_spans[1]
    assert before.end_time is not None and after.start_time is not None
    assert before.end_time <= after.start_time
    for context in contexts:
        assert (
            len(_spans(finished, "max.phase.prefill", context.request_id)) == 1
        )
    assert not scheduler._prefill_spans
    assert not scheduler._decode_spans
