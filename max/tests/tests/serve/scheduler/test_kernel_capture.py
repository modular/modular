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

"""Pins when the scheduler arms and stops a per-request kernel capture, and
what its ``max.batch`` spans link to."""

from __future__ import annotations

import atexit
import logging
import pathlib
import queue
import signal
import tempfile
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import pytest
from max.pipelines.context import (
    GenerationStatus,
    TextAndVisionContext,
    TextContext,
    TextGenerationOutput,
    TokenBuffer,
)
from max.pipelines.kv_cache import DummyKVCache
from max.pipelines.modeling.types import (
    Pipeline,
    RequestID,
    TextGenerationInputs,
)
from max.serve.config import KernelTraceLevel, Settings
from max.serve.scheduler import text_generation_scheduler
from max.serve.scheduler.batch_constructor.text_batch_constructor import (
    PreemptionReason,
)
from max.serve.scheduler.config import TokenGenerationSchedulerConfig
from max.serve.scheduler.text_generation_scheduler import (
    TokenGenerationScheduler,
)
from max.serve.telemetry import _kernel_capture, common
from max.serve.telemetry._kernel_capture import (
    KernelCapture,
    KernelCaptureThread,
    TracedPass,
)
from max.serve.telemetry._trace_context import RequestTraceLevel
from opentelemetry import propagate, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags


class _Pipeline(
    Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput]
):
    """Emits one token per request per pass and finishes those in ``done``."""

    def __init__(self) -> None:
        self.done: set[RequestID] = set()

    @property
    def max_batch_size(self) -> int:
        return 4

    def execute(
        self, inputs: TextGenerationInputs[TextContext]
    ) -> dict[RequestID, TextGenerationOutput]:
        outputs: dict[RequestID, TextGenerationOutput] = {}
        for ctx in inputs.flat_batch:
            ctx.update(new_token=1)
            if ctx.request_id in self.done:
                ctx.status = GenerationStatus.END_OF_SEQUENCE
            outputs[ctx.request_id] = ctx.to_generation_output()
        return outputs

    def release(self, request_id: RequestID) -> None:
        pass


@dataclass
class _Harness:
    scheduler: TokenGenerationScheduler
    pipeline: _Pipeline
    request_queue: queue.Queue[TextContext | TextAndVisionContext]
    cancel_queue: queue.Queue[list[RequestID]]
    exporter: InMemorySpanExporter
    events: list[tuple[object, ...]] = field(default_factory=list)
    stops: list[list[TracedPass]] = field(default_factory=list)
    open_ranges: list[int] = field(default_factory=list)
    busy: bool = False
    # False while another capture holds the profiler.
    profiler_idle: bool = True

    def add(
        self, level: str | None = None, parent: SpanContext | None = None
    ) -> RequestID:
        ctx = TextContext(
            request_id=RequestID(),
            max_length=100,
            tokens=TokenBuffer(np.ones(8, dtype=np.int64)),
        )
        carrier: dict[str, str] = {}
        if parent is not None:
            propagate.inject(
                carrier,
                context=trace.set_span_in_context(NonRecordingSpan(parent)),
            )
        if level is not None:
            carrier["x-max-trace-level"] = level
        ctx.trace_carrier = carrier or None
        self.request_queue.put_nowait(ctx)
        return ctx.request_id

    def step(self, *done: RequestID) -> None:
        self.pipeline.done.update(done)
        self.scheduler.run_iteration()

    @property
    def starts(self) -> int:
        return self.events.count(("start",))

    @property
    def batch_links(self) -> list[list[SpanContext]]:
        return [
            [link.context for link in span.links]
            for span in self.exporter.get_finished_spans()
            if span.name == "max.batch"
        ]

    @property
    def linked(self) -> list[RequestID]:
        """The requests the scheduler keeps ``max.batch`` links to."""
        return list(self.scheduler._batch_links._links)


def _request_span(n: int) -> SpanContext:
    return SpanContext(
        trace_id=n,
        span_id=n,
        is_remote=True,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )


def _harness(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tracing: bool = True,
    kernel_capture: bool = True,
    max_passes: int = _kernel_capture._MAX_CAPTURE_PASSES,
) -> _Harness:
    pipeline = _Pipeline()
    request_queue: queue.Queue[TextContext | TextAndVisionContext] = (
        queue.Queue()
    )
    cancel_queue: queue.Queue[list[RequestID]] = queue.Queue()
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # The scheduler reads it once, as it is built.
    monkeypatch.setattr(
        text_generation_scheduler, "_tracing_enabled", lambda: tracing
    )
    harness = _Harness(
        scheduler=TokenGenerationScheduler(
            scheduler_config=TokenGenerationSchedulerConfig(
                max_batch_size=4, target_tokens_per_batch_ce=64
            ),
            pipeline=pipeline,
            request_queue=request_queue,
            response_queue=queue.Queue(),
            cancel_queue=cancel_queue,
            kv_cache=DummyKVCache(),
            kernel_capture=(
                KernelCapture(max_passes) if kernel_capture else None
            ),
        ),
        pipeline=pipeline,
        request_queue=request_queue,
        cancel_queue=cancel_queue,
        exporter=exporter,
    )

    class FakeCaptureThread:
        def __init__(self) -> None:
            harness.events.append(("thread",))

        @property
        def busy(self) -> bool:
            return harness.busy

        def submit(self, output_path: str, passes: list[TracedPass]) -> None:
            assert output_path == _kernel_capture.kernel_capture_path()
            harness.events.append(("stop",))
            harness.stops.append(passes)

    monkeypatch.setattr(
        text_generation_scheduler, "_tracer", provider.get_tracer("test")
    )

    def start_capture() -> bool:
        harness.events.append(
            ("start",) if harness.profiler_idle else ("refused",)
        )
        return harness.profiler_idle

    monkeypatch.setattr(_kernel_capture, "start_capture", start_capture)

    def range_begin_with_id(batch_id: int, name: str) -> int:
        harness.events.append(("begin", batch_id))
        harness.open_ranges.append(batch_id + 100)
        return batch_id + 100

    def range_end(range_id: int) -> None:
        assert range_id == harness.open_ranges.pop()
        harness.events.append(("end",))

    monkeypatch.setattr(
        text_generation_scheduler, "range_begin_with_id", range_begin_with_id
    )
    monkeypatch.setattr(text_generation_scheduler, "range_end", range_end)
    monkeypatch.setattr(
        _kernel_capture, "KernelCaptureThread", FakeCaptureThread
    )
    return harness


def test_two_traced_requests_share_one_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = _harness(monkeypatch)
    a = h.add("kernel")
    b = h.add("batch")
    h.step()
    h.step(a)
    assert h.starts == 1
    assert h.stops == []
    h.step(b)

    assert h.events == [
        ("thread",),
        ("start",),
        ("begin", 0),
        ("end",),
        ("begin", 1),
        ("end",),
        ("begin", 2),
        ("end",),
        ("stop",),
    ]
    (passes,) = h.stops
    assert [(p.batch_id, p.level) for p in passes] == [
        (0, RequestTraceLevel.KERNEL),
        (1, RequestTraceLevel.KERNEL),
        (2, RequestTraceLevel.BATCH),
    ]
    assert all(p.host_start_ns <= p.host_end_ns for p in passes)

    # Traced passes emit max.batch although the global level is off.
    spans = h.exporter.get_finished_spans()
    batch_spans = [s for s in spans if s.name == "max.batch"]
    assert [p.batch_span_context for p in passes] == [
        s.get_span_context() for s in batch_spans
    ]


@pytest.mark.parametrize("exit_by", ["release", "cancel"])
def test_stops_when_the_last_traced_request_leaves(
    monkeypatch: pytest.MonkeyPatch, exit_by: str
) -> None:
    h = _harness(monkeypatch)
    traced = h.add("kernel")
    untraced = h.add()
    h.step()
    assert h.starts == 1

    if exit_by == "release":
        h.step(traced)
    else:
        h.cancel_queue.put_nowait([traced])
        h.step()
    assert len(h.stops) == 1

    # The capture stays down for the untraced request's passes.
    h.step()
    h.step(untraced)
    assert h.starts == 1
    assert len(h.stops) == 1


def test_grammar_failure_untraces_its_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = _harness(monkeypatch)
    running = h.add("kernel")
    h.step()
    bc = h.scheduler.batch_constructor
    enqueue = bc.enqueue_new_request
    failing = h.add("kernel")

    def fail_grammar(ctx: TextContext, replica_idx: int | None = None) -> None:
        if ctx.request_id == failing:
            bc._fail_grammar_request(ctx, "bad schema")
        else:
            enqueue(ctx, replica_idx)

    monkeypatch.setattr(bc, "enqueue_new_request", fail_grammar)
    # The failed request no longer holds the capture open.
    h.step(running)
    assert not bc.contains(failing)
    assert len(h.stops) == 1


def test_preemption_keeps_the_capture_armed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = _harness(monkeypatch)
    traced = h.add("op")
    h.step()
    ctx = h.scheduler.batch_constructor.all_tg_reqs[traced]
    h.scheduler.batch_constructor._preempt_request(
        ctx, 0, PreemptionReason.KV_CACHE_MEMORY
    )
    h.step()
    assert h.stops == []
    h.step(traced)
    assert h.starts == 1
    (passes,) = h.stops
    assert [p.batch_id for p in passes] == [0, 1, 2]


def test_long_capture_stops_at_the_pass_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = _harness(monkeypatch, max_passes=2)
    traced = h.add("kernel")
    h.step()
    h.step()
    assert [p.batch_id for p in h.stops[0]] == [0, 1]
    h.step()
    h.step(traced)
    assert (h.starts, len(h.stops)) == (1, 1)


def test_pass_cap_keeps_requests_outside_the_capture_traced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = _harness(monkeypatch, max_passes=1)
    running = [h.add("kernel") for _ in range(4)]
    # The batch is full, so this one waits through the capped capture.
    waiting = h.add("kernel")
    h.step()
    h.step(*running)
    h.step(waiting)
    assert [[p.batch_id for p in passes] for passes in h.stops] == [[0], [2]]


def test_pass_cap_counts_untraced_passes_while_armed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = _harness(monkeypatch, max_passes=2)
    h.add("kernel")
    h.step()
    h.add()
    h.step()
    assert [[p.batch_id for p in passes] for passes in h.stops] == [[0]]


def test_every_pass_opens_a_range_while_a_capture_may_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = _harness(monkeypatch)
    traced = h.add("kernel")
    h.step()
    untraced = h.add()
    # The untraced request's prefill runs alone, while the capture is armed.
    h.step()
    h.step(traced)
    h.busy = True
    h.step()
    h.busy = False
    h.step(untraced)

    assert h.events == [
        ("thread",),
        ("start",),
        ("begin", 0),
        ("end",),
        ("begin", 1),
        ("end",),
        ("begin", 2),
        ("end",),
        ("stop",),
        ("begin", 3),
        ("end",),
    ]
    (passes,) = h.stops
    assert [p.batch_id for p in passes] == [0, 2]


def test_rearm_waits_for_the_previous_save(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = _harness(monkeypatch)
    first = h.add("kernel")
    h.step(first)
    assert (h.starts, len(h.stops)) == (1, 1)

    h.busy = True
    second = h.add("kernel")
    h.step()
    assert h.starts == 1
    h.busy = False
    h.step(second)
    assert h.events.count(("thread",)) == 1
    assert (h.starts, len(h.stops)) == (2, 2)
    assert [p.batch_id for p in h.stops[1]] == [2]


def test_profiler_held_elsewhere_leaves_passes_untraced(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    h = _harness(monkeypatch)
    h.profiler_idle = False
    traced = h.add("kernel")
    with caplog.at_level(logging.WARNING, logger="max.serve"):
        h.step()
        h.step()
    assert len(caplog.records) == 1
    assert not [
        s for s in h.exporter.get_finished_spans() if s.name == "max.batch"
    ]

    # Arms once the other capture lets the profiler go.
    h.profiler_idle = True
    h.step(traced)
    assert h.events == [
        ("thread",),
        ("refused",),
        ("refused",),
        ("start",),
        ("begin", 2),
        ("end",),
        ("stop",),
    ]
    (passes,) = h.stops
    assert [p.batch_id for p in passes] == [2]


@pytest.mark.parametrize(
    ("tracing", "kernel_capture", "level"),
    [
        pytest.param(True, True, None, id="no-header"),
        pytest.param(True, True, "off", id="header-off"),
        pytest.param(True, True, "bogus", id="header-unknown"),
        pytest.param(False, True, "kernel", id="tracing-off"),
        pytest.param(True, False, "kernel", id="not-permitted"),
    ],
)
def test_no_binding_call_unless_traced(
    monkeypatch: pytest.MonkeyPatch,
    tracing: bool,
    kernel_capture: bool,
    level: str | None,
) -> None:
    h = _harness(monkeypatch, tracing=tracing, kernel_capture=kernel_capture)
    request = h.add(level, parent=_request_span(1))
    h.step()
    assert h.linked == []
    h.step(request)
    assert h.events == []
    capture = h.scheduler._kernel_capture
    assert capture is None or capture._traced == {}
    assert not [
        s for s in h.exporter.get_finished_spans() if s.name == "max.batch"
    ]


def test_global_batch_links_every_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        text_generation_scheduler, "batch_spans_enabled", lambda: True
    )
    h = _harness(monkeypatch, kernel_capture=False)
    a = h.add(parent=_request_span(1))
    b = h.add(parent=_request_span(2))
    c = h.add(parent=_request_span(3))
    h.add()
    h.step()
    h.cancel_queue.put_nowait([c])
    h.step(a)
    h.step(b)
    all_three = [_request_span(1), _request_span(2), _request_span(3)]
    assert h.batch_links == [all_three, all_three, [_request_span(2)]]
    assert h.linked == []


@pytest.mark.parametrize("exit_by", ["release", "cancel", "grammar"])
def test_batch_links_dropped_when_a_request_leaves(
    monkeypatch: pytest.MonkeyPatch, exit_by: str
) -> None:
    monkeypatch.setattr(
        text_generation_scheduler, "batch_spans_enabled", lambda: True
    )
    h = _harness(monkeypatch, kernel_capture=False)
    staying = h.add(parent=_request_span(1))
    leaving = h.add(parent=_request_span(2))
    bc = h.scheduler.batch_constructor

    if exit_by == "release":
        h.step(leaving)
    elif exit_by == "cancel":
        h.cancel_queue.put_nowait([leaving])
        h.step()
    else:
        admit = bc.enqueue_new_request

        def enqueue(ctx: TextContext, replica_idx: int | None = None) -> None:
            if ctx.request_id == leaving:
                bc._fail_grammar_request(ctx, "bad schema")
            else:
                admit(ctx, replica_idx)

        monkeypatch.setattr(bc, "enqueue_new_request", enqueue)
        h.step()
    assert not bc.contains(leaving)
    assert h.linked == [staying]


def test_batch_links_keep_traced_then_first_members_up_to_the_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        text_generation_scheduler, "batch_spans_enabled", lambda: True
    )
    monkeypatch.setattr(text_generation_scheduler, "_MAX_BATCH_LINKS", 2)
    h = _harness(monkeypatch)
    ids = [h.add(parent=_request_span(n)) for n in (1, 2, 3)]
    ids.append(h.add("kernel", parent=_request_span(4)))
    h.step(*ids)
    assert h.batch_links == [[_request_span(4), _request_span(1)]]


def test_batch_links_skip_unsampled_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        text_generation_scheduler, "batch_spans_enabled", lambda: True
    )
    h = _harness(monkeypatch, kernel_capture=False)
    unsampled = SpanContext(
        trace_id=2,
        span_id=2,
        is_remote=True,
        trace_flags=TraceFlags(TraceFlags.DEFAULT),
    )
    ids = [h.add(parent=_request_span(1)), h.add(parent=unsampled)]
    h.step(*ids)
    assert h.batch_links == [[_request_span(1)]]


def test_traced_pass_links_only_traced_members(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = _harness(monkeypatch)
    traced = h.add("kernel", parent=_request_span(1))
    untraced = h.add(parent=_request_span(2))
    cancelled = h.add("batch", parent=_request_span(3))
    h.step()
    h.cancel_queue.put_nowait([cancelled])
    h.step(traced)
    h.step(untraced)
    assert h.batch_links == [
        [_request_span(1), _request_span(3)],
        [_request_span(1), _request_span(3)],
    ]
    assert h.linked == []


def test_pass_cap_drops_the_links_of_requests_it_untraces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = _harness(monkeypatch, max_passes=2)
    capped = h.add("kernel", parent=_request_span(1))
    h.step()
    h.step()
    later = h.add("kernel", parent=_request_span(2))
    h.step()
    h.step(capped, later)
    assert h.batch_links == [
        [_request_span(1)],
        [_request_span(1)],
        [_request_span(2)],
        [_request_span(2)],
    ]


@pytest.mark.parametrize("link_every_member", [True, False])
def test_a_link_outlives_its_trace_only_when_every_member_is_linked(
    link_every_member: bool,
) -> None:
    links = common._BatchLinks(link_every_member, max_links=128)
    request = RequestID()
    parent = trace.set_span_in_context(NonRecordingSpan(_request_span(1)))
    links.admit(request, parent, traced=True)
    links.drop_untraced([request])
    assert list(links._links) == ([request] if link_every_member else [])


def test_capture_thread_stops_off_the_calling_thread(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    calls: list[tuple[str, str]] = []
    release = threading.Event()

    def stop_capture(output_path: str) -> str:
        release.wait(5)
        calls.append((threading.current_thread().name, output_path))
        return "" if len(calls) == 1 else "no trace"

    monkeypatch.setattr(_kernel_capture, "stop_capture", stop_capture)
    thread = KernelCaptureThread()
    assert not thread.busy
    thread.submit("/tmp/a.json", [])
    assert thread.busy
    release.set()
    _wait_idle(thread)
    with caplog.at_level(logging.WARNING, logger="max.serve"):
        thread.submit("/tmp/b.json", [])
        _wait_idle(thread)

    assert calls == [
        ("max-kernel-capture", "/tmp/a.json"),
        ("max-kernel-capture", "/tmp/b.json"),
    ]
    assert "no trace" in caplog.text


def _wait_idle(thread: KernelCaptureThread) -> None:
    deadline = time.monotonic() + 5
    while thread.busy:
        assert time.monotonic() < deadline
        time.sleep(0.01)


@pytest.mark.parametrize(
    ("gate", "level", "plugin", "permitted", "loads", "warns"),
    [
        pytest.param(False, KernelTraceLevel.OFF, True, False, 0, 0, id="off"),
        pytest.param(True, KernelTraceLevel.OFF, True, True, 1, 0, id="on"),
        pytest.param(
            True, KernelTraceLevel.BATCH, False, False, 1, 1, id="no-plugin"
        ),
        pytest.param(
            True, KernelTraceLevel.KERNEL, True, False, 0, 1, id="global-kernel"
        ),
    ],
)
def test_configure_kernel_capture_permits_capture(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    gate: bool,
    level: KernelTraceLevel,
    plugin: bool,
    permitted: bool,
    loads: int,
    warns: int,
) -> None:
    load_calls: list[None] = []

    def load_profiler_plugin() -> bool:
        load_calls.append(None)
        return plugin

    monkeypatch.setattr(
        _kernel_capture, "load_profiler_plugin", load_profiler_plugin
    )
    monkeypatch.setattr(common, "set_gpu_profiling_state", lambda state: None)
    monkeypatch.setattr(common, "_trace_level_header_enabled", gate)
    monkeypatch.setattr(common, "_kernel_trace_level", KernelTraceLevel.OFF)
    monkeypatch.setattr(_kernel_capture, "_kernel_capture_permitted", False)
    # The kernel level may install process-wide SIGTERM handling.
    monkeypatch.setattr(signal, "signal", lambda *args: None)
    monkeypatch.setattr(atexit, "register", lambda *args, **kwargs: None)
    # Set first so teardown removes what the kernel level's setdefault adds.
    monkeypatch.setenv("MODULAR_MAX_DEBUG_PROFILING_ENABLED", "")
    monkeypatch.delenv("MODULAR_MAX_DEBUG_PROFILING_ENABLED")
    with caplog.at_level(logging.WARNING, logger="max.serve"):
        settings = Settings(kernel_trace_level=level, kernel_trace_headers=gate)
        common.configure_kernel_tracing(settings)
        _kernel_capture.configure_kernel_capture(settings)
    assert _kernel_capture.kernel_capture_permitted() is permitted
    assert len(load_calls) == loads
    assert len(caplog.records) == warns


def test_capture_path_is_in_a_private_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(_kernel_capture, "_capture_dir", None)

    path = pathlib.Path(_kernel_capture.kernel_capture_path())

    assert path.parent.parent == tmp_path
    assert path.parent.name.startswith("max-kernel-capture-")
    # No other user can plant a symlink where the plugin writes.
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert _kernel_capture.kernel_capture_path() == str(path)
    assert list(tmp_path.iterdir()) == [path.parent]
