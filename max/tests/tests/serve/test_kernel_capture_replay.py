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
"""Tests replaying a libkineto capture into backdated spans.

``kernel_capture.json`` is synthetic, in the shape of a real B200 capture.
Its ``max.batch`` passes 5 to 7 have GPU work that trails their host ranges,
as under overlap scheduling: pass 5's kernels run inside pass 6's host range.
``kernel_capture_graph_b200.json`` is a real capture of four passes under
overlap scheduling, trimmed; passes 11 to 13 replay a device graph.
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from max.pipelines.lib import ProfilingConfig
from max.serve.telemetry import _kernel_capture
from max.serve.telemetry._kernel_capture import (
    KernelCaptureThread,
    TracedPass,
    replay_kernel_capture,
)
from max.serve.telemetry._trace_context import RequestTraceLevel
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import SpanContext, TraceFlags

BASE_NS = 1_782_967_788_000_000_000
T0_US = 7_551_180_780_000.0
# Each pass's max.batch range in the fixture, in µs after T0_US.
RANGES_US = {
    0: (700.0, 900.0),
    5: (1000.0, 1400.0),
    6: (1500.0, 1850.0),
    7: (1900.0, 2250.0),
    2**31 + 7: (2500.0, 2800.0),
    9: (2850.0, 2890.0),
}


def _us(us: float) -> int:
    return BASE_NS + round((T0_US + us) * 1000)


@pytest.fixture
def exporter(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(_kernel_capture, "_replay_provider", provider)
    yield exporter


@pytest.fixture
def capture(fixture_testdatadirectory: Path) -> str:
    return str(fixture_testdatadirectory / "kernel_capture.json")


def _batch_span(batch_id: int) -> SpanContext:
    assert _kernel_capture._replay_provider is not None
    tracer = _kernel_capture._replay_provider.get_tracer("test")
    span = tracer.start_span("max.batch", attributes={"max.batch.id": batch_id})
    span.end()
    return span.get_span_context()


def _pass(
    batch_id: int,
    level: RequestTraceLevel,
    clock_skew_ns: int = 0,
) -> TracedPass:
    """Returns a traced pass whose host times bracket its fixture range."""
    start_us, end_us = RANGES_US[batch_id & 0xFFFF_FFFF]
    host_start_ns = _us(start_us) - 20_000 - clock_skew_ns
    return TracedPass(
        batch_id=batch_id,
        level=level,
        batch_span_context=_batch_span(batch_id),
        host_start_ns=host_start_ns,
        host_end_ns=_us(end_us) + 30_000 - clock_skew_ns,
        batch_start_ns=host_start_ns + 5_000,
    )


def _replayed(
    exporter: InMemorySpanExporter,
) -> dict[str, list[ReadableSpan]]:
    by_name: dict[str, list[ReadableSpan]] = {}
    for span in exporter.get_finished_spans():
        if span.name != "max.batch":
            by_name.setdefault(span.name, []).append(span)
    return by_name


def _event(cat: str, name: str, us: float, **args: int) -> dict[str, object]:
    return {
        "ph": "X",
        "cat": cat,
        "name": name,
        "ts": T0_US + us,
        "dur": 5.0,
        "args": args,
    }


def test_kernel_level_groups_by_launch_time(
    exporter: InMemorySpanExporter, capture: str
) -> None:
    traced = _pass(5, RequestTraceLevel.KERNEL)
    replay_kernel_capture(capture, [traced])

    spans = _replayed(exporter)
    [gpu] = spans.pop("max.batch.gpu")
    assert gpu.parent is not None
    assert gpu.parent.span_id == traced.batch_span_context.span_id
    assert gpu.context.trace_id == traced.batch_span_context.trace_id
    assert gpu.attributes == {
        "max.batch.id": 5,
        "max.batch.busy_ns": 145_000,
        "max.batch.start_after_host_ns": 465_000,
        "max.batch.activity_spans": True,
    }
    # Backdated to the GPU window, which starts after the host range ends.
    assert (gpu.start_time, gpu.end_time) == (_us(1450.0), _us(1605.0))

    # The memcpy has no launch record, so the launches either side of its
    # correlation, and its External id, place it.
    assert sorted(spans) == [
        "Memcpy DtoD (Device -> Device)",
        "flash_attention",
        "matmul_sm100_bf16",
        "rms_norm_gpu",
        "silu_mul",
    ]
    [matmul] = spans["matmul_sm100_bf16"]
    assert matmul.parent is not None
    assert matmul.parent.span_id == gpu.context.span_id
    assert matmul.attributes == {
        "max.batch.id": 5,
        "max.kernel.name": "matmul_sm100_bf16_deadbeef",
        "max.kernel.kind": "kernel",
    }
    assert (matmul.start_time, matmul.end_time) == (_us(1462.0), _us(1540.0))
    [memcpy] = spans["Memcpy DtoD (Device -> Device)"]
    assert memcpy.attributes is not None
    assert memcpy.attributes["max.kernel.kind"] == "memcpy"


@pytest.mark.parametrize(
    ("level", "activity_spans"),
    [
        (RequestTraceLevel.BATCH, False),
        (RequestTraceLevel.OP, False),
        (RequestTraceLevel.KERNEL, True),
    ],
)
def test_gpu_busy_time_counts_overlapping_activity_once(
    exporter: InMemorySpanExporter,
    tmp_path: Path,
    level: RequestTraceLevel,
    activity_spans: bool,
) -> None:
    """Two kernels on different streams overlap for 30 µs, then a gap
    precedes a third."""
    capture = tmp_path / "overlap.json"
    capture.write_text(
        json.dumps(
            {
                "baseTimeNanoseconds": BASE_NS,
                "traceEvents": [
                    {
                        **_event(
                            "user_annotation",
                            "max.batch",
                            1000.0,
                            **{"External id": 5},
                        ),
                        "dur": 400.0,
                    },
                    _event(
                        "cuda_driver", "cuLaunchKernel", 1010.0, correlation=1
                    ),
                    _event(
                        "cuda_driver", "cuLaunchKernel", 1020.0, correlation=2
                    ),
                    _event(
                        "cuda_driver", "cuLaunchKernel", 1030.0, correlation=3
                    ),
                    {
                        **_event("kernel", "a", 1450.0, correlation=1),
                        "dur": 50.0,
                    },
                    {
                        **_event("kernel", "b", 1470.0, correlation=2),
                        "dur": 50.0,
                    },
                    {
                        **_event("kernel", "c", 1600.0, correlation=3),
                        "dur": 10.0,
                    },
                ],
            }
        )
    )

    replay_kernel_capture(str(capture), [_pass(5, level)])

    [gpu] = _replayed(exporter)["max.batch.gpu"]
    assert gpu.attributes is not None
    assert gpu.attributes["max.batch.busy_ns"] == 80_000
    assert gpu.end_time is not None and gpu.start_time is not None
    assert gpu.end_time - gpu.start_time == 160_000
    assert gpu.attributes["max.batch.activity_spans"] is activity_spans


def test_op_level_aggregates_each_kernel_name(
    exporter: InMemorySpanExporter, capture: str
) -> None:
    replay_kernel_capture(capture, [_pass(7, RequestTraceLevel.OP)])

    spans = _replayed(exporter)
    [gpu] = spans.pop("max.batch.gpu")
    assert sorted(spans) == [
        "kernel_agg:matmul_sm100_bf16",
        "kernel_agg:rms_norm_gpu",
    ]
    [matmul] = spans["kernel_agg:matmul_sm100_bf16"]
    assert matmul.parent is not None
    assert matmul.parent.span_id == gpu.context.span_id
    assert (matmul.start_time, matmul.end_time) == (_us(2312.0), _us(2435.0))
    assert matmul.attributes == {
        "max.batch.id": 7,
        "max.kernel.op": "matmul_sm100_bf16_deadbeef",
        "max.kernel.count": 2,
        "max.kernel.total_us": 118.0,
        "max.kernel.mean_us": 59.0,
        "max.kernel.max_us": 78.0,
    }


@pytest.mark.parametrize(
    ("pass_6", "with_kernels"),
    [
        pytest.param(
            RequestTraceLevel.KERNEL_SAMPLED, {5, 7}, id="all-sampled"
        ),
        # Only kernel-sampled passes advance the stride.
        pytest.param(RequestTraceLevel.KERNEL, {5, 6}, id="one-kernel"),
    ],
)
def test_kernel_sampled_takes_one_pass_in_stride(
    exporter: InMemorySpanExporter,
    capture: str,
    monkeypatch: pytest.MonkeyPatch,
    pass_6: RequestTraceLevel,
    with_kernels: set[int],
) -> None:
    monkeypatch.setattr(_kernel_capture, "_KERNEL_SAMPLE_STRIDE", 2)
    replay_kernel_capture(
        capture,
        [
            _pass(5, RequestTraceLevel.KERNEL_SAMPLED),
            _pass(6, pass_6),
            _pass(7, RequestTraceLevel.KERNEL_SAMPLED),
        ],
    )

    spans = _replayed(exporter)
    gpu_spans = spans.pop("max.batch.gpu")
    assert len(gpu_spans) == 3
    # Unsampled passes emit nothing beneath max.batch.gpu.
    assert {
        dict(span.attributes or {})["max.batch.id"]
        for named in spans.values()
        for span in named
    } == with_kernels
    assert {
        dict(span.attributes or {})["max.batch.id"]
        for span in gpu_spans
        if dict(span.attributes or {})["max.batch.activity_spans"]
    } == with_kernels


def test_full_adds_what_the_capture_has(
    exporter: InMemorySpanExporter, capture: str
) -> None:
    replay_kernel_capture(capture, [_pass(5, RequestTraceLevel.FULL)])

    spans = _replayed(exporter)
    [matmul] = spans["matmul_sm100_bf16"]
    assert matmul.attributes == {
        "max.batch.id": 5,
        "max.kernel.name": "matmul_sm100_bf16_deadbeef",
        "max.kernel.kind": "kernel",
        "max.kernel.registers_per_thread": 168,
        "max.kernel.occupancy_pct": 12.5,
        "max.kernel.shared_mem": 196608,
    }
    [memcpy] = spans["Memcpy DtoD (Device -> Device)"]
    assert memcpy.attributes == {
        "max.batch.id": 5,
        "max.kernel.name": "Memcpy DtoD (Device -> Device)",
        "max.kernel.kind": "memcpy",
    }


def test_untraced_passes_are_dropped(
    exporter: InMemorySpanExporter, capture: str
) -> None:
    """Pass 6 ran while armed but untraced. Passes 5 and 7 export; pass 6's
    work, and the warm-up kernel launched before any range, do not, even
    where GPU time overlaps a traced pass."""
    replay_kernel_capture(
        capture,
        [
            _pass(5, RequestTraceLevel.KERNEL),
            _pass(7, RequestTraceLevel.KERNEL),
        ],
    )

    spans = _replayed(exporter)
    assert {
        dict(span.attributes or {})["max.batch.id"]
        for span in spans["max.batch.gpu"]
    } == {5, 7}
    assert "warmup_fill" not in spans
    assert len(spans["rms_norm_gpu"]) == 2
    assert len(spans["matmul_sm100_bf16"]) == 3
    assert len(spans["Memcpy DtoD (Device -> Device)"]) == 1


@pytest.mark.parametrize(
    ("batch_id", "window_us"),
    [
        pytest.param(2**32 + 7, (2300.0, 2435.0), id="wraps"),
        pytest.param(2**31 + 7, (2900.0, 2905.0), id="negative-int32"),
        pytest.param(0, (800.0, 805.0), id="zero-omitted"),
    ],
)
def test_range_ids_match_truncated_to_32_bits(
    exporter: InMemorySpanExporter,
    capture: str,
    batch_id: int,
    window_us: tuple[float, float],
) -> None:
    replay_kernel_capture(capture, [_pass(batch_id, RequestTraceLevel.BATCH)])

    [gpu] = _replayed(exporter)["max.batch.gpu"]
    assert gpu.attributes is not None
    assert gpu.attributes["max.batch.id"] == batch_id
    assert (gpu.start_time, gpu.end_time) == tuple(_us(t) for t in window_us)


def test_clock_offset_applies_when_its_bound_excludes_zero(
    exporter: InMemorySpanExporter, capture: str
) -> None:
    """Host times 1 ms behind libkineto's bound the offset to [+0.97,
    +1.02] ms, so spans move back by the midpoint. BATCH adds no kernels."""
    replay_kernel_capture(
        capture,
        [_pass(7, RequestTraceLevel.BATCH, clock_skew_ns=1_000_000)],
    )

    [[gpu]] = _replayed(exporter).values()
    assert gpu.start_time == _us(2300.0) - 995_000


def test_span_cap_keeps_every_gpu_window(
    exporter: InMemorySpanExporter,
    capture: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    replay_kernel_capture(
        capture,
        [
            _pass(5, RequestTraceLevel.KERNEL),
            _pass(7, RequestTraceLevel.KERNEL),
        ],
        ProfilingConfig(kernel_trace_max_spans=4),
    )

    spans = _replayed(exporter)
    gpu_spans = spans.pop("max.batch.gpu")
    assert len(gpu_spans) == 2
    # Both passes lost kernel spans to the cap.
    assert all(
        span.attributes is not None
        and span.attributes["max.batch.activity_spans"] is False
        for span in gpu_spans
    )
    assert sum(len(v) for v in spans.values()) == 2
    assert "dropped 6 spans" in caplog.text


def test_pass_without_captured_activity_emits_nothing(
    exporter: InMemorySpanExporter, capture: str
) -> None:
    """Under overlap scheduling the last pass's kernels can still be running
    when the capture stops: pass 9's range and launch are captured, its
    kernel is not."""
    replay_kernel_capture(
        capture,
        [
            _pass(7, RequestTraceLevel.KERNEL),
            _pass(9, RequestTraceLevel.KERNEL),
        ],
    )

    [gpu] = _replayed(exporter)["max.batch.gpu"]
    assert gpu.attributes == {
        "max.batch.id": 7,
        "max.batch.busy_ns": 128_000,
        "max.batch.start_after_host_ns": 415_000,
        "max.batch.activity_spans": True,
    }


def test_oversized_capture_is_left_unreplayed(
    exporter: InMemorySpanExporter,
    capture: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    replay_kernel_capture(
        capture,
        [_pass(7, RequestTraceLevel.KERNEL)],
        ProfilingConfig(kernel_trace_max_capture_bytes=1024),
    )

    assert _replayed(exporter) == {}
    assert "over the 1024-byte replay limit" in caplog.text
    assert capture in caplog.text


@pytest.mark.parametrize(
    ("max_spans", "queue_size"), [(1, 512), (40_000, 40_000)]
)
def test_replay_queue_holds_the_span_cap(
    monkeypatch: pytest.MonkeyPatch, max_spans: int, queue_size: int
) -> None:
    """The queue holds the span cap, but no fewer spans than the SDK's
    512-span export batch, since the SDK rejects a queue smaller than its
    batch. Its exports warn on failure and on drops, as the main
    exporter's do."""
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://agent:4318"
    )
    monkeypatch.setattr(_kernel_capture, "_replay_provider", None)
    monkeypatch.setattr(_kernel_capture, "span_exporter", InMemorySpanExporter)
    _kernel_capture._replay_tracer(max_spans)
    provider = _kernel_capture._replay_provider
    assert provider is not None
    [processor] = provider._active_span_processor._span_processors
    provider.shutdown()
    assert isinstance(processor, _kernel_capture.CountingSpanProcessor)
    assert processor._batch_processor._queue.maxlen == queue_size


def test_replay_follows_its_max_batch_span_not_the_configured_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTEL_TRACES_SAMPLER", "always_off")
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://agent:4318"
    )
    monkeypatch.setattr(_kernel_capture, "_replay_provider", None)
    monkeypatch.setattr(_kernel_capture, "span_exporter", InMemorySpanExporter)
    batch = SpanContext(1, 1, False, TraceFlags(TraceFlags.SAMPLED))
    span = _kernel_capture._replay_tracer(1).start_span(
        "max.batch.gpu",
        context=trace.set_span_in_context(trace.NonRecordingSpan(batch)),
    )
    assert _kernel_capture._replay_provider is not None
    _kernel_capture._replay_provider.shutdown()
    assert span.get_span_context().trace_flags.sampled


def test_capture_thread_replays_after_a_successful_stop(
    exporter: InMemorySpanExporter,
    capture: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_kernel_capture, "stop_capture", lambda output_path: "")
    thread = KernelCaptureThread(ProfilingConfig(kernel_trace_max_spans=1))
    thread.submit(capture, [_pass(7, RequestTraceLevel.KERNEL)])
    deadline = time.monotonic() + 5
    while thread.busy:
        assert time.monotonic() < deadline
        time.sleep(0.01)

    assert list(_replayed(exporter)) == ["max.batch.gpu"]


def test_capture_thread_survives_a_failed_replay(
    capture: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    stops: list[str] = []

    def stop_capture(output_path: str) -> str:
        stops.append(output_path)
        return ""

    def replay(output_path: str, passes: object, limits: object) -> None:
        raise RuntimeError("bad capture")

    monkeypatch.setattr(_kernel_capture, "stop_capture", stop_capture)
    monkeypatch.setattr(_kernel_capture, "replay_kernel_capture", replay)
    thread = KernelCaptureThread()
    for path in ("/tmp/a.json", "/tmp/b.json"):
        thread.submit(path, [])
        deadline = time.monotonic() + 5
        while thread.busy:
            assert time.monotonic() < deadline
            time.sleep(0.01)

    assert stops == ["/tmp/a.json", "/tmp/b.json"]
    assert caplog.text.count("Kernel capture replay failed") == 2


def test_span_cap_counts_what_it_drops_at_op_level(
    exporter: InMemorySpanExporter,
    capture: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Pass 5 would add five aggregates and pass 7 two."""
    replay_kernel_capture(
        capture,
        [_pass(5, RequestTraceLevel.OP), _pass(7, RequestTraceLevel.OP)],
        ProfilingConfig(kernel_trace_max_spans=1),
    )

    assert list(_replayed(exporter)) == ["max.batch.gpu"]
    assert "dropped 8 spans" in caplog.text


def test_unreadable_capture_logs_and_exports_nothing(
    exporter: InMemorySpanExporter,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = tmp_path / "truncated.json"
    path.write_text('{"traceEvents": [')

    replay_kernel_capture(str(path), [_pass(5, RequestTraceLevel.KERNEL)])

    assert _replayed(exporter) == {}
    assert "Cannot read kernel capture" in caplog.text


def test_real_b200_device_graph_capture(
    exporter: InMemorySpanExporter, fixture_testdatadirectory: Path
) -> None:
    """Places every activity, including copies and device graph kernels,
    which have no launch record, in the pass that enqueued it; op ranges
    whose External ids equal pass ids do not mislead it."""
    path = fixture_testdatadirectory / "kernel_capture_graph_b200.json"
    ranges = {
        e["args"]["External id"]: (e["ts"], e["ts"] + e["dur"])
        for e in json.loads(path.read_text())["traceEvents"]
        if e["cat"] == "user_annotation" and e["name"] == "max.batch"
    }
    passes = [
        TracedPass(
            batch_id=batch_id,
            level=RequestTraceLevel.FULL,
            batch_span_context=_batch_span(batch_id),
            host_start_ns=BASE_NS + round(start_us * 1000) - 10_000,
            host_end_ns=BASE_NS + round(end_us * 1000) + 10_000,
            batch_start_ns=BASE_NS + round(start_us * 1000) - 10_000,
        )
        for batch_id, (start_us, end_us) in ranges.items()
    ]
    replay_kernel_capture(str(path), passes)

    spans = _replayed(exporter)
    del spans["max.batch.gpu"]
    by_pass: dict[int, list[str]] = {}
    for name, named in spans.items():
        for span in named:
            batch_id = dict(span.attributes or {})["max.batch.id"]
            assert isinstance(batch_id, int)
            by_pass.setdefault(batch_id, []).append(name)
    # Every activity in the capture is placed.
    assert {k: len(v) for k, v in by_pass.items()} == {
        10: 16,
        11: 24,
        12: 24,
        13: 16,
    }
    for batch_id in (11, 12, 13):
        assert "gather_r2_w8_b128_gs_False" in by_pass[batch_id]
        # A name the compiler shortens keeps its first 32 characters and
        # ends in a 16-digit hash.
        assert "algorithm_gpu_rowwise__BlockK6A6A6A6A6A6A" in by_pass[batch_id]
    assert "sm100_mha_1q_depth128_bfloat16_bfloat16_nqh32_nkvh8" in by_pass[10]
    assert "gemv_split_k_bfloat16_bfloat16_bfloat16_256" in by_pass[10]
    # The capture writes occupancy 0 as an int.
    [mha] = spans["sm100_mha_1q_depth128_bfloat16_bfloat16_nqh32_nkvh8"]
    occupancy = dict(mha.attributes or {})["max.kernel.occupancy_pct"]
    assert isinstance(occupancy, float) and occupancy == 0.0
    # Its times are large µs offsets, whose differences carry float noise.
    replay_kernel_capture(
        str(path),
        [dataclasses.replace(p, level=RequestTraceLevel.OP) for p in passes],
    )
    [mha_op] = _replayed(exporter)[
        "kernel_agg:sm100_mha_1q_depth128_bfloat16_bfloat16_nqh32_nkvh8"
    ]
    assert dict(mha_op.attributes or {})["max.kernel.total_us"] == 9.151


def test_activity_after_the_last_launch_record(
    exporter: InMemorySpanExporter, tmp_path: Path
) -> None:
    """A tail pass's graph can be its last launch. Its External id picks
    among the passes since the last launch record; a copy whose External id
    names an earlier pass is dropped."""

    capture = tmp_path / "tail.json"
    capture.write_text(
        json.dumps(
            {
                "baseTimeNanoseconds": BASE_NS,
                "traceEvents": [
                    _event("user_annotation", "max.batch", 700.0),
                    _event(
                        "user_annotation",
                        "max.batch",
                        1000.0,
                        **{"External id": 5},
                    ),
                    _event(
                        "cuda_driver",
                        "cuLaunchKernelEx",
                        1010.0,
                        correlation=10,
                    ),
                    _event("kernel", "rms_norm_gpu", 1450.0, correlation=10),
                    _event(
                        "user_annotation",
                        "max.batch",
                        1500.0,
                        **{"External id": 6},
                    ),
                    _event(
                        "kernel",
                        "graph_matmul",
                        1900.0,
                        correlation=20,
                        **{"External id": 6},
                    ),
                    _event("gpu_memcpy", "Memcpy DtoH", 1910.0, correlation=21),
                ],
            }
        )
    )
    replay_kernel_capture(
        str(capture),
        [_pass(b, RequestTraceLevel.KERNEL) for b in (0, 5, 6)],
    )

    spans = _replayed(exporter)
    assert sorted(spans) == ["graph_matmul", "max.batch.gpu", "rms_norm_gpu"]
    [graph_kernel] = spans["graph_matmul"]
    assert graph_kernel.attributes == {
        "max.batch.id": 6,
        "max.kernel.name": "graph_matmul",
        "max.kernel.kind": "kernel",
    }


def test_launch_after_the_last_range_is_dropped(
    exporter: InMemorySpanExporter, tmp_path: Path
) -> None:
    """A pass whose range began after the capture stopped recording ranges
    still has launch records; they don't belong to the pass before it."""
    capture = tmp_path / "stopped.json"
    capture.write_text(
        json.dumps(
            {
                "baseTimeNanoseconds": BASE_NS,
                "traceEvents": [
                    _event(
                        "user_annotation",
                        "max.batch",
                        1000.0,
                        **{"External id": 5},
                    ),
                    _event(
                        "cuda_driver",
                        "cuLaunchKernelEx",
                        1002.0,
                        correlation=10,
                    ),
                    _event("kernel", "rms_norm_gpu", 1450.0, correlation=10),
                    _event(
                        "cuda_driver",
                        "cuLaunchKernelEx",
                        1010.0,
                        correlation=11,
                    ),
                    _event("kernel", "silu_mul", 1460.0, correlation=11),
                ],
            }
        )
    )
    replay_kernel_capture(str(capture), [_pass(5, RequestTraceLevel.KERNEL)])

    assert sorted(_replayed(exporter)) == ["max.batch.gpu", "rms_norm_gpu"]


def test_cut_off_pass_activity_without_a_launch_record_is_dropped(
    exporter: InMemorySpanExporter, tmp_path: Path
) -> None:
    """The cut-off pass 6 also issues copies and a device graph kernel, which
    have no launch record. Whether a launch record after the last range
    follows them or none does, they don't belong to the pass before it."""
    capture = tmp_path / "stopped.json"
    capture.write_text(
        json.dumps(
            {
                "baseTimeNanoseconds": BASE_NS,
                "traceEvents": [
                    _event(
                        "user_annotation",
                        "max.batch",
                        1000.0,
                        **{"External id": 5},
                    ),
                    _event(
                        "cuda_driver",
                        "cuLaunchKernelEx",
                        1002.0,
                        correlation=10,
                    ),
                    _event("kernel", "rms_norm_gpu", 1450.0, correlation=10),
                    _event(
                        "gpu_memcpy",
                        "Memcpy HtoD",
                        1500.0,
                        correlation=11,
                        **{"External id": 6},
                    ),
                    _event(
                        "cuda_driver",
                        "cuLaunchKernelEx",
                        1010.0,
                        correlation=12,
                    ),
                    _event("kernel", "silu_mul", 1510.0, correlation=12),
                    _event(
                        "kernel",
                        "graph_matmul",
                        1530.0,
                        correlation=13,
                        **{"External id": 6},
                    ),
                    _event(
                        "gpu_memcpy",
                        "Memcpy DtoH",
                        1540.0,
                        correlation=14,
                        **{"External id": 6},
                    ),
                ],
            }
        )
    )
    replay_kernel_capture(str(capture), [_pass(5, RequestTraceLevel.KERNEL)])

    spans = _replayed(exporter)
    [gpu] = spans.pop("max.batch.gpu")
    assert (gpu.start_time, gpu.end_time) == (_us(1450.0), _us(1455.0))


def test_capture_without_a_sampled_pass_is_not_read(
    exporter: InMemorySpanExporter,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Spans beneath an unsampled max.batch span would be dropped too, so
    the capture isn't worth parsing."""
    unsampled = TracedPass(
        batch_id=5,
        level=RequestTraceLevel.KERNEL,
        batch_span_context=SpanContext(
            trace_id=1,
            span_id=1,
            is_remote=False,
            trace_flags=TraceFlags(TraceFlags.DEFAULT),
        ),
        host_start_ns=_us(980.0),
        host_end_ns=_us(1430.0),
        batch_start_ns=_us(980.0),
    )
    replay_kernel_capture(str(tmp_path / "missing.json"), [unsampled])

    assert _replayed(exporter) == {}
    assert "Cannot read kernel capture" not in caplog.text
