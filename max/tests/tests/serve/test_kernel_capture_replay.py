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
``kernel_capture_b200.json`` is a real capture of one pass, trimmed.
``kernel_capture_graph_b200.json`` is a real capture of four passes under
overlap scheduling, trimmed; passes 11 to 13 replay a device graph.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from max.serve.telemetry import _kernel_capture
from max.serve.telemetry._kernel_capture import (
    KernelCaptureThread,
    TracedPass,
    replay_kernel_capture,
)
from max.serve.telemetry._trace_context import RequestTraceLevel
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
    span = tracer.start_span("max.batch", attributes={"max.batch_id": batch_id})
    span.end()
    return span.get_span_context()


def _pass(
    batch_id: int,
    level: RequestTraceLevel,
    clock_skew_ns: int = 0,
) -> TracedPass:
    """Returns a traced pass whose host times bracket its fixture range."""
    start_us, end_us = RANGES_US[batch_id & 0xFFFF_FFFF]
    return TracedPass(
        batch_id=batch_id,
        level=level,
        batch_span_context=_batch_span(batch_id),
        host_start_ns=_us(start_us) - 20_000 - clock_skew_ns,
        host_end_ns=_us(end_us) + 30_000 - clock_skew_ns,
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
    assert gpu.attributes == {"max.batch_id": 5}
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
        "max.batch_id": 5,
        "max.kernel.name": "matmul_sm100_bf16_deadbeef",
    }
    assert (matmul.start_time, matmul.end_time) == (_us(1462.0), _us(1540.0))


def test_batch_level_emits_only_the_gpu_window(
    exporter: InMemorySpanExporter, capture: str
) -> None:
    replay_kernel_capture(capture, [_pass(7, RequestTraceLevel.BATCH)])

    spans = _replayed(exporter)
    assert list(spans) == ["max.batch.gpu"]
    [gpu] = spans["max.batch.gpu"]
    assert (gpu.start_time, gpu.end_time) == (_us(2300.0), _us(2435.0))


@pytest.mark.parametrize(
    ("level", "kernel_spans"),
    [
        (RequestTraceLevel.OP, False),
        (RequestTraceLevel.KERNEL_SAMPLED, False),
        (RequestTraceLevel.FULL, True),
    ],
)
def test_levels_without_their_own_export_fall_back(
    exporter: InMemorySpanExporter,
    capture: str,
    level: RequestTraceLevel,
    kernel_spans: bool,
) -> None:
    replay_kernel_capture(capture, [_pass(7, level)])

    assert ("rms_norm_gpu" in _replayed(exporter)) == kernel_spans


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
        dict(span.attributes or {})["max.batch_id"]
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
    assert gpu.attributes == {"max.batch_id": batch_id}
    assert (gpu.start_time, gpu.end_time) == tuple(_us(t) for t in window_us)


def test_clock_offset_applies_when_its_bound_excludes_zero(
    exporter: InMemorySpanExporter, capture: str
) -> None:
    """Host times 1 ms behind libkineto's bound the offset to [+0.97,
    +1.02] ms, so spans move back by the midpoint."""
    replay_kernel_capture(
        capture,
        [_pass(7, RequestTraceLevel.BATCH, clock_skew_ns=1_000_000)],
    )

    [gpu] = _replayed(exporter)["max.batch.gpu"]
    assert gpu.start_time == _us(2300.0) - 995_000


def test_span_cap_keeps_every_gpu_window(
    exporter: InMemorySpanExporter,
    capture: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(_kernel_capture, "_REPLAY_SPAN_CAP", 4)
    replay_kernel_capture(
        capture,
        [
            _pass(5, RequestTraceLevel.KERNEL),
            _pass(7, RequestTraceLevel.KERNEL),
        ],
    )

    spans = _replayed(exporter)
    assert len(spans.pop("max.batch.gpu")) == 2
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
    assert gpu.attributes == {"max.batch_id": 7}


def test_oversized_capture_is_left_unreplayed(
    exporter: InMemorySpanExporter,
    capture: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(_kernel_capture, "_REPLAY_MAX_CAPTURE_BYTES", 1024)
    replay_kernel_capture(capture, [_pass(7, RequestTraceLevel.KERNEL)])

    assert _replayed(exporter) == {}
    assert "over the 1024-byte replay limit" in caplog.text
    assert capture in caplog.text


def test_capture_thread_replays_after_a_successful_stop(
    exporter: InMemorySpanExporter,
    capture: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_kernel_capture, "stop_capture", lambda output_path: "")
    thread = KernelCaptureThread()
    thread.submit(capture, [_pass(7, RequestTraceLevel.BATCH)])
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

    def replay(output_path: str, passes: object) -> None:
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


def test_real_b200_capture(
    exporter: InMemorySpanExporter, fixture_testdatadirectory: Path
) -> None:
    """Its kernel ran on an AsyncRT thread with a Mojo op's External id, so
    only the launch record ties it to the pass."""
    range_start_us, range_end_us = 7551180781350.135, 7551180791296.097
    traced = TracedPass(
        batch_id=7,
        level=RequestTraceLevel.KERNEL,
        batch_span_context=_batch_span(7),
        host_start_ns=BASE_NS + round(range_start_us * 1000) - 10_000,
        host_end_ns=BASE_NS + round(range_end_us * 1000) + 10_000,
    )
    replay_kernel_capture(
        str(fixture_testdatadirectory / "kernel_capture_b200.json"), [traced]
    )

    spans = _replayed(exporter)
    [gpu] = spans.pop("max.batch.gpu")
    assert gpu.parent is not None
    assert gpu.parent.span_id == traced.batch_span_context.span_id
    [[kernel]] = spans.values()
    assert kernel.name == "mogg_foreach_r2_w8_b128_gs_False"
    assert kernel.attributes == {
        "max.batch_id": 7,
        "max.kernel.name": "mogg_foreach_r2_w8_b128_gs_False_73fac3ef",
    }
    kernel_start_us = 7551180790011.269
    assert kernel.start_time == gpu.start_time
    assert kernel.start_time == BASE_NS + round(kernel_start_us * 1000)
    assert kernel.end_time == BASE_NS + round((kernel_start_us + 2.112) * 1000)


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
            level=RequestTraceLevel.KERNEL,
            batch_span_context=_batch_span(batch_id),
            host_start_ns=BASE_NS + round(start_us * 1000) - 10_000,
            host_end_ns=BASE_NS + round(end_us * 1000) + 10_000,
        )
        for batch_id, (start_us, end_us) in ranges.items()
    ]
    replay_kernel_capture(str(path), passes)

    spans = _replayed(exporter)
    del spans["max.batch.gpu"]
    by_pass: dict[int, list[str]] = {}
    for name, named in spans.items():
        for span in named:
            batch_id = dict(span.attributes or {})["max.batch_id"]
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
        "max.batch_id": 6,
        "max.kernel.name": "graph_matmul",
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


@pytest.mark.parametrize(
    "level",
    [level for level in RequestTraceLevel if level != RequestTraceLevel.OFF],
    ids=lambda level: level.value,
)
def test_cut_off_pass_activity_without_a_launch_record_is_dropped(
    exporter: InMemorySpanExporter, tmp_path: Path, level: RequestTraceLevel
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
    replay_kernel_capture(str(capture), [_pass(5, level)])

    spans = _replayed(exporter)
    [gpu] = spans.pop("max.batch.gpu")
    assert (gpu.start_time, gpu.end_time) == (_us(1450.0), _us(1455.0))
    assert {
        dict(span.attributes or {})["max.kernel.name"]
        for named in spans.values()
        for span in named
    } <= {"rms_norm_gpu"}


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
    )
    replay_kernel_capture(str(tmp_path / "missing.json"), [unsampled])

    assert _replayed(exporter) == {}
    assert "Cannot read kernel capture" not in caplog.text
