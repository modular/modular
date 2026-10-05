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
"""Tests the kernel-tracing limits a ``--config-file`` sets for the model
worker."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from max.pipelines.lib import (
    PipelineArgs,
    PipelineConfig,
    PipelineRole,
    ProfilingConfig,
)
from max.serve.config import Settings
from max.serve.pipelines import model_worker
from max.serve.telemetry import _kernel_capture, common
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.trace import Link, SpanContext


def test_config_file_limits_reach_the_pipeline_config(tmp_path: Path) -> None:
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        "profiling:\n"
        "  kernel_trace_max_passes: 128\n"
        "  kernel_trace_max_batch_links: 256\n"
        "  kernel_trace_max_spans: 40000\n"
        "  kernel_trace_max_capture_bytes: 134217728\n",
        encoding="utf-8",
    )
    profiling = PipelineConfig.from_args(
        PipelineArgs.from_flat_kwargs(config_file=str(recipe))
    ).profiling
    assert (
        profiling.kernel_trace_max_passes,
        profiling.kernel_trace_max_batch_links,
        profiling.kernel_trace_max_spans,
        profiling.kernel_trace_max_capture_bytes,
    ) == (128, 256, 40_000, 128 * 1024 * 1024)


@pytest.mark.parametrize(
    ("limits", "capture", "batch", "warned"),
    [
        pytest.param({}, True, True, [], id="defaults"),
        pytest.param(
            {"kernel_trace_max_passes": 32}, True, True, [], id="lowered"
        ),
        pytest.param(
            {"kernel_trace_max_passes": 65},
            True,
            False,
            ["kernel_trace_max_passes"],
            id="raised",
        ),
        pytest.param(
            {"kernel_trace_max_passes": 65, "kernel_trace_max_spans": 20_001},
            False,
            True,
            [],
            id="no-capture",
        ),
        pytest.param(
            {"kernel_trace_max_batch_links": 129},
            False,
            True,
            ["kernel_trace_max_batch_links"],
            id="batch-links",
        ),
        pytest.param(
            {"kernel_trace_max_batch_links": 129},
            False,
            False,
            [],
            id="below-batch-level",
        ),
        pytest.param(
            {
                "kernel_trace_max_passes": 65,
                "kernel_trace_max_batch_links": 129,
                "kernel_trace_max_spans": 20_001,
                "kernel_trace_max_capture_bytes": 64 * 1024 * 1024 + 1,
            },
            True,
            False,
            [
                "kernel_trace_max_passes",
                "kernel_trace_max_spans",
                "kernel_trace_max_capture_bytes",
                "kernel_trace_max_batch_links",
            ],
            id="all",
        ),
    ],
)
def test_warns_once_per_raised_limit_in_use(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    limits: dict[str, int],
    capture: bool,
    batch: bool,
    warned: list[str],
) -> None:
    monkeypatch.setattr(common, "batch_spans_enabled", lambda: batch)
    monkeypatch.setattr(common, "_tracing_enabled", lambda: True)
    with caplog.at_level(logging.WARNING, logger="max.serve"):
        model_worker._warn_raised_kernel_trace_limits(
            ProfilingConfig.model_validate(limits),
            "prefill_and_decode",
            kernel_capture=capture,
        )
    assert [r.getMessage().split(" ")[0] for r in caplog.records] == [
        f"profiling.{name}" for name in warned
    ]


def test_no_link_warning_without_exported_spans(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """At level batch, a worker that exports no spans has no ``max.batch``
    spans."""
    monkeypatch.setattr(common, "batch_spans_enabled", lambda: True)
    monkeypatch.setattr(common, "_tracing_enabled", lambda: False)
    with caplog.at_level(logging.WARNING, logger="max.serve"):
        model_worker._warn_raised_kernel_trace_limits(
            ProfilingConfig(kernel_trace_max_batch_links=129),
            "prefill_and_decode",
            kernel_capture=False,
        )
    assert caplog.records == []


@pytest.mark.parametrize("role", ["prefill_only", "decode_only"])
def test_no_link_warning_in_a_disaggregated_worker(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    role: PipelineRole,
) -> None:
    """Disaggregated schedulers emit no ``max.batch`` spans."""
    monkeypatch.setattr(common, "batch_spans_enabled", lambda: True)
    monkeypatch.setattr(common, "_tracing_enabled", lambda: True)
    with caplog.at_level(logging.WARNING, logger="max.serve"):
        model_worker._warn_raised_kernel_trace_limits(
            ProfilingConfig(kernel_trace_max_batch_links=129),
            role,
            kernel_capture=False,
        )
    assert caplog.records == []


@pytest.mark.parametrize(
    ("headers", "role", "permitted", "expected"),
    [
        (False, "prefill_and_decode", True, False),
        (True, "decode_only", True, False),
        (True, "prefill_and_decode", False, False),
        (True, "prefill_and_decode", True, True),
    ],
)
def test_configure_kernel_capture_returns_the_permission(
    monkeypatch: pytest.MonkeyPatch,
    headers: bool,
    role: PipelineRole,
    permitted: bool,
    expected: bool,
) -> None:
    """The worker warns about the capture limits only when it may capture."""
    monkeypatch.setattr(common, "_trace_level_header_enabled", headers)
    monkeypatch.setattr(
        _kernel_capture, "configure_kernel_capture", lambda settings: None
    )
    monkeypatch.setattr(
        _kernel_capture, "kernel_capture_permitted", lambda: permitted
    )
    assert model_worker._configure_kernel_capture(Settings(), role) is expected


@pytest.mark.parametrize(("links", "kept"), [(64, 128), (128, 128), (129, 129)])
def test_worker_tracing_raises_only_a_raised_link_limit(
    monkeypatch: pytest.MonkeyPatch, links: int, kept: int
) -> None:
    """128 kept from 64 means the SDK's default limit, not the value."""
    providers: list[TracerProvider] = []
    monkeypatch.setattr(common, "set_tracer_provider", providers.append)
    monkeypatch.delenv("MAX_SERVE_DISABLE_TELEMETRY", raising=False)
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    monkeypatch.delenv("OTEL_SPAN_LINK_COUNT_LIMIT", raising=False)
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://localhost:4318/v1/traces"
    )
    model_worker._configure_worker_tracing(
        Settings(disable_telemetry=False),
        ProfilingConfig(kernel_trace_max_batch_links=links),
    )
    link = Link(SpanContext(trace_id=1, span_id=1, is_remote=True))
    # Left open, so nothing is exported.
    span = providers[0].get_tracer("test").start_span("s", links=[link] * 300)
    assert isinstance(span, ReadableSpan)
    assert len(span.links) == kept
    providers[0].shutdown()
