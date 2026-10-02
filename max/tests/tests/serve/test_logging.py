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
"""Tests for MAX Serve telemetry configuration: the log-handler component
allowlist, the correlation IDs in structured records, and kernel-trace
gating."""

from __future__ import annotations

import json
import logging
import queue
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from max.pipelines.context import TextContext, TextGenerationOutput
from max.pipelines.kv_cache import DummyKVCache
from max.pipelines.modeling.types import RequestID, TextGenerationInputs
from max.serve import request as request_module
from max.serve.config import KernelTraceLevel, Settings
from max.serve.request import register_request
from max.serve.scheduler import text_generation_scheduler
from max.serve.scheduler.config import TokenGenerationSchedulerConfig
from max.serve.scheduler.text_generation_scheduler import (
    TokenGenerationScheduler,
)
from max.serve.telemetry import common
from max.serve.telemetry.common import (
    batch_spans_enabled,
    configure_kernel_tracing,
    configure_logging,
)
from opentelemetry import propagate as otel_propagate
from opentelemetry.sdk._logs import LogData
from opentelemetry.sdk._logs.export import SimpleLogRecordProcessor


@pytest.fixture(autouse=True)
def restore_root_logging() -> Iterator[None]:
    """Undoes ``configure_logging``'s global mutation of the logging tree."""
    root = logging.getLogger()
    uvicorn_logger = logging.getLogger("uvicorn")
    saved = (list(root.handlers), root.level, uvicorn_logger.level)
    try:
        yield
    finally:
        handlers, root_level, uvicorn_level = saved
        for handler in list(root.handlers):
            if handler not in handlers:
                handler.close()
        root.handlers[:] = handlers
        root.setLevel(root_level)
        uvicorn_logger.setLevel(uvicorn_level)


@pytest.fixture
def emitted(tmp_path: Path) -> Iterator[Path]:
    log_path = tmp_path / "max-serve.log"
    configure_logging(
        Settings(
            logs_console_level=None,
            logs_file_level="WARNING",
            logs_file_path=str(log_path),
            disable_telemetry=True,
        )
    )
    yield log_path


def _emit(name: str, level: int, message: str) -> None:
    logging.getLogger(name).log(level, message)
    for handler in logging.getLogger().handlers:
        handler.flush()


def test_admits_uvicorn_error_records(emitted: Path) -> None:
    # uvicorn owns the HTTP error log. Filtering it out left TCP-level
    # failures -- an exception escaping the ASGI app, in-flight requests
    # cancelled when the shutdown drain expires -- with no server-side trace.
    _emit("uvicorn.error", logging.ERROR, "Exception in ASGI application")
    assert "Exception in ASGI application" in emitted.read_text()


def test_admits_max_component_records(emitted: Path) -> None:
    _emit("max.serve", logging.WARNING, "max-serve-warning")
    assert "max-serve-warning" in emitted.read_text()


def test_excludes_uvicorn_access_records(emitted: Path) -> None:
    # The access stream logs at INFO and would swamp the sink one line per
    # request; the WARNING pin on the ``uvicorn`` logger keeps it out.
    _emit("uvicorn.access", logging.INFO, "GET /health")
    assert "GET /health" not in emitted.read_text()


def test_excludes_unrelated_third_party_records(emitted: Path) -> None:
    _emit("httpx", logging.WARNING, "third-party-chatter")
    assert "third-party-chatter" not in emitted.read_text()


def test_kernel_trace_level_gates_batch_spans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # setattr records the pre-test global so teardown restores it even after
    # the configure calls below overwrite it.
    monkeypatch.setattr(common, "_kernel_trace_level", KernelTraceLevel.OFF)
    for level, expected in (("batch", True), ("off", False)):
        monkeypatch.setenv("MAX_SERVE_KERNEL_TRACE_LEVEL", level)
        configure_kernel_tracing(Settings())
        assert batch_spans_enabled() == expected
    # op/kernel imply batch spans. Set the global directly: at these levels
    # configure_kernel_tracing also touches GPU profiling state.
    for deep_level in (KernelTraceLevel.OP, KernelTraceLevel.KERNEL):
        monkeypatch.setattr(common, "_kernel_trace_level", deep_level)
        assert batch_spans_enabled()


# The W3C specification's example IDs, with the hex encoding pinned so a
# change has to be deliberate.
_TRACEPARENT = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
_TRACE_ID_HEX = "4bf92f3577b34da6a3ce929d0e0e4736"


@pytest.fixture(autouse=True)
def reset_request_context() -> Iterator[None]:
    """Stops one test's request context leaking into the next."""
    trace_token = common.request_trace_ctx.set(None)
    id_token = common._request_id_ctx.set(None)
    batch_token = common._batch_id_ctx.set(None)
    try:
        yield
    finally:
        common.request_trace_ctx.reset(trace_token)
        common._request_id_ctx.reset(id_token)
        common._batch_id_ctx.reset(batch_token)


def _configure_structured(log_path: Path) -> None:
    configure_logging(
        Settings(
            logs_console_level=None,
            logs_file_level="WARNING",
            logs_file_path=str(log_path),
            structured_logging=True,
            disable_telemetry=True,
        )
    )


@pytest.fixture
def structured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Structured file logging with tracing on, so records are correlated."""
    monkeypatch.setattr(common, "_tracing_enabled", lambda: True)
    log_path = tmp_path / "max-serve.json"
    _configure_structured(log_path)
    return log_path


def _emit_structured(path: Path, name: str, message: str) -> dict[str, object]:
    _emit(name, logging.WARNING, message)
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    assert len(lines) == 1, f"expected one record, got {lines}"
    return json.loads(lines[0])


def _set_inbound_context() -> None:
    common.request_trace_ctx.set(
        otel_propagate.extract({"traceparent": _TRACEPARENT})
    )


def test_correlation_fields_reach_records_from_child_loggers(
    structured: Path,
) -> None:
    # Every MAX record reaches the handler by propagation from a child
    # logger, which a logger-level filter would skip. The key and encoding
    # matter as much as the value: Datadog joins on the exact string.
    _set_inbound_context()
    common._request_id_ctx.set("req-1234")
    record = _emit_structured(
        structured, "max.serve.router.openai_routes", "handled"
    )

    assert record["dd.trace_id"] == _TRACE_ID_HEX
    assert record["request_id"] == "req-1234"


def test_records_outside_a_request_render_the_fields_empty(
    structured: Path,
) -> None:
    # Startup and shutdown records have no request context, and must still be
    # emitted with the fields empty rather than absent.
    record = _emit_structured(structured, "max.serve", "starting up")

    assert record["message"] == "starting up"
    assert not record["request_id"]
    assert not record["dd.trace_id"]


def test_middleware_stamps_both_ids_on_an_arbitrary_route(
    structured: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The tests above set the ContextVars by hand, so only a real request
    # covers the wiring. This route builds no max.request span, which is what
    # makes it a test of "every route".
    monkeypatch.setattr(request_module, "_tracing_enabled", lambda: True)
    app = FastAPI()
    register_request(app)

    @app.get("/probe")
    async def probe() -> dict[str, bool]:
        logging.getLogger("max.serve").warning("in-route")
        return {"ok": True}

    with TestClient(app) as client:
        response = client.get("/probe", headers={"traceparent": _TRACEPARENT})

    assert response.status_code == 200
    lines = [
        line
        for line in structured.read_text().splitlines()
        if "in-route" in line
    ]
    assert len(lines) == 1, f"expected one in-route record, got {lines}"
    record = json.loads(lines[0])
    assert record["request_id"] == response.headers["X-Request-ID"]
    assert record["dd.trace_id"] == _TRACE_ID_HEX


def _run_forward_pass(
    execute: Callable[
        [TextGenerationInputs[TextContext]],
        dict[RequestID, TextGenerationOutput],
    ],
) -> None:
    """Schedules one empty batch, numbered 41, through ``execute``."""
    pipeline = Mock()
    pipeline.execute = Mock(side_effect=execute)
    scheduler = TokenGenerationScheduler(
        scheduler_config=TokenGenerationSchedulerConfig(
            max_batch_size=1, target_tokens_per_batch_ce=32
        ),
        pipeline=pipeline,
        request_queue=queue.Queue(),
        response_queue=queue.Queue(),
        cancel_queue=queue.Queue(),
        kv_cache=DummyKVCache(),
    )
    scheduler._batch_counter = 41
    scheduler._schedule(TextGenerationInputs(batches=[[]]))


def test_batch_id_is_scoped_to_the_forward_pass(
    structured: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The record the pipeline logs mid-pass carries the batch ID; one logged
    # after the pass must not keep the finished batch's ID.
    monkeypatch.setattr(
        text_generation_scheduler, "_tracing_enabled", lambda: True
    )

    def execute(
        inputs: TextGenerationInputs[TextContext],
    ) -> dict[RequestID, TextGenerationOutput]:
        logging.getLogger("max.pipelines").warning("in-batch")
        return {}

    _run_forward_pass(execute)
    _emit("max.serve", logging.WARNING, "after-batch")

    records = {
        record["message"]: record
        for record in map(json.loads, structured.read_text().splitlines())
    }
    assert records["in-batch"]["batch_id"] == 41
    assert records["after-batch"]["batch_id"] is None


def test_batch_id_is_not_set_without_tracing(
    structured: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Nothing reads it then, even with structured logging on, so a forward
    # pass must not pay to set it.
    monkeypatch.setattr(
        text_generation_scheduler, "_tracing_enabled", lambda: False
    )
    seen: list[int | None] = []

    def execute(
        inputs: TextGenerationInputs[TextContext],
    ) -> dict[RequestID, TextGenerationOutput]:
        seen.append(common._batch_id_ctx.get())
        return {}

    _run_forward_pass(execute)

    assert seen == [None]


@pytest.mark.parametrize("tracing", [False, True], ids=["off", "tracing"])
@pytest.mark.parametrize(
    "structured_logging", [False, True], ids=["plain", "structured"]
)
def test_middleware_captures_the_context_only_when_something_reads_it(
    monkeypatch: pytest.MonkeyPatch, structured_logging: bool, tracing: bool
) -> None:
    # Structured logs read only the request ID, and spans and dd.trace_id
    # need tracing, so the middleware must capture no more than that.
    monkeypatch.setattr(request_module, "_tracing_enabled", lambda: tracing)
    app = FastAPI()
    register_request(app, structured_logging=structured_logging)
    seen: list[object] = []

    @app.get("/probe")
    async def probe() -> dict[str, bool]:
        seen.extend(
            [common._request_id_ctx.get(), common.request_trace_ctx.get()]
        )
        return {"ok": True}

    with TestClient(app) as client:
        response = client.get("/probe", headers={"traceparent": _TRACEPARENT})

    assert response.status_code == 200
    request_id, trace_context = seen
    if tracing or structured_logging:
        assert request_id == response.headers["X-Request-ID"]
    else:
        assert request_id is None
    assert (trace_context is not None) == tracing


def test_structured_logging_without_tracing_adds_only_the_request_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The MAX container images turn structured logging on. With tracing off
    # its lines carry request_id, but no dd.trace_id field.
    monkeypatch.setattr(common, "_tracing_enabled", lambda: False)
    monkeypatch.setattr(request_module, "_tracing_enabled", lambda: False)
    log_path = tmp_path / "max-serve.json"
    _configure_structured(log_path)
    app = FastAPI()
    register_request(app, structured_logging=True)

    @app.get("/probe")
    async def probe() -> dict[str, bool]:
        logging.getLogger("max.serve").warning("in-route")
        return {"ok": True}

    with TestClient(app) as client:
        response = client.get("/probe", headers={"traceparent": _TRACEPARENT})

    for handler in logging.getLogger().handlers:
        assert not isinstance(
            handler.formatter, common._CorrelatedJsonFormatter
        )
    (line,) = [
        line for line in log_path.read_text().splitlines() if "in-route" in line
    ]
    record = json.loads(line)
    assert record["request_id"] == response.headers["X-Request-ID"]
    assert "dd.trace_id" not in record


@pytest.fixture
def exported(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """Returns the attributes of each record the OTLP log handler exports."""
    exported: list[dict[str, object]] = []

    class _Capture:
        def export(self, batch: Iterable[LogData]) -> None:
            for data in batch:
                exported.append(dict(data.log_record.attributes or {}))

        def shutdown(self) -> None:
            pass

        def force_flush(self, timeout_millis: int = 30000) -> bool:
            return True

    monkeypatch.setattr(common, "OTLPLogExporter", lambda **kwargs: _Capture())
    # Export on emit, so the batch processor's timer is not in the way.
    monkeypatch.setattr(
        common, "BatchLogRecordProcessor", SimpleLogRecordProcessor
    )
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    return exported


def test_otlp_export_without_tracing_carries_no_request_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    exported: list[dict[str, object]],
) -> None:
    # Structured logging alone renders request_id, and it too must stay in
    # the console and file output.
    monkeypatch.setattr(common, "_tracing_enabled", lambda: False)
    log_path = tmp_path / "max-serve.json"
    configure_logging(
        Settings(
            logs_console_level="WARNING",
            logs_file_level="WARNING",
            logs_file_path=str(log_path),
            logs_otlp_level="WARNING",
            structured_logging=True,
            disable_telemetry=False,
        )
    )
    common._request_id_ctx.set("req-1234")
    _emit("max.serve", logging.WARNING, "handled")

    rendered = json.loads(log_path.read_text().splitlines()[0])
    assert rendered["request_id"] == "req-1234"
    assert "dd.trace_id" not in rendered
    assert exported, "expected the handler to export a record"
    for attributes in exported:
        assert "request_id" not in attributes, attributes


def test_otlp_export_carries_no_correlation_ids(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    exported: list[dict[str, object]],
) -> None:
    # The OTLP log exporter ships to Modular on a hardcoded endpoint, and the
    # SDK handler exports every non-reserved record attribute. A customer's
    # traceparent is their data, so it must not ride along, even though the
    # console and file handlers render it from the same record first.
    monkeypatch.setattr(common, "_tracing_enabled", lambda: True)
    log_path = tmp_path / "max-serve.json"
    configure_logging(
        Settings(
            logs_console_level="WARNING",
            logs_file_level="WARNING",
            logs_file_path=str(log_path),
            logs_otlp_level="WARNING",
            structured_logging=True,
            disable_telemetry=False,
        )
    )
    _set_inbound_context()
    common._request_id_ctx.set("req-1234")
    common._batch_id_ctx.set(7)
    _emit("max.serve", logging.WARNING, "handled")

    rendered = json.loads(log_path.read_text().splitlines()[0])
    assert rendered["dd.trace_id"] == _TRACE_ID_HEX
    assert rendered["batch_id"] == 7
    assert exported, "expected the handler to export a record"
    for attributes in exported:
        assert "dd.trace_id" not in attributes, attributes
        assert "request_id" not in attributes, attributes
        assert "batch_id" not in attributes, attributes
