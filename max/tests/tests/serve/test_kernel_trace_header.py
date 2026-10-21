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
"""Pins when the API server reads ``x-max-trace-level`` and how the level
reaches the model worker in ``trace_carrier``."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi import FastAPI, Request
from fastapi.requests import HTTPConnection
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from max.pipelines.context import TextContext, TokenBuffer
from max.pipelines.modeling.types import RequestID
from max.serve import request as serve_request
from max.serve.config import Settings
from max.serve.request import register_request
from max.serve.telemetry import common
from max.serve.telemetry._trace_context import (
    RequestTraceLevel,
    inject_trace_carrier,
)
from opentelemetry.sdk.trace import TracerProvider


@pytest.fixture(autouse=True)
def restore_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(common, "_trace_level_header_enabled", False)


def _open_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    # The gate implies a provider, which the middleware checks first.
    monkeypatch.setattr(serve_request, "_tracing_enabled", lambda: True)
    monkeypatch.setattr(common, "_trace_level_header_enabled", True)


def _make_app() -> FastAPI:
    # A plain Starlette route: FastAPI's dependency solving reads every
    # request's headers, which would hide the middleware's reads.
    async def carrier(request: Request) -> JSONResponse:
        context = TextContext(
            request_id=RequestID(),
            max_length=16,
            tokens=TokenBuffer(np.ones(4, dtype=np.int64)),
        )
        inject_trace_carrier(context)
        return JSONResponse({"carrier": context.trace_carrier})

    app = FastAPI()
    app.add_route("/v1/carrier", carrier)
    register_request(app)
    return app


def _carrier(headers: dict[str, str]) -> object:
    with TestClient(_make_app()) as client:
        response = client.get("/v1/carrier", headers=headers)
    assert response.status_code == 200
    return response.json()["carrier"]


@pytest.mark.parametrize(
    (
        "headers_on",
        "telemetry_disabled",
        "traces_endpoint",
        "expected_installed",
        "expected_gate",
    ),
    [
        pytest.param(False, False, True, True, False, id="default"),
        pytest.param(True, True, True, False, False, id="no-provider"),
        pytest.param(True, False, True, True, True, id="permitted"),
        pytest.param(True, False, False, False, False, id="no-traces-endpoint"),
    ],
)
def test_configure_tracing_sets_the_gate(
    monkeypatch: pytest.MonkeyPatch,
    headers_on: bool,
    telemetry_disabled: bool,
    traces_endpoint: bool,
    expected_installed: bool,
    expected_gate: bool,
) -> None:
    installed: list[TracerProvider] = []
    monkeypatch.setattr(common, "set_tracer_provider", installed.append)
    if traces_endpoint:
        monkeypatch.setenv(
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
            "http://localhost:4318/v1/traces",
        )
    else:
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    common.configure_tracing(
        Settings(
            kernel_trace_headers=headers_on,
            disable_telemetry=telemetry_disabled,
        )
    )
    for provider in installed:
        provider.shutdown()
    assert bool(installed) is expected_installed
    assert common._trace_level_header_enabled is expected_gate


def test_default_never_reads_the_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads: list[str] = []
    real_headers = HTTPConnection.headers

    def counting_headers(self: HTTPConnection) -> object:
        reads.append(self.url.path)
        return real_headers.__get__(self)

    monkeypatch.setattr(HTTPConnection, "headers", property(counting_headers))
    assert _carrier({"x-max-trace-level": "kernel"}) is None
    assert reads == []

    # Tracing reads them once, for the inbound trace context.
    monkeypatch.setattr(serve_request, "_tracing_enabled", lambda: True)
    assert _carrier({"x-max-trace-level": "kernel"}) is None
    assert reads == ["/v1/carrier"]

    # The counter does see the middleware's read once the gate opens.
    reads.clear()
    _open_gate(monkeypatch)
    assert _carrier({"x-max-trace-level": "kernel"}) is not None
    assert reads == ["/v1/carrier", "/v1/carrier"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param("kernel", "kernel", id="kernel"),
        pytest.param("Kernel_Sampled", "kernel-sampled", id="normalized"),
        pytest.param("bogus", None, id="unknown-ignored"),
    ],
)
def test_permitted_request_puts_the_level_in_the_carrier(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: str | None
) -> None:
    _open_gate(monkeypatch)
    carrier = _carrier({"x-max-trace-level": value})
    if expected is None:
        assert carrier is None
    else:
        assert carrier == {"x-max-trace-level": expected}


def test_permitted_request_without_the_header_has_no_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _open_gate(monkeypatch)
    assert _carrier({}) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("off", RequestTraceLevel.OFF),
        ("BATCH", RequestTraceLevel.BATCH),
        ("kernel-sampled", RequestTraceLevel.KERNEL_SAMPLED),
        ("kernel_sampled", RequestTraceLevel.KERNEL_SAMPLED),
        ("full", RequestTraceLevel.FULL),
        ("verbose", None),
        ("", None),
    ],
)
def test_parse(value: str, expected: RequestTraceLevel | None) -> None:
    assert RequestTraceLevel.parse(value) is expected


def test_levels_are_ordered() -> None:
    members = list(RequestTraceLevel)
    assert all(members[i] < members[i + 1] for i in range(len(members) - 1))
    assert RequestTraceLevel.OFF < RequestTraceLevel.BATCH
    assert max(RequestTraceLevel.OP, RequestTraceLevel.KERNEL_SAMPLED) is (
        RequestTraceLevel.KERNEL_SAMPLED
    )
