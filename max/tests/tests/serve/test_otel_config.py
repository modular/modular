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
"""Tests that MAX Serve honours the standard OTel SDK configuration
environment variables.

MAX deliberately supplies no ``service.name``, no sampler and, once an
operator has set an endpoint variable, no endpoint — leaving each to the
SDK's own resolution. Those omissions decide the behaviour but are
invisible at the call site, so they are pinned here.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator

import pytest
from max.serve.config import Settings
from max.serve.telemetry import common
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

GENERIC = "OTEL_EXPORTER_OTLP_ENDPOINT"
TRACES = "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"


@pytest.fixture(autouse=True)
def clear_otel_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Isolates each test from the environment it inherits.

    The bazel target sets MAX_SERVE_DISABLE_TELEMETRY, which reaches Settings
    by alias and would otherwise decide the outcome of tests about the OTel
    variables.
    """
    for name in [n for n in os.environ if n.startswith("OTEL_")]:
        monkeypatch.delenv(name)
    monkeypatch.delenv("MAX_SERVE_DISABLE_TELEMETRY", raising=False)
    yield


@pytest.mark.parametrize(
    ("build", "signal_var", "path"),
    [pytest.param(common._span_exporter, TRACES, "/v1/traces", id="traces")],
)
def test_exporter_endpoint_precedence(
    monkeypatch: pytest.MonkeyPatch,
    build: Callable[[], OTLPSpanExporter],
    signal_var: str,
    path: str,
) -> None:
    """The last two assertions are the ones that matter: reading either
    variable's value and passing it as ``endpoint=`` would let the generic one
    win, inverting the precedence the OTel spec requires."""
    assert build()._endpoint == common.otelBaseUrl + path

    monkeypatch.setenv(GENERIC, "http://agent:4318")
    assert build()._endpoint == "http://agent:4318" + path

    monkeypatch.setenv(signal_var, "http://specific:4318/sink")
    assert build()._endpoint == "http://specific:4318/sink"

    monkeypatch.delenv(GENERIC)
    assert build()._endpoint == "http://specific:4318/sink"


@pytest.mark.parametrize(
    "attrs",
    [common._LOGS_RESOURCE_ATTRS, common._METRICS_RESOURCE_ATTRS],
    ids=["logs", "metrics"],
)
def test_resource_leaves_service_name_to_the_sdk(
    monkeypatch: pytest.MonkeyPatch, attrs: dict[str, str]
) -> None:
    """``Resource.create`` merges the env-derived resource first and explicit
    attributes last, so a ``service.name`` passed by MAX would beat
    ``OTEL_SERVICE_NAME`` and strip an operator's ability to name the
    service. Rebuilt here under a chosen value: asserting on the imported
    resource would pass against a MAX-hardcoded ``unknown_service`` too."""
    monkeypatch.setenv("OTEL_SERVICE_NAME", "operator-chosen")
    assert (
        Resource.create(attrs).attributes["service.name"] == "operator-chosen"
    )


def test_configure_tracing_passes_no_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Captured at the call site because the global provider is write-once, so
    a test that installed one would couple to test order."""
    captured: list[TracerProvider] = []
    monkeypatch.setattr(common, "set_tracer_provider", captured.append)
    # Keeps the exporter's batch thread off the production collector.
    monkeypatch.setenv(TRACES, "http://localhost:4318/v1/traces")
    monkeypatch.setenv("OTEL_TRACES_SAMPLER", "traceidratio")
    monkeypatch.setenv("OTEL_TRACES_SAMPLER_ARG", "0.25")
    common.configure_tracing(Settings(disable_telemetry=False))
    assert isinstance(captured[0].sampler, TraceIdRatioBased)
