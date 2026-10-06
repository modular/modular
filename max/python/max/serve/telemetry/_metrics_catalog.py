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
"""Describes the metrics a running server exports.

The exported series name, its Prometheus type and its description are
produced by the exporter, not by this module: the catalog comes from a
real Prometheus scrape, so it matches what a user scraping ``/metrics``
sees. Names are not reconstructed here, and neither are types -- an
up-down counter reaches Prometheus as a gauge, which a scrape reports
and a reading of the registration would not.

:func:`build_catalog` answers what the server exports. The MAX serving
metrics reference renders that answer, so nothing records it in
between and nothing can go stale.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from max.serve.telemetry.metrics import (
    HISTOGRAM_SHADOW_SUFFIX,
    SERVE_METRICS,
)
from opentelemetry import metrics as otel_metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from prometheus_client import REGISTRY, generate_latest

_HELP = re.compile(r"^# HELP (?P<name>\S+) (?P<help>.*)$", re.MULTILINE)
_TYPE = re.compile(r"^# TYPE (?P<name>\S+) (?P<type>\S+)$", re.MULTILINE)
_EXPORTED_PREFIX = "maxserve_"


@dataclass(frozen=True)
class ExportedMetric:
    """One time series as a Prometheus scrape reports it."""

    name: str
    """The exported series name, including the unit and any ``_total``."""

    type: str
    """The Prometheus type, such as ``counter``, ``gauge`` or ``histogram``."""

    description: str
    """The text the exposition carries on the series' ``# HELP`` line."""


def _prime(name: str, instrument: Any) -> None:
    """Records one measurement so the instrument reaches a scrape.

    An instrument that never records produces no data point, and a metric
    with no data point is absent from the exposition entirely.

    Args:
        name: The registered name of the instrument, used in the error.
        instrument: The OpenTelemetry instrument to record into.

    Raises:
        RuntimeError: If the instrument offers no ``add``, ``record`` or
            ``set`` method, leaving no way to make it observable.
    """
    for method in ("add", "record", "set"):
        call = getattr(instrument, method, None)
        if callable(call):
            call(0)
            return
    raise RuntimeError(
        f"{name} offers no add/record/set, so a scrape cannot observe it. "
        "Teach _metrics_catalog how to prime this instrument type."
    )


def build_catalog() -> list[ExportedMetric]:
    """Returns every exported metric, read from a Prometheus scrape.

    Binds a meter provider and records into every registered instrument,
    so call this from a generator or a test rather than from a server: it
    mutates process-global OpenTelemetry state and writes a zero into each
    instrument. The exponential-histogram shadows stay out, since only the
    OTLP endpoint receives those.

    Returns:
        The exported metrics, sorted by series name.

    Raises:
        RuntimeError: If the scrape describes fewer metrics than the
            registry holds instruments bound for Prometheus, which means
            one recorded nothing and would vanish from the catalog.
    """
    otel_metrics.set_meter_provider(
        MeterProvider(metric_readers=[PrometheusMetricReader()])
    )
    for name, instrument in SERVE_METRICS.items():
        _prime(name, instrument)

    # Every histogram carries an exponential shadow for the OTLP endpoint.
    # The server's Prometheus reader drops those, so they are not part of
    # what a scrape of /metrics reports.
    shadow_suffix = HISTOGRAM_SHADOW_SUFFIX.replace(".", "_")

    exposition = generate_latest(REGISTRY).decode()
    types = {m["name"]: m["type"] for m in _TYPE.finditer(exposition)}
    catalog = [
        ExportedMetric(
            name=m["name"],
            type=types.get(m["name"], "unknown"),
            description=m["help"].strip(),
        )
        for m in _HELP.finditer(exposition)
        if m["name"].startswith(_EXPORTED_PREFIX)
        and not m["name"].endswith(shadow_suffix)
    ]
    exported = [
        name
        for name in SERVE_METRICS
        if not name.endswith(HISTOGRAM_SHADOW_SUFFIX)
    ]
    if len(catalog) < len(exported):
        raise RuntimeError(
            f"The scrape described {len(catalog)} metrics for "
            f"{len(exported)} instruments that reach Prometheus, so one of "
            "them recorded nothing and would vanish from the catalog."
        )
    return sorted(catalog, key=lambda metric: metric.name)
