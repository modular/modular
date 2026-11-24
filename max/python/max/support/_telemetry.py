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
"""Telemetry plumbing shared by ``max.serve`` and cascade.

Both build the same kind of histogram ladder and both export OTEL metrics to a
Prometheus endpoint and OTLP at once, so both need the exporter workaround
below.

Here rather than in ``max.serve.telemetry.common`` for two reasons. An
architecture module may reference a cascade pipeline, and
``max/tests/tests/cascade/test_arch_import_layering.py`` guards that such an
import never reaches ``max.serve``. Importing that module also pulls in
``max.serve.telemetry.metrics``, which builds every ``SERVE_METRICS``
instrument at module scope: measured at 0.69 s and 829 modules.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable

from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics._internal.point import (
    ExponentialHistogram,
    Metric,
)
from opentelemetry.sdk.metrics.export import MetricsData


def log_spaced_buckets(
    start: float, stop: float, max_ratio: float
) -> tuple[float, ...]:
    """Builds a 0 boundary plus geometrically spaced boundaries up to ``stop``.

    Args:
        start: First non-zero boundary.
        stop: Last boundary.
        max_ratio: Largest ratio allowed between neighbours, which sets how
            many boundaries the range needs.

    Returns:
        Ascending boundaries, opening at 0 and closing at ``stop``, each
        rounded to three significant figures.
    """
    steps = math.ceil(math.log(stop / start) / math.log(max_ratio))
    log_start, log_stop = math.log(start), math.log(stop)
    return (0.0,) + tuple(
        float(f"{math.exp(log_start + (log_stop - log_start) * i / steps):.3g}")
        for i in range(steps + 1)
    )


DEFAULT_BUCKET_RATIO = 1.5
"""Largest ratio between neighbouring boundaries a ladder may use.

Every boundary is 50% above the last, so a percentile resolves to the same
relative precision wherever it lands. Ladders whose range is wide and whose
consumers are coarse (bytes, GiB/s) pass a larger ratio of their own.
"""

LATENCY_BUCKETS_MS: tuple[float, ...] = log_spaced_buckets(
    1.0, 8 * 60 * 60 * 1000.0, DEFAULT_BUCKET_RATIO
)
"""Latency ladder, 1ms out to 8 hours.

Covers slow ops (request time, model load, compile) and long agentic requests.
Shared so every stack's latency percentiles mean the same thing; ``max.serve``
records milliseconds and cascade seconds, hence the pair.
"""

LATENCY_BUCKETS_S: tuple[float, ...] = tuple(
    float(f"{boundary / 1000:.3g}") for boundary in LATENCY_BUCKETS_MS
)
""":py:data:`LATENCY_BUCKETS_MS` in seconds, rounded as it rounds."""


def drop_metrics_matching(
    metrics_data: MetricsData | None,
    should_drop: Callable[[Metric], bool],
) -> MetricsData | None:
    """Rebuilds the MetricsData tree without metrics matching ``should_drop``.

    MetricsData/ResourceMetrics/ScopeMetrics are frozen dataclasses, so this
    reconstructs the tree rather than mutating it in place.
    """
    if metrics_data is None:
        return None
    return dataclasses.replace(
        metrics_data,
        resource_metrics=[
            dataclasses.replace(
                resource_metrics,
                scope_metrics=[
                    dataclasses.replace(
                        scope_metrics,
                        metrics=[
                            m
                            for m in scope_metrics.metrics
                            if not should_drop(m)
                        ],
                    )
                    for scope_metrics in resource_metrics.scope_metrics
                ],
            )
            for resource_metrics in metrics_data.resource_metrics
        ],
    )


class SkipExponentialHistogramsPrometheusReader(PrometheusMetricReader):
    """Drops exponential-histogram data points instead of crashing on them.

    The classic Prometheus text exposition format (and this exporter) only
    knows how to render explicit-bucket histograms:
    ``_translate_to_prometheus`` assumes every non-``HistogramDataPoint`` has
    a ``.value`` attribute, which ``ExponentialHistogramDataPoint`` doesn't
    have, so it would raise ``AttributeError`` and 500 the whole ``/metrics``
    scrape. The exponential-histogram shadow metrics
    (metrics.HISTOGRAM_SHADOW_SUFFIX) only exist when
    ``MAX_SERVE_OTLP_METRICS_ENDPOINT`` is set, and are meant for that
    endpoint only (see _ExponentialShadowOnlyReader below) — this keeps
    Prometheus showing exactly the explicit-bucket histograms it always has,
    unaffected either way.
    """

    def _receive_metrics(
        self,
        metrics_data: MetricsData,
        timeout_millis: float = 10_000,
        **kwargs: object,
    ) -> None:
        filtered = drop_metrics_matching(
            metrics_data, lambda m: isinstance(m.data, ExponentialHistogram)
        )
        if filtered is not None:
            super()._receive_metrics(filtered, timeout_millis, **kwargs)
