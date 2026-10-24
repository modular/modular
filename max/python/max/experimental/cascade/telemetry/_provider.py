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
"""Per-process OpenTelemetry setup for cascade.

A process that records anything configures a ``MeterProvider`` at startup,
which today is the frontend: it is where the HTTP surface is observed.

Metrics reach Prometheus always, through a reader that registers into the
default ``prometheus_client`` registry, and OTLP when an endpoint is
configured.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable

from max.experimental.cascade.telemetry._metrics import LADDERS, Ladder
from max.support._telemetry import (
    SkipExponentialHistogramsPrometheusReader,
)
from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
    OTLPMetricExporter,
)
from opentelemetry.metrics import set_meter_provider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics._internal.aggregation import (
    ExplicitBucketHistogramAggregation,
)
from opentelemetry.sdk.metrics.export import (
    MetricReader,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.metrics.view import View
from opentelemetry.sdk.resources import Resource

logger = logging.getLogger(__name__)


# The standard OpenTelemetry variables, in the order the exporters resolve
# them. Worker subprocesses inherit the environment, which is how a single
# setting reaches every cascade process.
OTEL_EXPORTER_OTLP_METRICS_ENDPOINT = "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"
OTEL_EXPORTER_OTLP_ENDPOINT = "OTEL_EXPORTER_OTLP_ENDPOINT"


def otlp_metrics_endpoint() -> str | None:
    """Returns the OTLP metrics endpoint the environment configures, if any."""
    for var in (
        OTEL_EXPORTER_OTLP_METRICS_ENDPOINT,
        OTEL_EXPORTER_OTLP_ENDPOINT,
    ):
        value = os.environ.get(var)
        if value:
            return value
    return None


def configure_telemetry(
    service_name: str,
    endpoint: str,
    ladders: Iterable[Ladder] = LADDERS,
) -> None:
    """Configures this process's ``MeterProvider``.

    Call once per process, not idempotent since OpenTelemetry
    refuses the new providers and warns. Views are fixed at construction, so
    every histogram a process will record into has to be passed here.

    Args:
        service_name: Value for the ``service.name`` resource attribute, for
            example ``cascade.frontend``.
        endpoint: The address this process serves on, which is what identifies
            it within a deployment. Recorded as a resource attribute, so on
            OTLP it says which process produced a metric.
        ladders: The histograms to render with their own boundaries,
            defaulting to every one cascade defines. A histogram left out
            falls back to the OTEL SDK defaults, which are milliseconds and
            put every cascade latency in a single bucket.
    """
    attributes = {
        "service.name": service_name,
        "cascade.endpoint": endpoint,
        # OTel's standard key for one instance of a service, so a backend
        # can group by it without knowing cascade.
        "service.instance.id": endpoint,
    }
    resource = Resource.create(attributes)
    otlp_endpoint = otlp_metrics_endpoint()

    readers: list[MetricReader] = [
        SkipExponentialHistogramsPrometheusReader(True)
    ]
    if otlp_endpoint:
        readers.append(PeriodicExportingMetricReader(OTLPMetricExporter()))
    set_meter_provider(
        MeterProvider(
            metric_readers=readers,
            resource=resource,
            views=[
                View(
                    instrument_name=ladder.name,
                    aggregation=ExplicitBucketHistogramAggregation(
                        ladder.buckets
                    ),
                )
                for ladder in ladders
            ],
        )
    )

    logger.debug(
        "Cascade telemetry configured for %s (%s), OTLP %s",
        service_name,
        endpoint,
        "enabled" if otlp_endpoint else "disabled",
    )
