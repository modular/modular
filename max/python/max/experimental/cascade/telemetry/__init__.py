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
"""Metrics cascade records, and the per-process OpenTelemetry setup for them.

An add-on applied by whoever serves cascade rather than part of its core
interfaces, so nothing under ``cascade.core`` imports it.
"""

from max.experimental.cascade.telemetry._metrics import (
    HTTP_INFLIGHT,
    INTER_CHUNK_TIME,
    LADDERS,
    METRICS_PATH,
    REQUEST_TIME,
    TIME_TO_FIRST_CHUNK,
    Ladder,
    Outcome,
    generate_latest,
    is_measured,
    measured,
)
from max.experimental.cascade.telemetry._provider import (
    OTEL_EXPORTER_OTLP_METRICS_ENDPOINT,
    configure_telemetry,
    otlp_metrics_endpoint,
)

__all__ = [
    "HTTP_INFLIGHT",
    "INTER_CHUNK_TIME",
    "LADDERS",
    "METRICS_PATH",
    "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
    "REQUEST_TIME",
    "TIME_TO_FIRST_CHUNK",
    "Ladder",
    "Outcome",
    "configure_telemetry",
    "generate_latest",
    "is_measured",
    "measured",
    "otlp_metrics_endpoint",
]
