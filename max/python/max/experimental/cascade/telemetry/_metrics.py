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
"""Cascade metrics.

Names are OTEL dot-names with a declared unit, and the Prometheus exporter
derives the exposition name from both: ``cascade.http.request_time`` with
``unit="s"`` surfaces as ``cascade_http_request_time_seconds``.

One instrument per thing measured, with the route as an attribute rather than
part of the name, so a question asked of one route and the same question asked
of the whole frontend are the same query with and without a matcher. A ``View``
is matched by instrument name, so this also means every route shares one set of
bucket boundaries; the ladder is geometric and therefore holds its relative
precision at any scale, which is what makes that shareable.
"""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Callable
from typing import Any, TypeVar

from max.support._telemetry import LATENCY_BUCKETS_S
from opentelemetry import metrics
from prometheus_client import generate_latest as generate_latest

_meter = metrics.get_meter("max.experimental.cascade")


@dataclasses.dataclass(frozen=True, slots=True)
class Ladder:
    """A histogram and the bucket boundaries it must render with.

    The boundaries travel with the instrument because a ``View`` is the only
    way to set them and is matched by instrument name, so whoever builds the
    provider needs both.
    """

    name: str
    description: str
    buckets: tuple[float, ...]
    unit: str = "s"
    histogram: metrics.Histogram = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "histogram",
            _meter.create_histogram(
                self.name, unit=self.unit, description=self.description
            ),
        )


# All three take max.serve's latency ladder, as its own 25 latency metrics do
# -- inter-token latency and time to first token among them. A narrower span
# would be the same boundaries with the top cut off, since a geometric ladder
# of one ratio and one floor cannot diverge from another; only a finer ratio
# would buy resolution, and none of these is read closely enough to want one.
REQUEST_TIME = Ladder(
    "cascade.http.request_time",
    "End-to-end time serving an HTTP request.",
    LATENCY_BUCKETS_S,
)

TIME_TO_FIRST_CHUNK = Ladder(
    "cascade.http.time_to_first_chunk",
    (
        "Time from receiving a request to the first streamed body message of "
        "its response, same origin as request_time. Streamed responses only."
    ),
    LATENCY_BUCKETS_S,
)

INTER_CHUNK_TIME = Ladder(
    "cascade.http.inter_chunk_time",
    "Gap between successive body messages of a response, as delivered.",
    LATENCY_BUCKETS_S,
)

LADDERS = (REQUEST_TIME, TIME_TO_FIRST_CHUNK, INTER_CHUNK_TIME)
"""Every histogram the HTTP layer records into."""


HTTP_INFLIGHT = _meter.create_up_down_counter(
    "cascade.http.num_requests_running",
    description=(
        "Requests to measured endpoints currently being handled by this "
        "process."
    ),
)


METRICS_PATH = "/metrics"
"""Path the frontend serves its own Prometheus exposition on."""


_MEASURED_ATTR = "_cascade_measured"

Endpoint = TypeVar("Endpoint", bound=Callable[..., Any])


def measured(endpoint: Endpoint) -> Endpoint:
    """Marks an endpoint for the HTTP middleware to time.

    Applied below the routing decorator, so the function the router registers
    is the one carrying the mark::

        @router.post("/v1/chat/completions")
        @measured
        async def chat_completions(...): ...
    """
    setattr(endpoint, _MEASURED_ATTR, True)
    return endpoint


def is_measured(endpoint: object) -> bool:
    """Returns whether an endpoint was marked with :py:func:`measured`."""
    return getattr(endpoint, _MEASURED_ATTR, False)


class Outcome(
    str, enum.Enum
):  # TODO(stan): switch to :py:class:`~enum.StrEnum` for Python 3.11+
    """How a request ended, recorded as the ``outcome`` attribute.

    Record ``outcome.value``, never the member: a ``(str, Enum)`` member
    stringifies as ``"Outcome.OK"``.

    Recorded instead of the status code, which cannot answer the question at
    this layer. A streamed response commits to its code before its body is
    produced, so a stream that dies halfway is a 200; and the middleware runs
    inside starlette's ``ServerErrorMiddleware``, so a handler that raises
    sends no status line at all. These describe the whole response.

    Client and server errors stay apart because they are different problems:
    a rising ``server_error`` rate is ours, a rising ``client_error`` rate is
    the caller's, and an error-rate alert usually wants only the first.
    """

    OK = "ok"
    CLIENT_ERROR = "client_error"
    SERVER_ERROR = "server_error"
    CANCELLED = "cancelled"
