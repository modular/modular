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
"""Pins the ``/metrics`` line names of the ``maxserve.media.*`` family.

The Prometheus exporter mangles an instrument's name with its ``unit``: it
appends ``_<unit>`` unless the name already ends in it, and ``_total`` to a
counter. So the name an operator writes a query against is not the name in
``SERVE_METRICS``, and the published documentation has to be copied from a
real scrape rather than predicted. This test is that scrape.

**This has to be its own file.** OTel refuses a second
``set_meter_provider`` and ``PrometheusMetricReader`` registers into the
process-global ``prometheus_client.REGISTRY``, so a provider installed after
another test's would export nothing and every name below would be compared
against an empty scrape -- a vacuous pass. The bazel target per source file
makes this the first provider in its process, and
``test_scrape_is_not_empty`` makes the vacuous case fail loudly anyway.
"""

from __future__ import annotations

import prometheus_client
import pytest
from max.serve.config import Settings
from max.serve.telemetry import common, metrics

# One measurement per new instrument, with a representative attribute set.
_MEDIA_MEASUREMENTS: tuple[tuple[str, float, dict[str, str]], ...] = (
    ("maxserve.media.items", 1, {"media_kind": "image", "source": "inline"}),
    ("maxserve.media.resolve_time", 12.5, {"source": "inline"}),
    ("maxserve.media.item_size", 4096, {"media_kind": "image"}),
    ("maxserve.media.image_size", 65536, {"format": "png"}),
    ("maxserve.media.image_decodes", 1, {"format": "png"}),
    ("maxserve.media.image_decode_ms", 3.25, {"format": "png"}),
    ("maxserve.media.rejections", 1, {"reason": "undecodable"}),
    ("maxserve.media.items_per_request", 2, {"media_kind": "image"}),
    (
        "maxserve.media.preprocess_cache_evictions",
        1,
        {"media_kind": "image"},
    ),
    ("maxserve.media.preprocess_cache_size", 4096, {"media_kind": "image"}),
    (
        "maxserve.media.preprocess_cache_capacity",
        8192,
        {"media_kind": "image"},
    ),
)

# Exactly what a scrape must show. Every name published in
# ``docs/max/serve/metrics.mdx`` is copied from this list.
_EXPECTED_MEDIA_LINES = frozenset(
    {
        "maxserve_media_items_total",
        "maxserve_media_resolve_time_milliseconds",
        "maxserve_media_item_size_bytes",
        "maxserve_media_image_size_pixels",
        "maxserve_media_image_decodes_total",
        "maxserve_media_image_decode_ms_total",
        "maxserve_media_rejections_total",
        "maxserve_media_items_per_request_items",
        "maxserve_media_preprocess_cache_evictions_total",
        "maxserve_media_preprocess_cache_size_bytes",
        "maxserve_media_preprocess_cache_capacity_bytes",
    }
)

_HISTOGRAM_SUFFIXES = ("_bucket", "_sum", "_count", "_created")


@pytest.fixture(scope="module")
def media_scrape() -> str:
    common.configure_metrics(Settings())
    for name, value, attributes in _MEDIA_MEASUREMENTS:
        metrics.MaxMeasurement(name, value, attributes).commit()
    return prometheus_client.generate_latest().decode()


def _line_names(scrape: str, prefix: str) -> set[str]:
    """The distinct sample-line names under ``prefix``, histogram parts fused.

    Reads the sample lines rather than the ``# TYPE`` headers, because
    ``prometheus_client`` strips ``_total`` from a counter's family name while
    the line an operator queries keeps it.
    """
    names = set()
    for line in scrape.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name = line.partition(" ")[0].split("{", 1)[0]
        if not name.startswith(prefix):
            continue
        for suffix in _HISTOGRAM_SUFFIXES:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        names.add(name)
    return names


def test_scrape_is_not_empty(media_scrape: str) -> None:
    """Guards every other assertion in this file against a vacuous pass."""
    assert _line_names(media_scrape, "maxserve_"), (
        "no maxserve_ series on the scrape at all -- the meter provider this"
        " test installed is not the one the instruments are bound to, so the"
        " name assertions below would pass against an empty string"
    )


def test_media_metrics_export_under_their_documented_names(
    media_scrape: str,
) -> None:
    assert _line_names(media_scrape, "maxserve_media") == (
        _EXPECTED_MEDIA_LINES
    )
