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

"""Parser for Prometheus metrics from different serving backends."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from prometheus_client.parser import text_string_to_metric_families

from .backend_metric_names import (
    BATCH_METRIC_NAMES,
    BatchPhase,
    MetricAvailability,
    engine_family,
    resolve_batch_metric,
    resolve_spec_decode_metric,
)
from .config import Backend
from .metrics import HistogramMetric, SpecDecodeMetrics
from .prometheus_fetch import fetch_metrics

if TYPE_CHECKING:
    from prometheus_client.metrics_core import Metric

logger = logging.getLogger(__name__)


class MissingEngineMetricsError(RuntimeError):
    """A successful scrape lacked a metric the backend always emits.

    Raised by :func:`collect_benchmark_metrics` (and surfaced by the
    benchmark run) instead of silently nulling every engine-side
    ``BenchmarkResult`` field. A failed scrape keeps the warning behavior.
    """


# Import MAX metrics port from SDK config, with fallback
try:
    from max.serve.config import Settings

    _MAX_METRICS_PORT = Settings().metrics_port
except Exception:
    _MAX_METRICS_PORT = 8001  # Fallback if SDK not available


@dataclass
class HistogramData:
    """Histogram data with buckets, sum and count."""

    buckets: list[tuple[str, float]]
    """List of (upper_bound, cumulative_count) tuples representing histogram buckets."""
    sum: float
    """Total sum of all observed values."""
    count: float
    """Total number of observations."""

    def __bool__(self) -> bool:
        """Check if histogram has any observations.

        Returns:
            True if count > 0, False otherwise

        Note:
            A histogram object is considered falsy when it is empty or does not contain data.
        """
        return bool(self.count)

    @property
    def mean(self) -> float:
        """Calculate mean from sum and count.

        Returns:
            Mean value, or 0.0 if count is 0
        """
        return self.sum / self.count if self.count > 0 else 0.0


@dataclass
class ParsedMetrics:
    """Structured metrics data parsed from Prometheus endpoint.

    Note:
        Metrics with labels are stored with keys in the format:
        "metric_name{label1="value1",label2="value2"}"

        Use get_histogram() for convenient access to labeled histograms.
    """

    counters: dict[str, float]
    """Dictionary of counter metrics (monotonically increasing values)."""
    gauges: dict[str, float]
    """Dictionary of gauge metrics (can increase or decrease)."""
    histograms: dict[str, HistogramData]
    """Dictionary of histogram metrics with bucket distributions."""
    raw_text: str
    """Raw Prometheus text format for debugging."""
    backend: Backend | None = None
    """Backend these metrics were scraped from.

    Set by the collection helpers (:func:`collect_benchmark_metrics` /
    :func:`fetch_and_parse_metrics`) so consumers can resolve metric names
    and ``batch_type`` label values per engine through
    ``backend_metric_names``. ``None`` (metrics built directly) keeps the
    MAX Serve names.
    """

    def get_histogram(
        self, metric_name: str, labels: dict[str, str] | None = None
    ) -> HistogramData | None:
        """Get a histogram by name and optional labels.

        Args:
            metric_name: Base metric name (such as "maxserve_batch_execution_time_milliseconds")
            labels: Optional dictionary of labels (such as {"batch_type": "CE"})

        Returns:
            HistogramData if found, None otherwise

        Examples:
            >>> # Get histogram without labels
            >>> hist = metrics.get_histogram("maxserve_batch_size")
            >>>
            >>> # Get prefill batch execution time
            >>> prefill = metrics.get_histogram(
            ...     "maxserve_batch_execution_time_milliseconds", {"batch_type": "CE"}
            ... )
            >>> if prefill:
            ...     print(f"Prefill avg: {prefill.mean:.2f} ms")
            >>>
            >>> # Get decode batch execution time
            >>> decode = metrics.get_histogram(
            ...     "maxserve_batch_execution_time_milliseconds", {"batch_type": "TG"}
            ... )
        """
        key = _format_metric_key(metric_name, labels or {})
        return self.histograms.get(key)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dict."""
        return {
            "counters": self.counters,
            "gauges": self.gauges,
            "histograms": {
                name: {
                    "buckets": hist.buckets,
                    "sum": hist.sum,
                    "count": hist.count,
                    "mean": hist.mean,
                }
                for name, hist in self.histograms.items()
            },
        }


def get_metrics_url(backend: Backend, base_url: str) -> str:
    """Get the metrics URL for a backend.

    Args:
        backend: Backend name (Backend enum)
        base_url: Base API URL (such as 'http://localhost:8000')

    Returns:
        Metrics endpoint URL for the backend

    Raises:
        ValueError: If backend is not supported
    """
    parsed_url = urlparse(base_url)
    host = parsed_url.hostname or "localhost"
    # Re-bracket IPv6 addresses stripped by urlparse().hostname (RFC 3986).
    if ":" in host:
        host = f"[{host}]"

    # MAX engines expose Prometheus metrics on a dedicated port: MAX Serve
    # via its own metrics server and the Mach engine via its
    # ``--metrics-address`` flag (the benchmark CI passes the same port the
    # SDK config uses; see benchmarkServingRun.yaml). Other backends are
    # scraped on the same port as the base URL.
    if backend in ("modular", "modular-chat", "mach"):
        metrics_port = _MAX_METRICS_PORT
    else:
        metrics_port = parsed_url.port or 8000

    return f"http://{host}:{metrics_port}/metrics"


def _format_metric_key(metric_name: str, labels: dict[str, str]) -> str:
    """Format a metric key with labels in Prometheus style.

    Args:
        metric_name: Base metric name
        labels: Dictionary of label key-value pairs

    Returns:
        Formatted metric key like "metric_name{label1=value1,label2=value2}"
        or just "metric_name" if no labels
    """
    if not labels:
        return metric_name

    label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    return f"{metric_name}{{{label_str}}}"


def _extract_simple_metrics(
    family: Metric, metric_name: str
) -> dict[str, float]:
    """Extract counter or gauge metrics from a family.

    Helper function to handle label processing for simple metric types.

    Args:
        family: Prometheus metric family
        metric_name: Base metric name

    Returns:
        Dictionary mapping metric names (with labels if present) to values
    """
    result = {}
    for sample in family.samples:
        key = _format_metric_key(metric_name, sample.labels)
        result[key] = sample.value
    return result


def parse_metrics(raw_text: str) -> ParsedMetrics:
    """Parse Prometheus metrics from raw text.

    Parses metrics into structured format with counters, gauges, and histograms.
    Handles metric labels by appending them to metric names.

    Args:
        raw_text: Raw Prometheus text format

    Returns:
        ParsedMetrics object containing all parsed metrics

    Note:
        - Counters: Monotonically increasing values (such as total requests)
        - Gauges: Values that can increase or decrease (such as current memory usage)
        - Histograms: Distribution data with buckets, sum, and count
    """
    counters: dict[str, float] = {}
    gauges: dict[str, float] = {}
    histograms: dict[str, HistogramData] = {}

    # Parse Prometheus text format
    for family in text_string_to_metric_families(raw_text):
        metric_name = family.name
        if family.type == "counter":
            counters.update(_extract_simple_metrics(family, metric_name))

        elif family.type == "gauge":
            gauges.update(_extract_simple_metrics(family, metric_name))

        elif family.type == "histogram":
            # Group histogram samples by their label set (excluding 'le')
            histogram_groups: dict[str, HistogramData] = defaultdict(
                lambda: HistogramData(buckets=[], sum=0.0, count=0.0)
            )

            for sample in family.samples:
                # Get labels excluding 'le' for grouping
                grouping_labels = {
                    k: v for k, v in sample.labels.items() if k != "le"
                }
                key = _format_metric_key(metric_name, grouping_labels)

                if sample.name.endswith("_bucket"):
                    upper_bound = sample.labels.get(
                        "le", ""
                    )  # `le` = less than equal to (upper_bound)
                    histogram_groups[key].buckets.append(
                        (upper_bound, sample.value)
                    )
                elif sample.name.endswith("_sum"):
                    histogram_groups[key].sum = sample.value
                elif sample.name.endswith("_count"):
                    histogram_groups[key].count = sample.value

            # Store the histogram data
            histograms.update(histogram_groups)

    return ParsedMetrics(
        counters=counters,
        gauges=gauges,
        histograms=histograms,
        raw_text=raw_text,
    )


def fetch_and_parse_metrics(backend: Backend, base_url: str) -> ParsedMetrics:
    """Fetch and parse metrics for a backend.

    Convenience function that combines get_metrics_url, fetch_metrics, and parse_metrics.

    Args:
        backend: Backend name (Backend enum)
        base_url: Base API URL (such as 'http://localhost:8000')

    Returns:
        ParsedMetrics object containing all parsed metrics

    Raises:
        ValueError: If backend is not supported
        requests.HTTPError: If the response status is not 200
        requests.RequestException: For network/connection errors
    """
    metrics_url = get_metrics_url(backend, base_url)
    raw_text = fetch_metrics(metrics_url)
    parsed = parse_metrics(raw_text)
    parsed.backend = backend
    return parsed


def compute_metrics_delta(
    baseline: ParsedMetrics, final: ParsedMetrics
) -> ParsedMetrics:
    """Compute the difference between two ParsedMetrics objects.

    For counters and histograms (monotonically increasing), compute the delta.
    For gauges, use the final value (as they represent current state).

    Args:
        baseline: Metrics captured at the start of the benchmark
        final: Metrics captured at the end of the benchmark

    Returns:
        ParsedMetrics object containing the deltas
    """
    # Compute delta for counters (monotonically increasing)
    delta_counters = {}
    for name, final_value in final.counters.items():
        baseline_value = baseline.counters.get(name, 0.0)
        delta_counters[name] = final_value - baseline_value

    # For gauges, use final value (they represent current state, not cumulative)
    delta_gauges = dict(final.gauges)

    # Compute delta for histograms (sum and count are monotonically increasing)
    delta_histograms = {}
    for name, final_hist in final.histograms.items():
        baseline_hist = baseline.histograms.get(
            name, HistogramData(buckets=[], sum=0.0, count=0.0)
        )
        delta_sum = final_hist.sum - baseline_hist.sum
        delta_count = final_hist.count - baseline_hist.count

        # Compute delta buckets
        delta_buckets = []
        baseline_buckets_dict = defaultdict(float, baseline_hist.buckets)
        for upper_bound, final_count in final_hist.buckets:
            baseline_count = baseline_buckets_dict[upper_bound]
            delta_buckets.append((upper_bound, final_count - baseline_count))

        delta_histograms[name] = HistogramData(
            buckets=delta_buckets,
            sum=delta_sum,
            count=delta_count,
        )

    return ParsedMetrics(
        counters=delta_counters,
        gauges=delta_gauges,
        histograms=delta_histograms,
        raw_text="",  # Delta doesn't have raw text
        backend=final.backend,
    )


def collect_benchmark_metrics(
    urls: Mapping[str, str],
    backend: Backend,
    base_url: str,
    baseline: Mapping[str, ParsedMetrics] | None = None,
    verify_required_metrics: bool = False,
) -> dict[str, ParsedMetrics]:
    """Scrape Prometheus metrics from the given endpoints.

    When *urls* is empty, a single endpoint is synthesised from *backend* and
    *base_url* via :func:`get_metrics_url` and stored under the ``"server"``
    label, preserving the auto-derive behaviour that existed before explicit
    URLs were supported.

    When *baseline* is provided and contains a matching label, the returned
    metrics for that label are the delta from baseline.

    Args:
        urls: Explicit metrics endpoint URLs, keyed by label.
        backend: Backend the endpoints belong to.
        base_url: Base API URL, used to derive the metrics URL when *urls* is
            empty.
        baseline: Optional earlier snapshot to compute deltas against.
        verify_required_metrics: Verify each successful scrape contains the
            metrics this backend always emits (only for the final collection,
            see below).

    Raises:
        MissingEngineMetricsError: When a scrape succeeds but lacks a metric
            the backend always emits (see
            :func:`verify_always_emitted_metrics`). A scrape that itself
            fails stays a logged warning with the endpoint dropped.

    Verification runs only when the caller asks for it, never on a baseline
    collection. Mach exports through the OpenTelemetry Prometheus exporter,
    which omits a histogram until its first data point, so a fresh idle
    engine legitimately has no batch families yet at baseline time; the final
    scrape runs after the benchmark and must have them.
    """
    if not urls:
        urls = {"server": get_metrics_url(backend, base_url)}

    results: dict[str, ParsedMetrics] = {}
    for label, url in urls.items():
        try:
            raw_text = fetch_metrics(url)
            final = parse_metrics(raw_text)
            final.backend = backend
            if verify_required_metrics:
                verify_always_emitted_metrics(backend, final, url)
            if baseline and label in baseline:
                final = compute_metrics_delta(
                    baseline=baseline[label], final=final
                )
            results[label] = final
        except MissingEngineMetricsError:
            # A reachable endpoint without the metrics this backend always
            # emits nulls every engine-side field; fail the run loudly rather
            # than warn (the classic cause is scraping the wrong server).
            raise
        except Exception as exc:
            logger.warning(
                "Failed to collect metrics from %s (%s): %s", label, url, exc
            )
    return results


def verify_always_emitted_metrics(
    backend: Backend, parsed: ParsedMetrics, metrics_url: str
) -> None:
    """Check a successful scrape contains the metrics it always should.

    Only the always-emitted batch metrics count (see
    ``backend_metric_names``): for mach, every constructed batch records
    ``mach_batch_size`` and every executed batch records
    ``mach_batch_execution_time_milliseconds``. Optional families (the
    spec-decode histograms when speculative decoding is off) and metrics
    declared unavailable never fail a run.

    Args:
        backend: Backend the metrics were scraped from.
        parsed: Metrics parsed from a successful scrape of *metrics_url*.
        metrics_url: The endpoint that was scraped, named in the error.

    Raises:
        MissingEngineMetricsError: When an always-emitted metric is absent
            from a successful scrape (usually a wrong-endpoint problem).
    """
    family = engine_family(backend)
    missing: list[str] = []
    for per_family in BATCH_METRIC_NAMES.values():
        spec = per_family[family]
        if (
            spec.availability is not MetricAvailability.ALWAYS_EMITTED
            or spec.name is None
        ):
            continue
        if not any(
            key == spec.name or key.startswith(f"{spec.name}{{")
            for key in parsed.histograms
        ):
            missing.append(spec.name)
    if missing:
        names = ", ".join(f"'{name}'" for name in missing)
        raise MissingEngineMetricsError(
            f"Scrape of {metrics_url} succeeded, but required engine "
            f"metric(s) {names} on backend '{backend}' are missing from "
            f"the payload. Backend '{backend}' always emits them while "
            f"serving, so this is usually a wrong-endpoint problem: check "
            f"that {metrics_url} is the engine's /metrics endpoint and not "
            f"another server's."
        )


_POSITION_LABEL_RE = re.compile(r'position="([^"]*)"')


def _extract_position(metric_key: str) -> int | None:
    """Extract the ``position`` label value from a formatted metric key."""
    match = _POSITION_LABEL_RE.search(metric_key)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def parse_spec_decode_metrics(
    raw_text: str, backend: Backend | None = None
) -> SpecDecodeMetrics | None:
    """Parse speculative decoding metrics from Prometheus text output.

    Metric names are resolved per backend through ``backend_metric_names``:

    - vLLM-style counters under ``vllm:spec_decode_*`` (not emitted by mach;
      declared unavailable there).
    - Histogram ``maxserve_spec_decode_acceptance_rate_per_position`` on MAX
      Serve, ``mach_spec_decode_acceptance_rate_per_position`` on mach
      (per-position acceptance-rate observations, 0--100).
    - Histogram ``maxserve_spec_decode_avg_acceptance_length`` on MAX Serve,
      ``mach_spec_decode_avg_acceptance_length`` on mach (batch mean
      acceptance length in tokens).

    Args:
        raw_text: Raw Prometheus text-format payload.
        backend: Backend the payload was scraped from. ``None`` keeps the
            MAX Serve names.

    Returns:
        Parsed metrics, or ``None`` when no spec-decode families are present.
    """
    parsed = parse_metrics(raw_text)

    draft_counters = resolve_spec_decode_metric(backend, "draft_counters")
    acceptance_rate = resolve_spec_decode_metric(
        backend, "acceptance_rate_per_position"
    )
    avg_acceptance_length = resolve_spec_decode_metric(
        backend, "avg_acceptance_length"
    )

    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    accepted_per_pos: dict[int, int] = {}
    per_pos_rate_sum: dict[int, float] = {}
    per_pos_rate_count: dict[int, int] = {}
    found = False
    avg_acceptance_length_sum = 0.0
    avg_acceptance_length_count = 0.0

    if (
        draft_counters.availability is not MetricAvailability.UNAVAILABLE
        and draft_counters.name is not None
    ):
        for key, value in parsed.counters.items():
            if not key.startswith(draft_counters.name):
                continue
            found = True
            if "num_accepted_tokens_per_pos" in key:
                pos = _extract_position(key)
                if pos is not None:
                    accepted_per_pos[pos] = accepted_per_pos.get(pos, 0) + int(
                        value
                    )
            elif "num_drafts" in key:
                num_drafts += int(value)
            elif "num_draft_tokens" in key:
                num_draft_tokens += int(value)
            elif "num_accepted_tokens" in key:
                num_accepted_tokens += int(value)

    if acceptance_rate.name is not None:
        for key, hist in parsed.histograms.items():
            if not key.startswith(acceptance_rate.name):
                continue
            pos = _extract_position(key)
            if pos is None:
                continue
            found = True
            per_pos_rate_sum[pos] = per_pos_rate_sum.get(pos, 0.0) + hist.sum
            per_pos_rate_count[pos] = per_pos_rate_count.get(pos, 0) + int(
                hist.count
            )

    al_hist = (
        parsed.histograms.get(avg_acceptance_length.name)
        if avg_acceptance_length.name is not None
        else None
    )
    if al_hist is not None and al_hist.count > 0:
        found = True
        avg_acceptance_length_sum = al_hist.sum
        avg_acceptance_length_count = al_hist.count

    if not found:
        return None

    return SpecDecodeMetrics(
        num_drafts=num_drafts,
        num_draft_tokens=num_draft_tokens,
        num_accepted_tokens=num_accepted_tokens,
        accepted_per_pos=accepted_per_pos,
        per_pos_rate_sum=per_pos_rate_sum,
        per_pos_rate_count=per_pos_rate_count,
        avg_acceptance_length_sum=avg_acceptance_length_sum,
        avg_acceptance_length_count=avg_acceptance_length_count,
    )


def fetch_spec_decode_metrics(
    backend: Backend,
    base_url: str,
    metrics_urls: Mapping[str, str] | None = None,
) -> SpecDecodeMetrics | None:
    """Fetch and merge speculative decoding metrics from Prometheus endpoints.

    Returns ``None`` when no endpoint exposes speculative decoding metrics.

    Args:
        backend: Backend type (``vllm`` / ``vllm-chat``, ``modular`` /
            ``modular-chat`` and so on).
        base_url: Server base URL (such as ``http://localhost:8000``).
        metrics_urls: Explicit Prometheus metrics endpoint URLs, keyed by
            label. When empty, a single endpoint is derived from *backend*
            and *base_url*.
    """
    urls = metrics_urls or {"server": get_metrics_url(backend, base_url)}

    merged: SpecDecodeMetrics | None = None
    for label, url in urls.items():
        try:
            metrics_text = fetch_metrics(url)
        except Exception:
            logger.warning(
                "Failed to fetch spec-decode metrics from %s (%s)", label, url
            )
            continue
        parsed = parse_spec_decode_metrics(metrics_text, backend)
        if parsed is None:
            continue
        if merged is None:
            merged = parsed
        else:
            merged += parsed
    return merged


def print_server_metrics(metrics: ParsedMetrics) -> None:
    """Print server-side metrics in a formatted way.

    Args:
        metrics: ParsedMetrics object containing counters, gauges, and histograms
    """
    # Print section header
    print(
        "{s:{c}^{n}}".format(s="Server Metrics (from Prometheus)", n=50, c="=")
    )

    # Print counters
    if metrics.counters:
        print("\nCounters:")
        for name, value in sorted(metrics.counters.items()):
            print(f"  {name}: {value}")

    # Print gauges
    if metrics.gauges:
        print("\nGauges:")
        for name, value in sorted(metrics.gauges.items()):
            print(f"  {name}: {value}")

    # Print histogram metrics
    if metrics.histograms:
        print("\nHistograms:")
        for metric_name, hist in sorted(metrics.histograms.items()):
            if hist.count > 0:
                print(f"\n  {metric_name}:")
                print(f"    Count: {hist.count}")
                print(f"    Sum: {hist.sum:.2f}")
                print(f"    Mean: {hist.mean:.2f}")

        # Special handling for batch_execution_time - show prefill/decode
        # breakdown, resolving the metric name and batch_type labels for the
        # scraped backend (MAX Serve CE/TG, mach prefill/decode).
        prefill_resolved = resolve_batch_metric(
            metrics.backend,
            HistogramMetric.MAXSERVE_BATCH_EXECUTION_TIME_MILLISECONDS.value,
            BatchPhase.PREFILL,
        )
        decode_resolved = resolve_batch_metric(
            metrics.backend,
            HistogramMetric.MAXSERVE_BATCH_EXECUTION_TIME_MILLISECONDS.value,
            BatchPhase.DECODE,
        )
        prefill = (
            metrics.get_histogram(
                prefill_resolved.name, dict(prefill_resolved.labels)
            )
            if prefill_resolved
            else None
        )
        decode = (
            metrics.get_histogram(
                decode_resolved.name, dict(decode_resolved.labels)
            )
            if decode_resolved
            else None
        )

        if prefill or decode:
            print("\n  Batch Execution Time Breakdown:")
            if prefill and prefill_resolved:
                print(
                    f"    Mean Prefill ({prefill_resolved.labels['batch_type']})"
                    f" Time: {prefill.mean:.2f} ms "
                    f"(count={int(prefill.count)}, sum={prefill.sum:.2f} ms)"
                )
            if decode and decode_resolved:
                print(
                    f"    Mean Decode ({decode_resolved.labels['batch_type']})"
                    f" Time:  {decode.mean:.2f} ms "
                    f"(count={int(decode.count)}, sum={decode.sum:.2f} ms)"
                )

    print("=" * 50)
