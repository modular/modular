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

"""Per-backend Prometheus names for the engine metrics the client reads.

MAX Serve and the Mach engine expose the same concepts under different
metric names and different ``batch_type`` label values, so the client
resolves every engine metric through the tables here instead of
hard-coding one engine's names. ``BenchmarkResult`` field names never
change; only the Prometheus names the values are looked up under do.

Batch-histogram table keys are the metric's MAX Serve name (matching the
``HistogramMetric`` members in ``benchmark_shared.metrics``).

TODO(MAX-938): this table collapses once MSN settles on one metric-naming
scheme across engines.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from .backend_names import Backend

__all__ = [
    "BATCH_METRIC_NAMES",
    "BATCH_TYPE_LABELS",
    "SPEC_DECODE_METRIC_NAMES",
    "BatchPhase",
    "EngineFamily",
    "EngineMetricName",
    "MetricAvailability",
    "engine_family",
    "resolve_batch_metric",
    "resolve_spec_decode_metric",
]


class MetricAvailability(Enum):
    """When a metric may be missing from a successful scrape of the engine."""

    ALWAYS_EMITTED = "always_emitted"
    """Present on every scrape of a live engine. A successful scrape without
    it fails the run (``verify_always_emitted_metrics`` in
    ``server_metrics``) instead of silently nulling the field."""

    OPTIONAL = "optional"
    """May legitimately be absent, such as the spec-decode families when the
    engine runs without speculative decoding. Never fails a run."""

    UNAVAILABLE = "unavailable"
    """This engine does not emit the metric, or emits it without the labels
    the client's fields need. Lookups resolve to nothing (and the
    corresponding ``BenchmarkResult`` field stays null) rather than guessing
    an attribution."""


class EngineFamily(Enum):
    """A family of engines that share metric names and label vocabularies."""

    MAX_SERVE = "max_serve"
    MACH = "mach"


class BatchPhase(Enum):
    """The forward-pass phase the client splits batch stats by."""

    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True)
class EngineMetricName:
    """One client-read metric's Prometheus name on one engine family.

    ``name`` is ``None`` when the family has no metric of this shape at all;
    an existing-but-unusable name (such as a Mach histogram that lacks the
    ``batch_type`` label) is recorded together with ``UNAVAILABLE`` so the
    table stays a truthful inventory.
    """

    name: str | None
    availability: MetricAvailability


@dataclass(frozen=True)
class ResolvedMetricName:
    """A fully resolved Prometheus name plus the labels to look it up under."""

    name: str
    labels: Mapping[str, str]


# ``batch_type`` label values per engine family. MAX Serve tags its batch
# histograms with the scheduler's phase names (``CE`` = context encoding,
# ``TG`` = token generation); the Mach engine uses ``prefill`` / ``decode``
# (``BatchKind`` in ``mach/crates/engine-telemetry/src/metrics.rs``).
#
# TODO(MAX-942): mach also emits ``batch_type="mixed"`` batches, one
# forward pass carrying both prefill and decode rows. They count toward
# neither phase here, so mach's prefill/decode batch counts and times
# exclude mixed batches.
BATCH_TYPE_LABELS: Mapping[EngineFamily, Mapping[BatchPhase, str]] = {
    EngineFamily.MAX_SERVE: {
        BatchPhase.PREFILL: "CE",
        BatchPhase.DECODE: "TG",
    },
    EngineFamily.MACH: {
        BatchPhase.PREFILL: "prefill",
        BatchPhase.DECODE: "decode",
    },
}

# Batch histograms the client reads, keyed by their MAX Serve name. Only
# ``mach_batch_size`` and ``mach_batch_execution_time_milliseconds`` are
# tagged with ``batch_type`` and recorded per constructed / executed batch,
# so they are the only Mach batch metrics splittable by phase (and the only
# ones a scrape is required to contain). Mach's other batch histograms carry
# no label, so they are UNAVAILABLE: pooling every batch, including the
# ``mixed`` ones, into one phase would misstate both. MAX Serve families are
# not marked always-emitted because their presence across versions is not
# verified here, so a missing family keeps the historical null-field behavior
# for non-mach backends.
BATCH_METRIC_NAMES: Mapping[str, Mapping[EngineFamily, EngineMetricName]] = {
    "maxserve_batch_context_tokens": {
        EngineFamily.MAX_SERVE: EngineMetricName(
            "maxserve_batch_context_tokens", MetricAvailability.OPTIONAL
        ),
        # Unlabeled in Mach (record_constructed records it without batch_type).
        EngineFamily.MACH: EngineMetricName(
            "mach_batch_context_tokens", MetricAvailability.UNAVAILABLE
        ),
    },
    "maxserve_batch_creation_time_milliseconds": {
        EngineFamily.MAX_SERVE: EngineMetricName(
            "maxserve_batch_creation_time_milliseconds",
            MetricAvailability.OPTIONAL,
        ),
        # Unlabeled in Mach (record_constructed records it without batch_type).
        EngineFamily.MACH: EngineMetricName(
            "mach_batch_creation_time_milliseconds",
            MetricAvailability.UNAVAILABLE,
        ),
    },
    "maxserve_batch_generation_throughput_tokens_per_second": {
        EngineFamily.MAX_SERVE: EngineMetricName(
            "maxserve_batch_generation_throughput_tokens_per_second",
            MetricAvailability.OPTIONAL,
        ),
        # Unlabeled in Mach (record_executed records it without batch_type).
        EngineFamily.MACH: EngineMetricName(
            "mach_batch_generation_throughput_per_second",
            MetricAvailability.UNAVAILABLE,
        ),
    },
    "maxserve_batch_input_tokens": {
        EngineFamily.MAX_SERVE: EngineMetricName(
            "maxserve_batch_input_tokens", MetricAvailability.OPTIONAL
        ),
        # Unlabeled in Mach (record_constructed records it without batch_type).
        EngineFamily.MACH: EngineMetricName(
            "mach_batch_input_tokens", MetricAvailability.UNAVAILABLE
        ),
    },
    "maxserve_batch_prompt_throughput_tokens_per_second": {
        EngineFamily.MAX_SERVE: EngineMetricName(
            "maxserve_batch_prompt_throughput_tokens_per_second",
            MetricAvailability.OPTIONAL,
        ),
        # Unlabeled in Mach (record_executed records it without batch_type).
        EngineFamily.MACH: EngineMetricName(
            "mach_batch_prompt_throughput_per_second",
            MetricAvailability.UNAVAILABLE,
        ),
    },
    "maxserve_batch_size": {
        EngineFamily.MAX_SERVE: EngineMetricName(
            "maxserve_batch_size", MetricAvailability.OPTIONAL
        ),
        EngineFamily.MACH: EngineMetricName(
            "mach_batch_size", MetricAvailability.ALWAYS_EMITTED
        ),
    },
    "maxserve_batch_execution_time_milliseconds": {
        EngineFamily.MAX_SERVE: EngineMetricName(
            "maxserve_batch_execution_time_milliseconds",
            MetricAvailability.OPTIONAL,
        ),
        EngineFamily.MACH: EngineMetricName(
            "mach_batch_execution_time_milliseconds",
            MetricAvailability.ALWAYS_EMITTED,
        ),
    },
}

# Speculative-decode families the client reads. Both engines expose the
# same two histogram shapes under their own prefixes (the per-position
# acceptance rate tagged ``position`` and the batch mean acceptance length),
# and both only when speculative decoding is on, hence OPTIONAL. The third
# entry is the vLLM-style draft-counter family, matched by this prefix
# wherever it appears; Mach does not emit it, so it is UNAVAILABLE for mach.
SPEC_DECODE_METRIC_NAMES: Mapping[
    str, Mapping[EngineFamily, EngineMetricName]
] = {
    "acceptance_rate_per_position": {
        EngineFamily.MAX_SERVE: EngineMetricName(
            "maxserve_spec_decode_acceptance_rate_per_position",
            MetricAvailability.OPTIONAL,
        ),
        EngineFamily.MACH: EngineMetricName(
            "mach_spec_decode_acceptance_rate_per_position",
            MetricAvailability.OPTIONAL,
        ),
    },
    "avg_acceptance_length": {
        EngineFamily.MAX_SERVE: EngineMetricName(
            "maxserve_spec_decode_avg_acceptance_length",
            MetricAvailability.OPTIONAL,
        ),
        EngineFamily.MACH: EngineMetricName(
            "mach_spec_decode_avg_acceptance_length",
            MetricAvailability.OPTIONAL,
        ),
    },
    "draft_counters": {
        EngineFamily.MAX_SERVE: EngineMetricName(
            "vllm:spec_decode", MetricAvailability.OPTIONAL
        ),
        EngineFamily.MACH: EngineMetricName(
            None, MetricAvailability.UNAVAILABLE
        ),
    },
}


def engine_family(backend: Backend | str | None) -> EngineFamily:
    """Map a benchmark backend name onto its engine-metric naming family.

    ``mach`` targets the Mach engine. Every other backend (MAX Serve via
    ``modular`` / ``modular-chat``, and the third-party servers) keeps the
    MAX Serve names, which is what the client read before per-backend names
    existed; a missing ``backend`` (metrics built directly rather than
    scraped) keeps them too.
    """
    if backend == "mach":
        return EngineFamily.MACH
    return EngineFamily.MAX_SERVE


def resolve_batch_metric(
    backend: Backend | None,
    client_metric: str,
    phase: BatchPhase,
) -> ResolvedMetricName | None:
    """Resolve a client batch metric to a backend's Prometheus name + labels.

    Args:
        backend: Backend the metrics were scraped from.
        client_metric: The client-facing metric identity (its MAX Serve
            name, a key of :data:`BATCH_METRIC_NAMES`).
        phase: The forward-pass phase to read (prefill or decode).

    Returns:
        The resolved name and ``batch_type`` labels, or ``None`` when this
        backend does not emit the metric with the labels the client's fields
        need, so callers leave the corresponding field null.
    """
    spec = BATCH_METRIC_NAMES[client_metric][engine_family(backend)]
    if spec.availability is MetricAvailability.UNAVAILABLE or spec.name is None:
        return None
    return ResolvedMetricName(
        name=spec.name,
        labels={"batch_type": BATCH_TYPE_LABELS[engine_family(backend)][phase]},
    )


def resolve_spec_decode_metric(
    backend: Backend | None, client_metric: str
) -> EngineMetricName:
    """Resolve a client spec-decode metric for a backend's engine family.

    Args:
        backend: Backend the metrics were scraped from.
        client_metric: A key of :data:`SPEC_DECODE_METRIC_NAMES`
            (``acceptance_rate_per_position``, ``avg_acceptance_length`` or
            ``draft_counters``).

    Returns:
        The per-family name and availability; an UNAVAILABLE result (or a
        ``None`` name) means the backend never exposes this family and the
        caller must skip it.
    """
    return SPEC_DECODE_METRIC_NAMES[client_metric][engine_family(backend)]
