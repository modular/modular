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

"""Metric instruments and the clients that record them."""

from __future__ import annotations

import abc
import functools
import logging
import time
from collections.abc import AsyncGenerator, Callable, Iterator, Mapping
from contextlib import (
    AbstractAsyncContextManager,
    AbstractContextManager,
    asynccontextmanager,
    contextmanager,
)
from dataclasses import dataclass, field
from typing import Literal, cast, get_args

from max.serve.config import Settings
from opentelemetry import context
from opentelemetry.metrics import get_meter_provider
from opentelemetry.metrics._internal import instrument as api_instrument
from opentelemetry.sdk.metrics._internal import instrument as sdk_instrument
from opentelemetry.sdk.metrics._internal import measurement

"""!! Jank alert !!

We want to use OTEL for propagating telemetry. It is the best vendor-agnostic
metrics system, but that doesn't mean that it is _good_.  OTEL is _slow_. If we
use it directly, it significally degrades the perf of Max Serve. Consequently,
we have all this machinery to observe some metric (MaxMeasurement) and record
the observation async.

OTEL actively obscures its machinery, uses bunch of proxy classes, has an baroque inheritance tree, and is generally awful.
To record an observation at a specific point in time you do the following:
`meter.create_{foo}._real_instrument._measurement_consumer(Measurement(value, timestamp, instrument, ...))`

Here is how you work with metrics (Instruments) observations (Measurements) and recording them (Consumers):
Lets unpack:
1. meter.create_{foo} gives you a proxy instrument with an obscured type eg _internal.instrument._ProxyCounter.
2. `._real_instrument` The proxy can't do anything, you need to grab the _real_ instrument to record.
3. `._measurement_consumer` The _real_ instrument doesn't expose a way to set the time of the observation, so you have to directly talk to the consumer.
4. `Measurement(...)` now we can create a measurement with a timestamp & pass it down.
"""
logger = logging.getLogger("max.serve")
_meter = get_meter_provider().get_meter("modular")


NumberType = float | int
OtelAttributes = Mapping[str, str] | None
# The ``turn`` label on the per-request token and prefix coverage histograms.
ConversationTurn = Literal["first", "later"]
# The ``phase`` label on ``max.request.client_disconnect_cancellations``:
# whether the status code was still unsent when the client hung up.
_DisconnectPhase = Literal["before_response", "during_response"]
# The ``kind`` label on ``max.structured_output.requests``.
_StructuredOutputKind = Literal["json_object", "json_schema"]
# The ``reason`` label on ``max.cache.connector_lookup_admission_stops``.
LookupAdmissionStop = Literal["budget", "max_pending", "held"]

# Exclusive upper bounds, in tokens, of the ``prompt_length`` label on
# ``max.time_to_first_token_by_prompt_length``. Coarse on purpose: the
# label multiplies the histogram's series.
_PROMPT_LENGTH_BUCKETS: tuple[tuple[int, str], ...] = (
    (8 * 1024, "0-8k"),
    (32 * 1024, "8k-32k"),
    (64 * 1024, "32k-64k"),
    (128 * 1024, "64k-128k"),
    (256 * 1024, "128k-256k"),
)


def _prompt_length_bucket(prompt_tokens: int) -> str:
    for upper, label in _PROMPT_LENGTH_BUCKETS:
        if prompt_tokens < upper:
            return label
    return "256k-plus"


# API_PROXIES the "types" of measurements we make from a meter
# SDK instruments are the "types" that actually do recording
API_PROXIES = (
    api_instrument._ProxyCounter
    | api_instrument._ProxyGauge
    | api_instrument._ProxyHistogram
    | api_instrument._ProxyUpDownCounter
)

SDK_INSTRUMENTS = (
    sdk_instrument._Counter
    | sdk_instrument._Gauge
    | sdk_instrument._Histogram
    | sdk_instrument._UpDownCounter
)

SupportedInstruments = API_PROXIES | SDK_INSTRUMENTS


def _create_counter(name: str, description: str) -> SupportedInstruments:
    """Creates a counter, typed as the instrument the meter usually returns.

    ``create_counter`` is annotated with the abstract counter, but the meter
    hands back a proxy counter, or the SDK one once a provider is set. A
    no-op or disabled provider hands back neither, and a runtime check here
    would fail the import, so the type is asserted statically and
    ``_commit_measurement`` skips such a counter as it does any other.
    """
    counter = _meter.create_counter(name, description=description)
    return cast(SupportedInstruments, counter)


# Sorry for the type ignores, OTEL goes out of its way to obscure its types.
# We need to use the fact that these are actually _Proxy{Type} objects  rather
# than {Type} objects
SERVE_METRICS: dict[str, SupportedInstruments] = {
    "max.request_count": _meter.create_counter(
        "max.request_count",
        description=(
            "Total API requests since server start. Liveness and "
            "observability endpoints (/health, /version, /ping and /metrics) "
            "are excluded."
        ),
    ),  # type: ignore
    "max.request.client_disconnect_cancellations": _meter.create_counter(
        "max.request.client_disconnect_cancellations",
        description=(
            "Requests whose client disconnected before the server finished "
            "responding, split by `phase`. `before_response` covers the chat "
            "and text completion routes, which watch for a disconnect while "
            "a request waits for its result; `max_request_count_total` "
            "also records these as code 499. `during_response` is a "
            "disconnect after a streamed response started and before its "
            "last chunk, which `max_request_count_total` records under "
            "the response's status code, so it is visible only here. Both "
            "phases are exported at zero from startup."
        ),
    ),  # type: ignore
    "max.request_time": _meter.create_histogram(
        "max.request_time",
        unit="ms",
        description=(
            "Total time spent handling a request, measured from the HTTP "
            "middleware receiving it to the response completing: request "
            "parsing, validation, queueing, model execution and response "
            "generation."
        ),
    ),  # type: ignore
    "max.input_processing_time": _meter.create_histogram(
        "max.input_processing_time",
        unit="ms",
        description="Input processing time (IPT).",
    ),  # type: ignore
    "max.output_processing_time": _meter.create_histogram(
        "max.output_processing_time",
        unit="ms",
        description="Output generation time (OGT).",
    ),  # type: ignore
    "max.time_to_first_token": _meter.create_histogram(
        "max.time_to_first_token",
        unit="ms",
        description=(
            "Time to first token (TTFT). Measured from when the server received the request, so it includes request parsing, validation, media resolution, tokenization, queueing, and prefill."
        ),
    ),  # type: ignore
    "max.time_to_first_token_by_prompt_length": _meter.create_histogram(
        "max.time_to_first_token_by_prompt_length",
        unit="ms",
        description=(
            "Time to first token as `max_time_to_first_token_milliseconds` measures it, split by a `prompt_length` label that buckets the request's prompt tokens in units of 1024: `0-8k`, `8k-32k`, `32k-64k`, `64k-128k`, `128k-256k` or `256k-plus`. Compares TTFT across deployments at matched prompt length."
        ),
    ),  # type: ignore
    "max.num_input_tokens": _meter.create_counter(
        "max.num_input_tokens",
        description="Cumulative input tokens processed.",
    ),  # type: ignore
    "max.num_output_tokens": _meter.create_counter(
        "max.num_output_tokens",
        description="Cumulative output tokens generated.",
    ),  # type: ignore
    "max.num_requests_queued": _meter.create_gauge(
        "max.num_requests_queued",
        description=(
            "Current depth of the scheduler's CE/prefill queue, sampled once per scheduler iteration."
        ),
    ),  # type: ignore
    "max.num_requests_running": _meter.create_up_down_counter(
        "max.num_requests_running",
        description="Requests currently being processed.",
    ),  # type: ignore
    "max.num_requests_awaiting_admission": _meter.create_up_down_counter(
        "max.num_requests_awaiting_admission",
        description=(
            "Requests received by the API server but not yet handed off to the model worker. A persistently high value indicates a backlog in the API server rather than the scheduler."
        ),
    ),  # type: ignore
    "max.requests_awaiting_admission": _meter.create_histogram(
        "max.requests_awaiting_admission",
        description=(
            "Distribution of the ingress backlog (requests accepted by the API server but not yet handed off to the model worker), sampled periodically. Captures the p50/p99 tail over time."
        ),
    ),  # type: ignore
    "max.num_responses_buffered": _meter.create_gauge(
        "max.num_responses_buffered",
        description=(
            "Model-worker responses received by the API server but not yet consumed by the streaming layer (sum of per-request output-queue depths), sampled periodically."
        ),
    ),  # type: ignore
    "max.responses_buffered": _meter.create_histogram(
        "max.responses_buffered",
        description=(
            "Distribution of the egress backlog, sampled periodically. Captures the p50/p99 tail that a scrape-interval gauge sample would miss."
        ),
    ),  # type: ignore
    "max.response_queue_time": _meter.create_histogram(
        "max.response_queue_time",
        unit="ms",
        description=(
            "Time a model-worker response waits in the per-request output queue before the streaming layer consumes it."
        ),
    ),  # type: ignore
    "max.model_load_time": _meter.create_histogram(
        "max.model_load_time",
        unit="ms",
        description=(
            "Model-worker startup duration. Also split by `component` tag (`build`, `compile`, `init`, `graph_capture`, `pinned_memory`, `spawn`, `total`)."
        ),
    ),  # type: ignore
    "max.itl": _meter.create_histogram(
        "max.itl", unit="ms", description="Inter-token latency."
    ),  # type: ignore
    "max.time_per_output_token": _meter.create_histogram(
        "max.time_per_output_token",
        unit="ms",
        description=(
            "Mean decode-phase latency per generated token (TPOT). Emitted once per request as decode_time / (num_generated_tokens - 1). Excludes the first token and prefill/TTFT."
        ),
    ),  # type: ignore
    "max.pipeline_load": _meter.create_counter(
        "max.pipeline_load",
        description="Pipelines loaded for each model.",
    ),  # type: ignore
    "max.batch.size": _meter.create_histogram(
        "max.batch.size",
        description=(
            "Distribution of batch sizes (number of requests), labeled by `batch_type` (CE prefill or TG decode)."
        ),
    ),  # type: ignore
    "max.batch.execution_time": _meter.create_histogram(
        "max.batch.execution_time",
        unit="ms",
        description="Host executor service time per completed batch.",
    ),  # type: ignore
    "max.scheduler.batch_step_time": _meter.create_histogram(
        "max.scheduler.batch_step_time",
        unit="ms",
        description=(
            "Whole scheduler step including execution, request advancement, "
            "release and response publication; excludes batch construction."
        ),
    ),  # type: ignore
    "max.di.decode_admission_queue_wait_time": _meter.create_histogram(
        "max.di.decode_admission_queue_wait_time",
        unit="ms",
        description=(
            "Time a request spends in the decode node's admission queue before dispatch to prefill."
        ),
    ),  # type: ignore
    "max.di.decode_send_time": _meter.create_histogram(
        "max.di.decode_send_time",
        unit="ms",
        description=(
            "Time to build and send a prefill request on the decode side (struct construction, serialization, ZMQ enqueue)."
        ),
    ),  # type: ignore
    "max.di.dispatch_rtt": _meter.create_histogram(
        "max.di.dispatch_rtt",
        unit="ms",
        description=(
            "Decode-clock round trip from sending a prefill request to receiving the prefill node's acknowledgment."
        ),
    ),  # type: ignore
    "max.di.prefill_span": _meter.create_histogram(
        "max.di.prefill_span",
        unit="ms",
        description=(
            'Decode-clock time between the prefill node\'s "arrived" and "ce_done" acknowledgments for a request. Covers prefill queue wait plus CE execution plus the return hop.'
        ),
    ),  # type: ignore
    "max.di.reply_rtt": _meter.create_histogram(
        "max.di.reply_rtt",
        unit="ms",
        description=(
            'Decode-clock time from the prefill node\'s "ce_done" acknowledgment to the actual prefill response arriving. Covers KV-transfer initiation, reply serialization, and network transit.'
        ),
    ),  # type: ignore
    "max.di.prefill_queue_wait_time": _meter.create_histogram(
        "max.di.prefill_queue_wait_time",
        unit="ms",
        description=(
            "Prefill-local time a request spends enqueued before its first CE batch executes."
        ),
    ),  # type: ignore
    "max.ce_pool_wait_time": _meter.create_histogram(
        "max.ce_pool_wait_time",
        unit="ms",
        description=(
            "Time a CE request spends in the data-parallel balancer's unbound pool, from arrival until it is bound to a replica."
        ),
    ),  # type: ignore
    "max.di.decode_postprocess_time": _meter.create_histogram(
        "max.di.decode_postprocess_time",
        unit="ms",
        description=(
            "Decode-local time from receiving a prefill response to handing the result off to the response queue."
        ),
    ),  # type: ignore
    "max.di.early_sync_time": _meter.create_histogram(
        "max.di.early_sync_time",
        unit="ms",
        description=(
            "Duration of the early-sync guard's blocking commit of the "
            "previous batch's structured-output FSM state, when it fired "
            "(see '_should_early_sync_prev_batch')."
        ),
    ),  # type: ignore
    "max.di.api_pre_tokenize": _meter.create_histogram(
        "max.di.api_pre_tokenize",
        unit="ms",
        description=(
            "API-local time from HTTP accept to tokenize start (parse / route)."
        ),
    ),  # type: ignore
    "max.di.api_submit": _meter.create_histogram(
        "max.di.api_submit",
        unit="ms",
        description=(
            "API-local time from tokenize done to model-worker ZMQ put."
        ),
    ),  # type: ignore
    "max.di.mw_queue_wait": _meter.create_histogram(
        "max.di.mw_queue_wait",
        unit="ms",
        description=(
            "Decode-worker time from first ZMQ pull of a request to "
            "admission-queue enqueue (drain gap)."
        ),
    ),  # type: ignore
    "max.di.ce_preempted_tg_iteration_count": _meter.create_counter(
        "max.di.ce_preempted_tg_iteration_count",
        description=(
            "Counter: incremented once per scheduling iteration in which a "
            "CE batch was admitted while TG requests were pending, so those "
            "TG requests got zero progress this iteration -- how often this "
            "preemption pattern occurs."
        ),
    ),  # type: ignore
    "max.di.ce_preempted_tg_pending_count": _meter.create_histogram(
        "max.di.ce_preempted_tg_pending_count",
        description=(
            "Histogram: the number of TG requests pending at the moment of "
            "one such preemption (paired with "
            "'max.di.ce_preempted_tg_iteration_count', which counts how "
            "often it happens; this measures how large each occurrence is)."
        ),
    ),  # type: ignore
    "max.di.handoff_to_first_token_time": _meter.create_histogram(
        "max.di.handoff_to_first_token_time",
        unit="ms",
        description=(
            "Wall-clock time from a prefill-to-decode handoff landing (its "
            "token discarded, re-queued as a one-token chunked-CE "
            "continuation) to decode's own re-forward actually sending "
            "token 1 to the API process. A TTFT contribution specific to "
            "the handoff path; the ordinary path's token already arrives "
            "with the 'PrefillResponse', so it has no equivalent wait."
        ),
    ),  # type: ignore
    "max.cache.num_used_blocks": _meter.create_gauge(
        "max.cache.num_used_blocks",
        unit="blocks",
        description="Number of KV cache blocks currently in use.",
    ),  # type: ignore
    "max.cache.num_total_blocks": _meter.create_gauge(
        "max.cache.num_total_blocks",
        unit="blocks",
        description="Total number of KV cache blocks.",
    ),  # type: ignore
    "max.cache.jenga_small_blocks": _meter.create_gauge(
        "max.cache.jenga_small_blocks",
        unit="blocks",
        description=(
            "A Jenga KV leaf's small blocks on one replica, by `state`: `used` by a request, `evictable` (unreferenced but still in the prefix cache), or `empty` to allocate without evicting. The leaf's remaining small blocks are bytes other leaves hold. Sampled at the scheduler's log interval."
        ),
    ),  # type: ignore
    "max.cache.jenga_huge_blocks": _meter.create_gauge(
        "max.cache.jenga_huge_blocks",
        unit="blocks",
        description=(
            "Jenga huge blocks on one replica, by `leaf` and `state`: `used` (a request holds small blocks in it) or `evictable` (no request holds it, but it is still typed to the leaf), plus `leaf=none, state=empty` for those no leaf has claimed. Sums to the replica's allocatable huge blocks. Sampled at the scheduler's log interval."
        ),
    ),  # type: ignore
    "max.cache.request_prefix_coverage": _meter.create_histogram(
        "max.cache.request_prefix_coverage",
        unit="percent",
        description=(
            "Per-request prefix cache coverage (cached prefix tokens / prompt tokens). For the token-weighted cache hit rate, derive it from `max_cache_hits_tokens_total` and `max_cache_misses_tokens_total`."
        ),
    ),  # type: ignore
    "max.cache.preemption_count": _meter.create_counter(
        "max.cache.preemption_count",
        description="Total number of preemptions.",
    ),  # type: ignore
    "max.cache.connector_cordon_wait_time": _meter.create_histogram(
        "max.cache.connector_cordon_wait_time",
        unit="ms",
        description=(
            "Time a prompt is held out of the batch on its KV connector, from the first time it is cordoned on a deferred connector lookup or an onload to its admission to a batch. Recorded once for each admitted prompt that was cordoned."
        ),
    ),  # type: ignore
    "max.cache.connector_cordon_passes": _meter.create_histogram(
        "max.cache.connector_cordon_passes",
        unit="passes",
        description=(
            "Scheduler passes from a prompt's first cordon on a deferred connector lookup or an onload to its admission to a batch, recorded with `max_cache_connector_cordon_wait_time_milliseconds`. A prompt admitted in the pass that cordoned it records 0."
        ),
    ),  # type: ignore
    "max.cache.connector_lookup_admission_stops": _create_counter(
        "max.cache.connector_lookup_admission_stops",
        description="Prompt admission passes that stopped on a deferred connector lookup, counted per data-parallel replica, split by `reason`: `budget` when the pass had spent its share of lookup tokens, `max_pending` when the replica already had its bound of unanswered lookups out, and `held` when the next prompt arrived behind one still out on its lookup. The prompt waits at the head of its queue for a later pass.",
    ),
    "max.cache.hits": _meter.create_counter(
        "max.cache.hits",
        unit="tokens",
        description=(
            "Cumulative KV cache hit tokens (prompt tokens served from the prefix cache). Tagged with a `tier` label (`g0` for the on-device prefix cache, `external` for the KV connector). The per-tier series sum to the untagged total."
        ),
    ),  # type: ignore
    "max.cache.misses": _meter.create_counter(
        "max.cache.misses",
        unit="tokens",
        description=(
            "Cumulative KV cache miss tokens (prompt tokens actually prefilled by the model)."
        ),
    ),  # type: ignore
    "max.input_tokens_per_request": _meter.create_histogram(
        "max.input_tokens_per_request",
        unit="tokens",
        description=(
            "Distribution of input tokens per request. Chat requests carry a `turn` label: `first` when the conversation has no assistant message yet, `later` when it resends earlier turns. Completion requests carry none."
        ),
    ),  # type: ignore
    "max.prefix_coverage_per_request": _meter.create_histogram(
        "max.prefix_coverage_per_request",
        unit="percent",
        description=(
            "Per-request prefix cache coverage (cached prefix tokens / prompt tokens) for chat requests, recorded when the request ends. Carries a `turn` label: `first` when the conversation has no assistant message yet, `later` when it resends earlier turns. `max_cache_request_prefix_coverage_percent` records the same ratio for every request from the scheduler, without the label."
        ),
    ),  # type: ignore
    "max.output_tokens_per_request": _meter.create_histogram(
        "max.output_tokens_per_request",
        unit="tokens",
        description=(
            "Distribution of output tokens per request. Chat requests carry a `turn` label: `first` when the conversation has no assistant message yet, `later` when it resends earlier turns. Completion requests carry none."
        ),
    ),  # type: ignore
    "max.dkv.nixl_read_latency": _meter.create_histogram(
        "max.dkv.nixl_read_latency",
        unit="ms",
        description="NIXL READ transfer latency.",
    ),  # type: ignore
    "max.dkv.nixl_read_latency_max": _meter.create_histogram(
        "max.dkv.nixl_read_latency_max",
        unit="ms",
        description="Slowest single NIXL READ transfer in a batch's window.",
    ),  # type: ignore
    "max.dkv.nixl_write_latency": _meter.create_histogram(
        "max.dkv.nixl_write_latency",
        unit="ms",
        description="NIXL WRITE transfer latency.",
    ),  # type: ignore
    "max.dkv.rpc_acquire_latency": _meter.create_histogram(
        "max.dkv.rpc_acquire_latency",
        unit="ms",
        description="dKV acquire_blocks RPC latency.",
    ),  # type: ignore
    "max.dkv.rpc_read_latency": _meter.create_histogram(
        "max.dkv.rpc_read_latency",
        unit="ms",
        description="dKV read_blocks RPC latency.",
    ),  # type: ignore
    "max.dkv.read_blocks": _meter.create_counter(
        "max.dkv.read_blocks",
        unit="blocks",
        description="Cache blocks LANDED in device memory from the dKV tier, over both the remote NIXL path and same-host device copies. Only confirmed-complete transfers count, so this is delivered reuse and not an upper bound on it: blocks dKV holds but cannot serve as a contiguous prefix are dropped before a transfer is ever built, and never reach this counter. For any load that lands it is proportional to max.cache.hits{tier=external}; a persistent shortfall means transfers are failing.",
    ),  # type: ignore
    "max.dkv.connected_clients": _meter.create_gauge(
        "max.dkv.connected_clients",
        unit="clients",
        description="Number of dKV connector clients currently connected to the external KV tier.",
    ),  # type: ignore
    "max.dkv.total_clients": _meter.create_gauge(
        "max.dkv.total_clients",
        unit="clients",
        description="Total number of dKV connector clients, one per data-parallel replica.",
    ),  # type: ignore
    "max.dkv.reconnect_attempts": _meter.create_gauge(
        "max.dkv.reconnect_attempts",
        unit="attempts",
        description="Cumulative dKV reconnect attempts across all clients, reported as a gauge of the current lifetime total.",
    ),  # type: ignore
    # The dKV settle worker and the owner release halt. Mach exports the halt
    # under the same name as dkv.connector.owner_releases_halted.
    "max.dkv.settle_queue_depth": _meter.create_gauge(
        "max.dkv.settle_queue_depth",
        unit="jobs",
        description="dKV settle jobs queued and not yet finished, summed over every connector client. A level.",
    ),  # type: ignore
    "max.dkv.settle_queue_oldest_age_ms": _meter.create_gauge(
        "max.dkv.settle_queue_oldest_age_ms",
        unit="ms",
        description="How long the oldest unfinished dKV settle job has waited, the max over every connector client, zero when none is queued. A head stuck on an RPC or a reconnect ages at wall-clock rate while max.dkv.settle_jobs_finished stays flat, and a worker that is only behind under load ages more slowly while that count keeps rising.",
    ),  # type: ignore
    "max.dkv.owner_releases_halted": _meter.create_gauge(
        "max.dkv.owner_releases_halted",
        description="1 while any dKV connector client's owner release halt is up, else 0. While it is up the owner's failing cleanups wait on the orphan queue and its drain backs off.",
    ),  # type: ignore
    "max.dkv.settle_jobs_finished": _create_counter(
        "max.dkv.settle_jobs_finished",
        description="dKV settle jobs the connector workers finished or dropped for a gone dKV incarnation.",
    ),
    # Cross-node pull. Counters, not gauges: take_metrics clears these every
    # batch, so each publish carries that window's delta. Mach exports the same
    # set under dkv.connector.peer_attach_count, peer_drop_count,
    # peer_load_count, peer_transfer_failure_count, peer_metadata_refresh_count
    # and hint_rejected_count, splitting success from failure by an outcome tag
    # rather than by metric name.
    "max.dkv.peer_attaches": _meter.create_counter(
        "max.dkv.peer_attaches",
        unit="attaches",
        description="Remote dKV peers attached from a cache hint. An attach is cached per peer, so a stable peer set leaves this at zero while cross-node pulls keep succeeding; max.dkv.peer_loads is the steady-state signal.",
    ),  # type: ignore
    "max.dkv.peer_attach_failures": _meter.create_counter(
        "max.dkv.peer_attach_failures",
        unit="attaches",
        description="Peer attaches that failed at dial, probe, handshake, or the attach timeout.",
    ),  # type: ignore
    "max.dkv.peers_dropped": _meter.create_counter(
        "max.dkv.peers_dropped",
        unit="peers",
        description="Remote dKV peers dropped, whether replaced after a peer restart, evicted over the peer table's cap, or torn down by the caller. Read against max.dkv.peer_attaches to tell a churning peer table from a stable one.",
    ),  # type: ignore
    "max.dkv.peer_loads": _meter.create_counter(
        "max.dkv.peer_loads",
        unit="loads",
        description="Loads served from the dKV peer a cache hint named. Counted per batch on the fused load path and per hinted lookup on the two-phase path, so it is not a request count.",
    ),  # type: ignore
    "max.dkv.peer_load_failures": _meter.create_counter(
        "max.dkv.peer_load_failures",
        unit="loads",
        description="Hinted peer loads that fell through to the next source or to the co-located dKV. Memo-capped rather than per-request: a failed source suppresses further attempts against the same instance and epoch for a fixed window, so one dead peer charges roughly one failure per window however many requests it affects.",
    ),  # type: ignore
    "max.dkv.peer_transfer_failures": _meter.create_counter(
        "max.dkv.peer_transfer_failures",
        unit="transfers",
        description="Hint-routed peer leases whose NIXL read failed to build, post or complete, counted once per peer a round leased from. The failure is the peer's alone: the connector stays healthy, and the peer is skipped for the failure memo window, so later rounds lease from the co-located dKV, and marked stale so its metadata is reloaded. The usual cause of a failed post is a peer that registered memory after the connector loaded its metadata.",
    ),  # type: ignore
    "max.dkv.peer_metadata_refreshes": _meter.create_counter(
        "max.dkv.peer_metadata_refreshes",
        unit="refreshes",
        description="Stale dKV peers whose NIXL metadata the connector reloaded in place, after which the peer is routed to again.",
    ),  # type: ignore
    "max.dkv.peer_metadata_refresh_failures": _meter.create_counter(
        "max.dkv.peer_metadata_refresh_failures",
        unit="refreshes",
        description="In-place peer metadata reloads that NIXL refused. Each strands its peer: the connector never routes to or dials it again for the life of the process. Each also counts in max.dkv.peer_attach_failures.",
    ),  # type: ignore
    "max.dkv.hints_rejected": _meter.create_counter(
        "max.dkv.hints_rejected",
        unit="hints",
        description="Cache hints that arrived but were unusable: bytes that did not parse, an unknown version, a chain that does not describe the request, no entry for the requested group, or a source instance record that failed validation. Counted per entry on the two-phase load path and per call on the fused one.",
    ),  # type: ignore
    # Fail-opens: loads the connector answered as misses without a transfer.
    # Counters, since take_metrics clears them every batch like the peer set.
    "max.dkv.prepare_load_degraded": _create_counter(
        "max.dkv.prepare_load_degraded",
        description="dKV lookups that found the connector degraded and planned nothing, so the request was served as a cache miss. Read with max.dkv.load_with_plan_degraded and max.dkv.load_with_plan_expired, which count an outage that starts after the lookup.",
    ),
    "max.dkv.load_with_plan_degraded": _create_counter(
        "max.dkv.load_with_plan_degraded",
        description="dKV loads whose connector degraded between the lookup and the post, so the leased blocks were released and the request was served as a cache miss.",
    ),
    "max.dkv.load_with_plan_expired": _create_counter(
        "max.dkv.load_with_plan_expired",
        description="dKV read leases that aged past their deadline before the load posted them, released as a miss rather than read. Counted per lease, so one load spanning several dKV instances can count more than one.",
    ),
    # The dKV orphan queue, where failed reader-pin releases and Filling aborts
    # wait to be sent again. Counters for the outcomes, since take_metrics
    # clears them every batch, and a gauge for the depth, a level.
    "max.dkv.orphan_cleanups_reissued": _create_counter(
        "max.dkv.orphan_cleanups_reissued",
        description="Deferred dKV releases and aborts that a drain sent again and dKV took.",
    ),
    "max.dkv.orphan_cleanups_abandoned": _create_counter(
        "max.dkv.orphan_cleanups_abandoned",
        description="Deferred dKV releases and aborts left to dKV's own sweep because the instance that held them was gone.",
    ),
    "max.dkv.orphan_cleanups_dropped": _create_counter(
        "max.dkv.orphan_cleanups_dropped",
        description="Deferred dKV releases and aborts dropped to keep the queue under its cap. Should be zero.",
    ),
    "max.dkv.orphan_cleanups_refused": _create_counter(
        "max.dkv.orphan_cleanups_refused",
        description="Deferred dKV releases and aborts that dKV refused, dropped rather than sent again.",
    ),
    "max.dkv.orphan_cleanups_expired": _create_counter(
        "max.dkv.orphan_cleanups_expired",
        description="Deferred dKV releases and aborts dropped once dKV's own sweep had freed what they named.",
    ),
    "max.dkv.orphan_queue_depth": _meter.create_gauge(
        "max.dkv.orphan_queue_depth",
        unit="cleanups",
        description="Deferred dKV releases and aborts queued or being reissued now, summed over every connector client. A level.",
    ),  # type: ignore
    # Per-call dKV client counters. Counters, not histograms, so a dashboard
    # derives the per-call mean as time over calls; the rpc_*_latency
    # histograms above record one per-batch mean each.
    "max.dkv.hints_rejected_by_reason": _meter.create_counter(
        "max.dkv.hints_rejected_by_reason",
        description="max.dkv.hints_rejected split by `reason`: unparseable bytes, no entry for the group, a chain that does not contain the request, or an instance record that failed validation.",
    ),  # type: ignore
    "max.dkv.rpc_calls": _meter.create_counter(
        "max.dkv.rpc_calls",
        description="dKV RPCs the connector clients issued, split by `op`, including calls to peer dKV instances, transfer settles and touches.",
    ),  # type: ignore
    "max.dkv.rpc_time_ms": _meter.create_counter(
        "max.dkv.rpc_time_ms",
        description="Cumulative wall milliseconds of dKV RPCs, split by `op`, including calls to peer dKV instances and the wait for the client's socket. Divide by max.dkv.rpc_calls for the per-call mean.",
    ),  # type: ignore
    "max.dkv.rpc_socket_wait_ms": _meter.create_counter(
        "max.dkv.rpc_socket_wait_ms",
        description="Cumulative milliseconds dKV RPCs queued for their client's single REQ socket behind another call in flight, split by `op`, including calls to peer dKV instances.",
    ),  # type: ignore
    "max.dkv.lock_waits": _meter.create_counter(
        "max.dkv.lock_waits",
        description="Times a dKV connector entry point took the connector lock, split by `entry`.",
    ),  # type: ignore
    "max.dkv.lock_wait_ms": _meter.create_counter(
        "max.dkv.lock_wait_ms",
        description="Cumulative milliseconds dKV connector entry points waited for the connector lock, split by `entry`.",
    ),  # type: ignore
    "max.dkv.touches_deduplicated": _create_counter(
        "max.dkv.touches_deduplicated",
        description="Recency touches the dKV client dropped because a touch it sent within the last second already covered their keys. Each one is a touch RPC the dKV server did not have to serve.",
    ),
    "max.dkv.blocked_calls": _create_counter(
        "max.dkv.blocked_calls",
        description="Calls into each dKV connector entry point, split by `entry`. Divide max.dkv.blocked_ms by this for the time the scheduler thread spent per call.",
    ),
    "max.dkv.blocked_ms": _create_counter(
        "max.dkv.blocked_ms",
        description="Cumulative milliseconds the scheduler thread spent inside each dKV connector entry point, lock wait and any RPC included, split by `entry`.",
    ),
    "max.spec_decode.acceptance_rate_per_position": _meter.create_histogram(
        "max.spec_decode.acceptance_rate_per_position",
        unit="percent",
        description="Draft-token acceptance rate per position (0-100%).",
    ),  # type: ignore
    "max.batch.input_tokens": _meter.create_histogram(
        "max.batch.input_tokens",
        unit="tokens",
        description="Distribution of input tokens per scheduler batch.",
    ),  # type: ignore
    "max.batch.context_tokens": _meter.create_histogram(
        "max.batch.context_tokens",
        unit="tokens",
        description="Distribution of accumulated context tokens per scheduler batch.",
    ),  # type: ignore
    "max.batch.creation_time": _meter.create_histogram(
        "max.batch.creation_time",
        unit="ms",
        description="Distribution of scheduler batch creation time.",
    ),  # type: ignore
    "max.batch.prompt_throughput": _meter.create_histogram(
        "max.batch.prompt_throughput",
        unit="tokens/s",
        description="Per-batch prompt-side throughput per second.",
    ),  # type: ignore
    "max.batch.generation_throughput": _meter.create_histogram(
        "max.batch.generation_throughput",
        unit="tokens/s",
        description="Per-batch generation-side throughput per second.",
    ),  # type: ignore
    "max.dp_active_token_occupancy": _meter.create_histogram(
        "max.dp_active_token_occupancy",
        unit="%",
        description=(
            "Per-batch data-parallel balance, measured as the mean/max of per-rank active-token load as a percentage. 100% means perfectly balanced ranks."
        ),
    ),  # type: ignore
    "max.dp_context_token_occupancy": _meter.create_histogram(
        "max.dp_context_token_occupancy",
        unit="%",
        description=(
            "Per-batch data-parallel balance, measured as the mean/max of per-rank context-token (KV/attention) load as a percentage. 100% means perfectly balanced ranks."
        ),
    ),  # type: ignore
    "max.dp_active_tokens": _meter.create_counter(
        "max.dp_active_tokens",
        unit="tokens",
        description=(
            "Cumulative active tokens scheduled across all data-parallel replicas, excluding padding. Divide by `max_dp_step_capacity_tokens_total` over the same window for token-weighted DP occupancy."
        ),
    ),  # type: ignore
    "max.dp_step_capacity_tokens": _meter.create_counter(
        "max.dp_step_capacity_tokens",
        unit="tokens",
        description=(
            "Cumulative synchronized step capacity. For each batch, this is the data-parallel degree times the heaviest rank's active tokens."
        ),
    ),  # type: ignore
    "max.batch.terminated_reqs": _meter.create_histogram(
        "max.batch.terminated_reqs",
        unit="reqs",
        description="Distribution of requests terminated per scheduler batch.",
    ),  # type: ignore
    "max.batch.pending_reqs": _meter.create_histogram(
        "max.batch.pending_reqs",
        unit="reqs",
        description="Distribution of requests pending in the queue, sampled once per scheduler batch.",
    ),  # type: ignore
    "max.cache.used_kv_pct": _meter.create_histogram(
        "max.cache.used_kv_pct",
        unit="percent",
        description="Percentage of device KV cache blocks in use (0-100%), sampled once per scheduler batch.",
    ),  # type: ignore
    "max.cache.used_host_kv_pct": _meter.create_histogram(
        "max.cache.used_host_kv_pct",
        unit="percent",
        description="Percentage of the KV connector's host tier bytes in use (0-100%), sampled once per scheduler batch when a connector with a host tier is configured.",
    ),  # type: ignore
    "max.cache.device_blocks_served": _meter.create_counter(
        "max.cache.device_blocks_served",
        unit="blocks",
        description=(
            "Cumulative KV blocks served directly from the local device prefix cache."
        ),
    ),  # type: ignore
    "max.cache.h2d_bytes_copied": _meter.create_counter(
        "max.cache.h2d_bytes_copied",
        unit="bytes",
        description=(
            "Cumulative KV bytes copied from the connector's host tier to "
            "device. Rate this to see the PCIe bandwidth the host tier "
            "consumes."
        ),
    ),  # type: ignore
    "max.cache.cross_replica_blocks_copied": _meter.create_counter(
        "max.cache.cross_replica_blocks_copied",
        unit="blocks",
        description=(
            "Cumulative KV blocks copied device-to-device across data-parallel replicas to reuse a prefix-cache hit on another replica."
        ),
    ),  # type: ignore
    "max.cache.cross_replica_bytes_copied": _meter.create_counter(
        "max.cache.cross_replica_bytes_copied",
        unit="bytes",
        description=(
            "Cumulative bytes moved by device-to-device KV copies across data-parallel replicas."
        ),
    ),  # type: ignore
    "max.cache.d2h_bytes_copied": _meter.create_counter(
        "max.cache.d2h_bytes_copied",
        unit="bytes",
        description=(
            "Cumulative KV bytes copied from device to the connector's host "
            "tier."
        ),
    ),  # type: ignore
    "max.cache.connector_loads_refused": _meter.create_counter(
        "max.cache.connector_loads_refused",
        unit="loads",
        description=(
            "Cumulative loads the KV connector refused, from its host and "
            "disk tiers or from dKV. "
            "Each refused load is served by recomputing its blocks, so a "
            "steady rate is hit rate the tiers reported and could not "
            "deliver."
        ),
    ),  # type: ignore
    "max.cache.connector_load_failures": _meter.create_counter(
        "max.cache.connector_load_failures",
        unit="loads",
        description=(
            "Cumulative KV connector loads whose copy failed. Each is served "
            "by recomputing the request's prefix without the connector, so "
            "any of these is a transport fault rather than a cache miss."
        ),
    ),  # type: ignore
    "max.cache.connector_offload_blocks_dropped": _meter.create_counter(
        "max.cache.connector_offload_blocks_dropped",
        unit="blocks",
        description=(
            "Cumulative offloaded KV blocks the connector's host pool had no "
            "room for. A pool starved by blocks pinned for in-flight transfers "
            "drops offloads before its hit rate falls."
        ),
    ),  # type: ignore
    "max.cache.disk_bytes_read": _meter.create_counter(
        "max.cache.disk_bytes_read",
        unit="bytes",
        description="Cumulative KV bytes read from the disk cache tier.",
    ),  # type: ignore
    "max.cache.disk_bytes_written": _meter.create_counter(
        "max.cache.disk_bytes_written",
        unit="bytes",
        description="Cumulative KV bytes written to the disk cache tier.",
    ),  # type: ignore
    "max.spec_decode.avg_acceptance_length": _meter.create_histogram(
        "max.spec_decode.avg_acceptance_length",
        unit="tokens",
        description="Mean draft-token acceptance length per speculative decoding batch.",
    ),  # type: ignore
    "max.dkv.nixl_read_gib_per_s": _meter.create_histogram(
        "max.dkv.nixl_read_gib_per_s",
        unit="GiB/s",
        description="NIXL READ throughput.",
    ),  # type: ignore
    "max.dkv.nixl_write_gib_per_s": _meter.create_histogram(
        "max.dkv.nixl_write_gib_per_s",
        unit="GiB/s",
        description="NIXL WRITE throughput.",
    ),  # type: ignore
    "max.cache.used_disk_kv_pct": _meter.create_histogram(
        "max.cache.used_disk_kv_pct",
        unit="percent",
        description="Percentage of the KV connector's disk tier bytes in use (0-100%), sampled once per scheduler batch when the connector has a disk tier.",
    ),  # type: ignore
    "max.vision.images_encoded": _meter.create_counter(
        "max.vision.images_encoded",
        unit="images",
        description="Cumulative images run through the vision encoder (cache misses).",
    ),  # type: ignore
    "max.vision.images_cached": _meter.create_counter(
        "max.vision.images_cached",
        unit="images",
        description="Cumulative images served from the vision encoder cache (cache hits).",
    ),  # type: ignore
    "max.vision.patches_encoded": _meter.create_counter(
        "max.vision.patches_encoded",
        unit="patches",
        description="Cumulative image patches fed to the vision encoder.",
    ),  # type: ignore
    "max.vision.tokens_encoded": _meter.create_counter(
        "max.vision.tokens_encoded",
        unit="tokens",
        description="Cumulative merged vision tokens produced by the vision encoder.",
    ),  # type: ignore
    "max.vision.cache_hit_rate": _meter.create_histogram(
        "max.vision.cache_hit_rate",
        unit="percent",
        description="Per-batch vision encoder cache hit rate (0-100%).",
    ),  # type: ignore
    "max.vision.preprocess_cache_hits": _meter.create_counter(
        "max.vision.preprocess_cache_hits",
        unit="images",
        description="Cumulative images found already preprocessed, whose decode and preprocessing are both skipped.",
    ),  # type: ignore
    "max.vision.preprocess_cache_misses": _meter.create_counter(
        "max.vision.preprocess_cache_misses",
        unit="images",
        description="Cumulative images not already preprocessed, which the API server must decode and preprocess.",
    ),  # type: ignore
    "max.vision.image_admission_decode_time": _meter.create_histogram(
        "max.vision.image_admission_decode_time",
        unit="ms",
        description="Per-request wall-clock time of the admission image work: deciding which images are already preprocessed, then decoding and validating the rest.",
    ),  # type: ignore
    "max.video.clips_encoded": _meter.create_counter(
        "max.video.clips_encoded",
        unit="clips",
        description="Cumulative video clips run through the video encoder (cache misses).",
    ),  # type: ignore
    "max.video.tokens_encoded": _meter.create_counter(
        "max.video.tokens_encoded",
        unit="tokens",
        description="Cumulative merged video tokens produced by the video encoder.",
    ),  # type: ignore
    "max.video.encoding_time_milliseconds": _meter.create_histogram(
        "max.video.encoding_time_milliseconds",
        unit="ms",
        description="Per-batch video encoder wall-clock time.",
    ),  # type: ignore
    "max.video.frames_per_clip": _meter.create_histogram(
        "max.video.frames_per_clip",
        unit="frames",
        description="Sampled frame count per newly encoded video clip.",
    ),  # type: ignore
    "max.media.items": _meter.create_counter(
        "max.media.items",
        description=(
            "Cumulative media items the API server resolved into bytes, split by `media_kind` (`image`, `video`) and `source` (`inline` for a data URI, `url` for http(s), `file` for a local path). A rejected item is counted by `max_media_rejections_total` instead."
        ),
    ),  # type: ignore
    "max.media.resolve_time": _meter.create_histogram(
        "max.media.resolve_time",
        unit="ms",
        description=(
            "Per-item time to turn one media reference into bytes: the http(s) download, the base64 decode of a data URI, or the local file read. Excludes the image decode. Split by `source` (`inline`, `url`, `file`). Items on one request are resolved concurrently, so these windows overlap and their sum isn't the request's elapsed resolve time."
        ),
    ),  # type: ignore
    "max.media.item_size": _meter.create_histogram(
        "max.media.item_size",
        unit="bytes",
        description=(
            "Encoded size of one resolved media item as it arrived on the wire or in the request body. Not the decoded pixel buffer. Split by `media_kind`."
        ),
    ),  # type: ignore
    "max.media.image_size": _meter.create_histogram(
        "max.media.image_size",
        unit="pixels",
        description=(
            "Pixel count (`width * height`) of one image as it arrived, read from the header before any resize, split by `format`."
        ),
    ),  # type: ignore
    "max.media.image_decodes": _meter.create_counter(
        "max.media.image_decodes",
        description=(
            "Cumulative images whose pixels the API server actually decoded, split by `format`. An image the preprocess cache let the server skip is sized but not counted here."
        ),
    ),  # type: ignore
    # Milliseconds are in the name rather than in a declared unit: with
    # unit="ms" the exporter publishes
    # "max_media_image_decode_ms_milliseconds_total", which Datadog
    # spells "..._milliseconds.count" -- indistinguishable from the ".count"
    # it appends to "max.media.image_decodes" beside it, while carrying
    # milliseconds instead of a count.
    "max.media.image_decode_ms": _meter.create_counter(
        "max.media.image_decode_ms",
        description=(
            "Cumulative milliseconds of per-image codec decode, split by `format`. Excludes the per-architecture processor (resize, patchify, normalize), whose cost follows the pixel count rather than the container."
        ),
    ),  # type: ignore
    "max.media.rejections": _meter.create_counter(
        "max.media.rejections",
        description=(
            "Cumulative media items the API server refused, split by `reason`. Every value is an HTTP 400 to the client, so this counts client-visible failures rather than server faults."
        ),
    ),  # type: ignore
    "max.media.items_per_request": _meter.create_histogram(
        "max.media.items_per_request",
        unit="items",
        description=(
            "Media items of one `media_kind` carried by a single request, counted before any media is downloaded and before the per-request count caps, so a request refused for carrying too many items is still sampled. Requests carrying none of that kind aren't sampled, so the distribution starts at one."
        ),
    ),  # type: ignore
    "max.media.preprocess_cache_evictions": _meter.create_counter(
        "max.media.preprocess_cache_evictions",
        description=(
            "Cumulative preprocessed-media entries the tokenizer dropped to stay inside its byte budget, split by `media_kind`. Budget pressure only, so an idle-timeout reclaim isn't counted."
        ),
    ),  # type: ignore
    "max.media.preprocess_cache_size": _meter.create_gauge(
        "max.media.preprocess_cache_size",
        unit="bytes",
        description=(
            "Host bytes the tokenizer's preprocessed-media cache retains, split by `media_kind`. Sampled after tokenization, once per media request rather than on a timer."
        ),
    ),  # type: ignore
    "max.media.preprocess_cache_capacity": _meter.create_gauge(
        "max.media.preprocess_cache_capacity",
        unit="bytes",
        description=(
            "The byte budget `max_media_preprocess_cache_size_bytes` is evicted down to, split by `media_kind`. Zero means caching is disabled for that kind."
        ),
    ),  # type: ignore
    "max.tool_call.conformance_errors": _meter.create_counter(
        "max.tool_call.conformance_errors",
        description=(
            "Generated tool calls that failed the observability-only schema-conformance check, split by `outcome` label (`invalid_json`, `unknown_tool`, `schema_mismatch`)."
        ),
    ),  # type: ignore
    "max.structured_output.grammar_rejections": _meter.create_counter(
        "max.structured_output.grammar_rejections",
        description=(
            "Structured-output requests rejected at admission (HTTP 400) because the grammar backend couldn't compile the schema, split by `kind` label (`tool_grammar`, `json_schema`)."
        ),
    ),  # type: ignore
    "max.structured_output.grammar_build_time": _meter.create_histogram(
        "max.structured_output.grammar_build_time",
        unit="ms",
        description=(
            "Wall-clock time of an off-thread grammar-matcher build in "
            "AsyncGrammarGate, from the start of the backend's compile/create "
            "call to its completion. For a DI handoff, this build overlaps "
            "prefill's round trip when submitted at admission time; a build "
            "that outlasts that round trip can still gate the handoff's "
            "chunked-CE continuation."
        ),
    ),  # type: ignore
    "max.response_format.conformance_errors": _meter.create_counter(
        "max.response_format.conformance_errors",
        description=(
            "`response_format` (`json_schema`/`json_object`) responses whose final content failed the observability-only schema-conformance check, split by `outcome` label (`invalid_json`, `schema_mismatch`)."
        ),
    ),  # type: ignore
    "max.tool_call.responses": _meter.create_counter(
        "max.tool_call.responses",
        description=(
            "Count of chat-completion requests whose response contained at "
            "least one tool call, counted once per request however many "
            "calls it emitted. Against 'max.tool_call.requests' this "
            "gives the share of declared tool inventories the model actually "
            "draws on; tools declared but never called cost prompt tokens on "
            "every request. Not a denominator for "
            "'max.tool_call.conformance_errors', which counts "
            "individual generated calls rather than requests."
        ),
    ),  # type: ignore
    "max.tool_call.tools_per_request": _meter.create_histogram(
        "max.tool_call.tools_per_request",
        unit="tools",
        description=(
            "Distribution of tools declared per chat-completion request. "
            "Declared tools are rendered into the prompt, so inventory size "
            "explains a client's prompt length and grammar compile cost. "
            "Shares its population with 'max.tool_call.requests'."
        ),
    ),  # type: ignore
    "max.tool_call.requests": _meter.create_counter(
        "max.tool_call.requests",
        description=(
            "Count of chat-completion requests that supplied a non-empty "
            "'tools' array, split by the 'choice' tag (auto, required, "
            "named). Pairs with 'max.tool_call.responses' to show how "
            "often a declared inventory is actually used. A "
            "tool_choice='none' request is not counted: it cannot produce a "
            "tool call."
        ),
    ),  # type: ignore
    "max.structured_output.requests": _meter.create_counter(
        "max.structured_output.requests",
        description=(
            "Count of chat-completion requests that carried a "
            "response_format, split by the 'kind' tag (json_object, "
            "json_schema); a 'text' format is the unconstrained default and "
            "is not counted. Counted before the schema is compiled, so a "
            "request later rejected by "
            "'max.structured_output.grammar_rejections' still counts. "
            "Both kinds are exported at zero from startup."
        ),
    ),  # type: ignore
    "max.tokenizer.chat_encoder_requests": _meter.create_counter(
        "max.tokenizer.chat_encoder_requests",
        description=(
            "Count of chat requests a --tokenizer-impl encoder was asked to "
            "encode, split by the 'outcome' tag: 'custom' when it served the "
            "request, 'fallback' when it failed and HuggingFace encoded the "
            "request instead."
        ),
    ),  # type: ignore
}


class UnknownMetric(Exception):
    """Raised for a metric name that has no instrument."""


# Suffix identifying the exponential-histogram shadow of a histogram metric.
# When histogram_shadow_emission is enabled (MAX_SERVE_OTLP_METRICS_ENDPOINT
# is set; see common.configure_metrics), every histogram commit is mirrored
# to "<name><HISTOGRAM_SHADOW_SUFFIX>", recorded with
# ExponentialBucketHistogramAggregation instead of the explicit-bucket
# aggregation used by "<name>". Both reach the OTLP endpoint side by side for
# MXSERV-258 comparison; the shadow name is matched by a View instrument_name
# glob in common.py, so it never needs enumerating per metric.
HISTOGRAM_SHADOW_SUFFIX = ".exponential"

_histogram_shadow_emission_enabled = False


def configure_histogram_shadow_emission(enabled: bool) -> None:
    """Enable or disable mirroring histogram commits to their shadow metric.

    Called from common.configure_metrics with whether
    MAX_SERVE_OTLP_METRICS_ENDPOINT is set.
    """
    global _histogram_shadow_emission_enabled
    _histogram_shadow_emission_enabled = enabled


def _is_histogram(inst: object) -> bool:
    # SERVE_METRICS entries are created at import time, before any real
    # meter provider is configured, so they are always proxy instruments
    # here (never the concrete SDK ones) — check the proxy type directly.
    return isinstance(inst, api_instrument._ProxyHistogram)


# Build the exponential-histogram shadow instruments once, alongside the
# primary ones above: one per histogram, sharing its unit, named
# "<name><HISTOGRAM_SHADOW_SUFFIX>". Created unconditionally (instrument
# creation is cheap and inert until something records to it) so
# configure_histogram_shadow_emission can be toggled at runtime without
# touching instrument setup.
for _name, _inst in list(SERVE_METRICS.items()):
    if _is_histogram(_inst):
        _shadow_name = _name + HISTOGRAM_SHADOW_SUFFIX
        SERVE_METRICS[_shadow_name] = _meter.create_histogram(
            _shadow_name,
            description=f"Exponential-histogram shadow of {_name} (MXSERV-258).",
        )  # type: ignore
del _name, _inst


def _commit_measurement(
    instrument_name: str,
    value: NumberType,
    attributes: OtelAttributes,
    time_unix_nano: int,
) -> None:
    # find the instrument
    try:
        instrument = SERVE_METRICS[instrument_name]
    except KeyError as e:
        raise UnknownMetric(instrument_name) from e

    # Sometimes the instrument is a proxy.  Unrap it.
    if isinstance(instrument, get_args(API_PROXIES)):
        instrument = instrument._real_instrument
        # bail if there is no underlying instrument
        if instrument is None:
            logger.error(f"instrument is None for {instrument_name}")
            return

    # instrument should be one of the supported sdk types now
    if not isinstance(instrument, get_args(SDK_INSTRUMENTS)):
        # If you're hitting this, metrics were likely not configured properly.
        logger.error(
            f"instrument {instrument_name} is not one of the supported sdk types"
        )
        return

    # convert to an otel measurement
    m = measurement.Measurement(
        value,
        time_unix_nano,
        instrument,
        context.get_current(),
        attributes,
    )

    # record the measurement
    consumer = instrument._measurement_consumer
    consumer.consume_measurement(m)
    logger.debug(f"consumed measurement for {instrument_name}")


@dataclass
class MaxMeasurement:
    """Shim around the recording of a metric observation.

    Simplifies decoupling the observation of a metric from its recording.
    """

    instrument_name: str
    value: NumberType
    attributes: OtelAttributes | None = None
    time_unix_nano: int = field(default_factory=time.time_ns)

    def commit(self) -> None:
        """Commits the measurement to OTLP."""
        _commit_measurement(
            self.instrument_name,
            self.value,
            self.attributes,
            self.time_unix_nano,
        )

        if _histogram_shadow_emission_enabled:
            shadow_name = self.instrument_name + HISTOGRAM_SHADOW_SUFFIX
            if shadow_name in SERVE_METRICS:
                _commit_measurement(
                    shadow_name,
                    self.value,
                    self.attributes,
                    self.time_unix_nano,
                )


TelemetryFn = Callable[[MaxMeasurement], None]


class MetricClient(abc.ABC):
    """Base class for the clients that commit measurements."""

    @abc.abstractmethod
    def send_measurement(self, metric: MaxMeasurement) -> None:
        """Commits the measurement."""
        ...

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Group a burst of measurements into a single flush.

        A caller that knows it is about to emit many measurements at once
        (such as the per-iteration scheduler metrics) can wrap them in a
        transaction so a client that crosses a process boundary batches them
        into one packet instead of one send per measurement. Measurements
        emitted outside a transaction are sent immediately.

        Clients that record in-process ignore this and always emit
        immediately; the default implementation is a no-op.
        """
        yield

    @abc.abstractmethod
    def cross_process_factory(
        self,
        settings: Settings,
    ) -> Callable[[], AbstractAsyncContextManager[MetricClient]]:
        """Get a copier for use of this client in another process.

        To use a MetricClient across processes, call cross_process_factory in
        the parent process and pass the result across the process boundary.
        Then in the child process, use an 'async with' to get a semantically
        identical MetricClient that can be used.

        This is needed because some metric clients require reinitialization on
        the other side of a process boundary before they can be safely used.
        """
        ...


@asynccontextmanager
async def _trivially_picklable_xprocess_factory(
    client: MetricClient,
) -> AsyncGenerator[MetricClient, None]:
    yield client


class NoopClient(MetricClient):
    """Metric client that drops measurements."""

    def send_measurement(self, m: MaxMeasurement) -> None:
        """Drops the measurement."""

    def cross_process_factory(
        self,
        settings: Settings,
    ) -> Callable[[], AbstractAsyncContextManager[MetricClient]]:
        """Returns the factory that rebuilds this client in another process."""
        return functools.partial(_trivially_picklable_xprocess_factory, self)


class SyncClient(MetricClient):
    """Metric client that commits each measurement inline."""

    def send_measurement(self, m: MaxMeasurement) -> None:
        """Commits the measurement inline."""
        m.commit()

    def cross_process_factory(
        self,
        settings: Settings,
    ) -> Callable[[], AbstractAsyncContextManager[MetricClient]]:
        """Returns the factory that rebuilds this client in another process."""
        return functools.partial(_trivially_picklable_xprocess_factory, self)


class _AsyncMetrics:
    """Centralizes metrics to hide the OTEL dependency and its schema changes.

    Produces metric measurements to be consumed elsewhere.
    """

    def __init__(self) -> None:
        self.client: MetricClient = NoopClient()
        self.extra_attributes: dict[str, str] = {}

    def configure(
        self,
        client: MetricClient,
        extra_attributes: dict[str, str] | None = None,
    ) -> None:
        self.client = client
        self.extra_attributes = extra_attributes or {}

    def transaction(self) -> AbstractContextManager[None]:
        """Group the measurements emitted in the ``with`` block into one flush.

        See :meth:`MetricClient.transaction`. Use this to wrap a burst of
        measurements that are always produced together so a client that
        crosses the telemetry process boundary sends them as a single packet::

            with METRICS.transaction():
                METRICS.batch_size(...)
                METRICS.batch_execution_time(...)
                ...

        Only cross-process clients batch; in-process clients treat this as a
        no-op and emit each measurement immediately.
        """
        return self.client.transaction()

    def export_zero_client_disconnect_cancellations(self) -> None:
        """Exports every phase at zero, so a quiet window reads as zero."""
        for phase in get_args(_DisconnectPhase):
            self.client.send_measurement(
                MaxMeasurement(
                    "max.request.client_disconnect_cancellations",
                    0,
                    {**self.extra_attributes, "phase": phase},
                ),
            )

    def client_disconnect_cancellation(self, phase: _DisconnectPhase) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.request.client_disconnect_cancellations",
                1,
                {**self.extra_attributes, "phase": phase},
            ),
        )

    def request_count(self, responseCode: int, urlPath: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.request_count",
                1,
                {
                    **self.extra_attributes,
                    "code": f"{responseCode:d}",
                    "path": urlPath,
                },
            ),
        )

    def request_time(self, value: float, urlPath: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.request_time",
                value,
                {**self.extra_attributes, "path": urlPath},
            ),
        )

    def input_time(self, value: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.input_processing_time", value, self.extra_attributes
            ),
        )

    def output_time(self, value: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.output_processing_time", value, self.extra_attributes
            ),
        )

    def ttft(self, value: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.time_to_first_token", value, self.extra_attributes
            ),
        )

    def ttft_by_prompt_length(self, value: float, prompt_tokens: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.time_to_first_token_by_prompt_length",
                value,
                {
                    **self.extra_attributes,
                    "prompt_length": _prompt_length_bucket(prompt_tokens),
                },
            ),
        )

    def input_tokens(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.num_input_tokens", value, self.extra_attributes
            ),
        )

    def output_tokens(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.num_output_tokens", value, self.extra_attributes
            ),
        )

    def reqs_queued(self, value: int) -> None:
        """Publish the current depth of the scheduler's CE / prefill queue.

        ``max.num_requests_queued`` is a synchronous gauge: every call
        replaces the previously reported value rather than accumulating.
        Schedulers should call this once per iteration with
        ``len(all_ce_reqs)`` (or the equivalent for their queue layout) at
        the same point that the "Pending: N reqs" log line is computed.
        """
        self.client.send_measurement(
            MaxMeasurement(
                "max.num_requests_queued", value, self.extra_attributes
            ),
        )

    def reqs_running(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.num_requests_running", value, self.extra_attributes
            ),
        )

    def reqs_awaiting_admission(self, value: int) -> None:
        """Adjust the count of API-side requests not yet handed to the worker.

        ``max.num_requests_awaiting_admission`` is an up/down counter:
        call with ``1`` when a request is accepted by the API server (before
        tokenization) and ``-1`` just before it is enqueued to the model
        worker. A persistently high value means requests are backing up in the
        API server (such as tokenization) rather than in the scheduler queue.
        """
        self.client.send_measurement(
            MaxMeasurement(
                "max.num_requests_awaiting_admission",
                value,
                self.extra_attributes,
            ),
        )

    def requests_awaiting_admission_dist(self, value: int) -> None:
        """Record a sample of the ingress backlog for distribution analysis.

        Companion to :meth:`reqs_awaiting_admission` (the live up/down
        counter): a periodic sample of the same running count is fed into the
        ``max.requests_awaiting_admission`` histogram so p50/p99 ingress
        backlog over time can be recovered.
        """
        self.client.send_measurement(
            MaxMeasurement(
                "max.requests_awaiting_admission",
                value,
                self.extra_attributes,
            ),
        )

    def responses_buffered(self, value: int) -> None:
        """Publish the current egress backlog (sum of output-queue depths).

        ``max.num_responses_buffered`` is a synchronous gauge: every call
        replaces the previously reported value. The API server should sample
        it periodically with the total number of model-worker responses
        received but not yet consumed by the streaming layer.
        """
        self.client.send_measurement(
            MaxMeasurement(
                "max.num_responses_buffered",
                value,
                self.extra_attributes,
            ),
        )

    def responses_buffered_dist(self, value: int) -> None:
        """Record a sample of the egress backlog for distribution analysis.

        Companion to :meth:`responses_buffered` (the live gauge): the same
        periodic sample is also fed into the ``max.responses_buffered``
        histogram so p50/p99 backlog over time can be recovered, which a
        scrape-interval gauge sample alone cannot provide.
        """
        self.client.send_measurement(
            MaxMeasurement(
                "max.responses_buffered", value, self.extra_attributes
            ),
        )

    def response_queue_time(self, ms: float) -> None:
        """Record how long a response waited in the API output queue (ms)."""
        self.client.send_measurement(
            MaxMeasurement(
                "max.response_queue_time", ms, self.extra_attributes
            ),
        )

    def model_load_time(self, ms: float, component: str | None = None) -> None:
        """Record a model-worker startup duration in milliseconds.

        Args:
            ms: The duration in milliseconds.
            component: Optional phase name (such as ``"build"``, ``"compile"``,
                ``"init"``, ``"graph_capture"``, ``"pinned_memory"``,
                ``"spawn"``, ``"total"``). Recorded as the ``component`` tag
                so a single metric can be split by startup phase. When
                omitted, records the untagged model-load aggregate.
        """
        attributes = self.extra_attributes
        if component is not None:
            attributes = {**attributes, "component": component}
        self.client.send_measurement(
            MaxMeasurement("max.model_load_time", ms, attributes),
        )

    def itl(self, ms: float) -> None:
        self.client.send_measurement(
            MaxMeasurement("max.itl", ms, self.extra_attributes),
        )

    def time_per_output_token(self, ms: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.time_per_output_token", ms, self.extra_attributes
            ),
        )

    def pipeline_load(self, name: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.pipeline_load",
                1,
                {**self.extra_attributes, "model": name},
            ),
        )

    def batch_size(self, size: int, batch_type: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.batch.size",
                size,
                {**self.extra_attributes, "batch_type": batch_type},
            ),
        )

    def scheduler_batch_step_time(self, value: float, batch_type: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.scheduler.batch_step_time",
                value,
                {**self.extra_attributes, "batch_type": batch_type},
            ),
        )

    def batch_execution_time(
        self, execution_time: float, batch_type: str
    ) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.batch.execution_time",
                execution_time,
                {**self.extra_attributes, "batch_type": batch_type},
            ),
        )

    def di_api_pre_tokenize(self, ms: float) -> None:
        """API-local time from HTTP accept to tokenize start."""
        self.client.send_measurement(
            MaxMeasurement(
                "max.di.api_pre_tokenize", ms, self.extra_attributes
            ),
        )

    def di_api_submit(self, ms: float) -> None:
        """API-local time from tokenize done to model-worker ZMQ put."""
        self.client.send_measurement(
            MaxMeasurement("max.di.api_submit", ms, self.extra_attributes),
        )

    def di_mw_queue_wait(self, ms: float) -> None:
        """Decode-worker time from first ZMQ pull to admission enqueue."""
        self.client.send_measurement(
            MaxMeasurement("max.di.mw_queue_wait", ms, self.extra_attributes),
        )

    def di_decode_admission_queue_wait_time(self, ms: float) -> None:
        """Decode-local wait in decode's own admission queue before dispatch."""
        self.client.send_measurement(
            MaxMeasurement(
                "max.di.decode_admission_queue_wait_time",
                ms,
                self.extra_attributes,
            ),
        )

    def di_decode_send_time(self, ms: float) -> None:
        """Decode-side PrefillRequest build+serialize+send duration."""
        self.client.send_measurement(
            MaxMeasurement(
                "max.di.decode_send_time", ms, self.extra_attributes
            ),
        )

    def di_dispatch_rtt(self, ms: float) -> None:
        """Decode-clock round trip to prefill's 'arrived' ack ping."""
        self.client.send_measurement(
            MaxMeasurement("max.di.dispatch_rtt", ms, self.extra_attributes),
        )

    def di_prefill_span(self, ms: float) -> None:
        """Decode-clock span between prefill's 'arrived' and 'ce_done' pings."""
        self.client.send_measurement(
            MaxMeasurement("max.di.prefill_span", ms, self.extra_attributes),
        )

    def di_reply_rtt(self, ms: float) -> None:
        """Decode-clock span from the 'ce_done' ping to the real reply."""
        self.client.send_measurement(
            MaxMeasurement("max.di.reply_rtt", ms, self.extra_attributes),
        )

    def di_prefill_queue_wait_time(self, ms: float) -> None:
        """Prefill-local queue wait before a request's first CE batch."""
        self.client.send_measurement(
            MaxMeasurement(
                "max.di.prefill_queue_wait_time",
                ms,
                self.extra_attributes,
            ),
        )

    def ce_pool_wait_time(self, ms: float) -> None:
        """Time from a CE request's arrival to its DP balancer binding."""
        self.client.send_measurement(
            MaxMeasurement("max.ce_pool_wait_time", ms, self.extra_attributes),
        )

    def di_decode_postprocess_time(self, ms: float) -> None:
        """Decode-local time from PrefillResponse receipt to response queue."""
        self.client.send_measurement(
            MaxMeasurement(
                "max.di.decode_postprocess_time",
                ms,
                self.extra_attributes,
            ),
        )

    def di_early_sync_time(self, ms: float) -> None:
        """Duration of the early-sync guard's blocking commit, when it fired.

        Covers the commit of the previous batch's structured-output FSM state (see
        ``_should_early_sync_prev_batch``).
        """
        self.client.send_measurement(
            MaxMeasurement("max.di.early_sync_time", ms, self.extra_attributes),
        )

    def di_ce_preempted_tg_iteration_count(self) -> None:
        """Counter: CE batches admitting work while TG requests are pending.

        Those TG requests get zero progress that iteration.
        """
        self.client.send_measurement(
            MaxMeasurement(
                "max.di.ce_preempted_tg_iteration_count",
                1,
                self.extra_attributes,
            ),
        )

    def di_ce_preempted_tg_pending_count(self, count: int) -> None:
        """Histogram: how many TG requests each such preemption left pending.

        Paired with ``di_ce_preempted_tg_iteration_count``, which counts how often.
        """
        self.client.send_measurement(
            MaxMeasurement(
                "max.di.ce_preempted_tg_pending_count",
                count,
                self.extra_attributes,
            ),
        )

    def di_handoff_to_first_token_time(self, ms: float) -> None:
        """Wall-clock time from a prefill-to-decode handoff to token 1 at the API.

        Starts when the handoff lands: its token is discarded, and the request is
        re-queued as a one-token chunked-CE continuation. Ends when decode's own
        re-forward sends token 1 to the API process.

        A TTFT contribution specific to the handoff path; the ordinary path's token
        already arrives with the ``PrefillResponse``, so it has no equivalent wait.
        """
        self.client.send_measurement(
            MaxMeasurement(
                "max.di.handoff_to_first_token_time",
                ms,
                self.extra_attributes,
            ),
        )

    def cache_num_used_blocks(self, num_used_blocks: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.cache.num_used_blocks",
                num_used_blocks,
                self.extra_attributes,
            ),
        )

    def cache_num_total_blocks(self, total_blocks: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.cache.num_total_blocks",
                total_blocks,
                self.extra_attributes,
            ),
        )

    def cache_jenga_small_blocks(
        self, blocks: int, *, replica: str, leaf: str, state: str
    ) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.cache.jenga_small_blocks",
                blocks,
                {
                    **self.extra_attributes,
                    "replica": replica,
                    "leaf": leaf,
                    "state": state,
                },
            ),
        )

    def cache_jenga_huge_blocks(
        self, blocks: int, *, replica: str, leaf: str, state: str
    ) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.cache.jenga_huge_blocks",
                blocks,
                {
                    **self.extra_attributes,
                    "replica": replica,
                    "leaf": leaf,
                    "state": state,
                },
            ),
        )

    def cache_request_prefix_coverage(self, coverage: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.cache.request_prefix_coverage",
                coverage,
                self.extra_attributes,
            ),
        )

    def cache_hits(self, hits: int, *, tier: str) -> None:
        """Records prefix-cache hit tokens served by ``tier``.

        Args:
            hits: Prompt tokens served from the cache.
            tier: Which tier served them, one of ``g0`` (device prefix cache) or
                ``external`` (the KV connector, tier unresolved). Shared with
                Mach's ``mach.cache.hits`` so one dashboard panel covers both
                engines; Mach additionally resolves ``g1`` and ``g2``.

        Every hit point carries a ``tier``, so the tiers sum to the counter's
        untagged total and a query that does not group by ``tier`` reads exactly
        what it read before the attribute existed.
        """
        self.client.send_measurement(
            MaxMeasurement(
                "max.cache.hits",
                hits,
                {**self.extra_attributes, "tier": tier},
            ),
        )

    def cache_misses(self, cache_misses: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.cache.misses", cache_misses, self.extra_attributes
            ),
        )

    def preemption(self) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.cache.preemption_count", 1, self.extra_attributes
            ),
        )

    def connector_cordon_wait_time(self, ms: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.cache.connector_cordon_wait_time",
                ms,
                self.extra_attributes,
            ),
        )

    def connector_cordon_passes(self, passes: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.cache.connector_cordon_passes",
                passes,
                self.extra_attributes,
            ),
        )

    def connector_lookup_admission_stop(
        self, reason: LookupAdmissionStop
    ) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.cache.connector_lookup_admission_stops",
                1,
                {**self.extra_attributes, "reason": reason},
            ),
        )

    def vision_images_encoded(self, images: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.vision.images_encoded",
                images,
                self.extra_attributes,
            ),
        )

    def vision_images_cached(self, images: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.vision.images_cached",
                images,
                self.extra_attributes,
            ),
        )

    def vision_patches_encoded(self, patches: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.vision.patches_encoded",
                patches,
                self.extra_attributes,
            ),
        )

    def vision_tokens_encoded(self, tokens: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.vision.tokens_encoded",
                tokens,
                self.extra_attributes,
            ),
        )

    def vision_preprocess_cache_hits(self, images: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.vision.preprocess_cache_hits",
                images,
                self.extra_attributes,
            ),
        )

    def vision_preprocess_cache_misses(self, images: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.vision.preprocess_cache_misses",
                images,
                self.extra_attributes,
            ),
        )

    def image_admission_decode_time(self, value: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.vision.image_admission_decode_time",
                value,
                self.extra_attributes,
            ),
        )

    def vision_cache_hit_rate(self, hit_rate: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.vision.cache_hit_rate",
                hit_rate,
                self.extra_attributes,
            ),
        )

    def video_clips_encoded(self, clips: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.video.clips_encoded",
                clips,
                self.extra_attributes,
            ),
        )

    def video_tokens_encoded(self, tokens: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.video.tokens_encoded",
                tokens,
                self.extra_attributes,
            ),
        )

    def video_encoding_time_milliseconds(self, time_ms: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.video.encoding_time_milliseconds",
                time_ms,
                self.extra_attributes,
            ),
        )

    def media_items(self, media_kind: str, source: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.media.items",
                1,
                {
                    **self.extra_attributes,
                    "media_kind": media_kind,
                    "source": source,
                },
            ),
        )

    def media_resolve_time(self, value: float, source: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.media.resolve_time",
                value,
                {**self.extra_attributes, "source": source},
            ),
        )

    def media_item_size(self, num_bytes: int, media_kind: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.media.item_size",
                num_bytes,
                {**self.extra_attributes, "media_kind": media_kind},
            ),
        )

    def media_image_size(self, pixels: int, image_format: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.media.image_size",
                pixels,
                {**self.extra_attributes, "format": image_format},
            ),
        )

    def media_image_decodes(self, image_format: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.media.image_decodes",
                1,
                {**self.extra_attributes, "format": image_format},
            ),
        )

    def media_image_decode_ms(self, value: float, image_format: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.media.image_decode_ms",
                value,
                {**self.extra_attributes, "format": image_format},
            ),
        )

    def media_rejections(self, reason: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.media.rejections",
                1,
                {**self.extra_attributes, "reason": reason},
            ),
        )

    def media_items_per_request(self, items: int, media_kind: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.media.items_per_request",
                items,
                {**self.extra_attributes, "media_kind": media_kind},
            ),
        )

    def media_preprocess_cache_evictions(
        self, evictions: int, media_kind: str
    ) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.media.preprocess_cache_evictions",
                evictions,
                {**self.extra_attributes, "media_kind": media_kind},
            ),
        )

    def media_preprocess_cache_size(
        self, num_bytes: int, media_kind: str
    ) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.media.preprocess_cache_size",
                num_bytes,
                {**self.extra_attributes, "media_kind": media_kind},
            ),
        )

    def media_preprocess_cache_capacity(
        self, num_bytes: int, media_kind: str
    ) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.media.preprocess_cache_capacity",
                num_bytes,
                {**self.extra_attributes, "media_kind": media_kind},
            ),
        )

    def video_frames_per_clip(self, frames: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.video.frames_per_clip",
                frames,
                self.extra_attributes,
            ),
        )

    def _with_turn(self, turn: ConversationTurn | None) -> Mapping[str, str]:
        """Returns the extra attributes, plus ``turn`` when it is known."""
        if turn is None:
            return self.extra_attributes
        return {**self.extra_attributes, "turn": turn}

    def input_tokens_per_request(
        self, value: int, turn: ConversationTurn | None = None
    ) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.input_tokens_per_request",
                value,
                self._with_turn(turn),
            ),
        )

    def prefix_coverage_per_request(
        self, coverage: float, turn: ConversationTurn | None = None
    ) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.prefix_coverage_per_request",
                coverage,
                self._with_turn(turn),
            ),
        )

    def output_tokens_per_request(
        self, value: int, turn: ConversationTurn | None = None
    ) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.output_tokens_per_request",
                value,
                self._with_turn(turn),
            ),
        )

    def dkv_nixl_read_latency(self, latency_ms: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.dkv.nixl_read_latency",
                latency_ms,
                self.extra_attributes,
            ),
        )

    def dkv_nixl_read_latency_max(self, latency_ms: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.dkv.nixl_read_latency_max",
                latency_ms,
                self.extra_attributes,
            ),
        )

    def dkv_nixl_write_latency(self, latency_ms: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.dkv.nixl_write_latency",
                latency_ms,
                self.extra_attributes,
            ),
        )

    def dkv_rpc_acquire_latency(self, latency_ms: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.dkv.rpc_acquire_latency",
                latency_ms,
                self.extra_attributes,
            ),
        )

    def dkv_rpc_read_latency(self, latency_ms: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.dkv.rpc_read_latency",
                latency_ms,
                self.extra_attributes,
            ),
        )

    def dkv_connected_clients(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.dkv.connected_clients",
                value,
                self.extra_attributes,
            ),
        )

    def dkv_total_clients(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.dkv.total_clients",
                value,
                self.extra_attributes,
            ),
        )

    def dkv_reconnect_attempts(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.dkv.reconnect_attempts",
                value,
                self.extra_attributes,
            ),
        )

    def dkv_settle_queue_depth(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.dkv.settle_queue_depth",
                value,
                self.extra_attributes,
            ),
        )

    def dkv_settle_queue_oldest_age_ms(self, value: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.dkv.settle_queue_oldest_age_ms",
                value,
                self.extra_attributes,
            ),
        )

    def dkv_owner_releases_halted(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.dkv.owner_releases_halted",
                value,
                self.extra_attributes,
            ),
        )

    def dkv_settle_jobs_finished(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.dkv.settle_jobs_finished",
                value,
                self.extra_attributes,
            ),
        )

    def dkv_peer_attaches(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.dkv.peer_attaches",
                value,
                self.extra_attributes,
            ),
        )

    def dkv_peer_attach_failures(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.dkv.peer_attach_failures",
                value,
                self.extra_attributes,
            ),
        )

    def dkv_peers_dropped(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.dkv.peers_dropped",
                value,
                self.extra_attributes,
            ),
        )

    def dkv_peer_loads(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.dkv.peer_loads",
                value,
                self.extra_attributes,
            ),
        )

    def dkv_peer_load_failures(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.dkv.peer_load_failures",
                value,
                self.extra_attributes,
            ),
        )

    def dkv_peer_transfer_failures(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.dkv.peer_transfer_failures",
                value,
                self.extra_attributes,
            ),
        )

    def dkv_peer_metadata_refreshes(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.dkv.peer_metadata_refreshes",
                value,
                self.extra_attributes,
            ),
        )

    def dkv_peer_metadata_refresh_failures(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.dkv.peer_metadata_refresh_failures",
                value,
                self.extra_attributes,
            ),
        )

    def dkv_hints_rejected(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.dkv.hints_rejected",
                value,
                self.extra_attributes,
            ),
        )

    def dkv_hints_rejected_by_reason(self, value: int, reason: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.dkv.hints_rejected_by_reason",
                value,
                {**self.extra_attributes, "reason": reason},
            ),
        )

    def dkv_rpc_calls(self, value: int, op: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.dkv.rpc_calls",
                value,
                {**self.extra_attributes, "op": op},
            ),
        )

    def dkv_rpc_time_ms(self, value: float, op: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.dkv.rpc_time_ms",
                value,
                {**self.extra_attributes, "op": op},
            ),
        )

    def dkv_rpc_socket_wait_ms(self, value: float, op: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.dkv.rpc_socket_wait_ms",
                value,
                {**self.extra_attributes, "op": op},
            ),
        )

    def dkv_lock_waits(self, value: int, entry: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.dkv.lock_waits",
                value,
                {**self.extra_attributes, "entry": entry},
            ),
        )

    def dkv_lock_wait_ms(self, value: float, entry: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.dkv.lock_wait_ms",
                value,
                {**self.extra_attributes, "entry": entry},
            ),
        )

    def dkv_touches_deduplicated(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.dkv.touches_deduplicated",
                value,
                self.extra_attributes,
            ),
        )

    def dkv_prepare_load_degraded(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.dkv.prepare_load_degraded",
                value,
                self.extra_attributes,
            ),
        )

    def dkv_load_with_plan_degraded(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.dkv.load_with_plan_degraded",
                value,
                self.extra_attributes,
            ),
        )

    def dkv_load_with_plan_expired(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.dkv.load_with_plan_expired",
                value,
                self.extra_attributes,
            ),
        )

    def dkv_orphan_cleanups_reissued(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.dkv.orphan_cleanups_reissued",
                value,
                self.extra_attributes,
            ),
        )

    def dkv_orphan_cleanups_abandoned(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.dkv.orphan_cleanups_abandoned",
                value,
                self.extra_attributes,
            ),
        )

    def dkv_orphan_cleanups_dropped(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.dkv.orphan_cleanups_dropped",
                value,
                self.extra_attributes,
            ),
        )

    def dkv_orphan_cleanups_refused(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.dkv.orphan_cleanups_refused",
                value,
                self.extra_attributes,
            ),
        )

    def dkv_orphan_cleanups_expired(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.dkv.orphan_cleanups_expired",
                value,
                self.extra_attributes,
            ),
        )

    def dkv_orphan_queue_depth(self, value: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.dkv.orphan_queue_depth",
                value,
                self.extra_attributes,
            ),
        )

    def dkv_blocked_calls(self, value: int, entry: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.dkv.blocked_calls",
                value,
                {**self.extra_attributes, "entry": entry},
            ),
        )

    def dkv_blocked_ms(self, value: float, entry: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.dkv.blocked_ms",
                value,
                {**self.extra_attributes, "entry": entry},
            ),
        )

    def dkv_read_blocks(self, blocks: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.dkv.read_blocks",
                blocks,
                self.extra_attributes,
            ),
        )

    def spec_decode_acceptance_rate_per_position(
        self, position: int, acceptance_rate: float
    ) -> None:
        """Emit draft token acceptance rate for a specific position.

        Args:
            position: The draft token position (0-indexed).
            acceptance_rate: The acceptance rate as a percentage (0-100).
        """
        self.client.send_measurement(
            MaxMeasurement(
                "max.spec_decode.acceptance_rate_per_position",
                acceptance_rate,
                {**self.extra_attributes, "position": str(position)},
            ),
        )

    def batch_input_tokens(self, value: int, batch_type: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.batch.input_tokens",
                value,
                {**self.extra_attributes, "batch_type": batch_type},
            ),
        )

    def batch_context_tokens(self, value: int, batch_type: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.batch.context_tokens",
                value,
                {**self.extra_attributes, "batch_type": batch_type},
            ),
        )

    def batch_creation_time(self, ms: float, batch_type: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.batch.creation_time",
                ms,
                {**self.extra_attributes, "batch_type": batch_type},
            ),
        )

    def batch_prompt_throughput(self, tps: float, batch_type: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.batch.prompt_throughput",
                tps,
                {**self.extra_attributes, "batch_type": batch_type},
            ),
        )

    def batch_generation_throughput(self, tps: float, batch_type: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.batch.generation_throughput",
                tps,
                {**self.extra_attributes, "batch_type": batch_type},
            ),
        )

    def dp_active_token_occupancy(self, pct: float, batch_type: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.dp_active_token_occupancy",
                pct,
                {**self.extra_attributes, "batch_type": batch_type},
            ),
        )

    def dp_context_token_occupancy(self, pct: float, batch_type: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.dp_context_token_occupancy",
                pct,
                {**self.extra_attributes, "batch_type": batch_type},
            ),
        )

    def dp_active_tokens(self, value: int, batch_type: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.dp_active_tokens",
                value,
                {**self.extra_attributes, "batch_type": batch_type},
            ),
        )

    def dp_step_capacity_tokens(self, value: int, batch_type: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.dp_step_capacity_tokens",
                value,
                {**self.extra_attributes, "batch_type": batch_type},
            ),
        )

    def batch_terminated_reqs(self, value: int, batch_type: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.batch.terminated_reqs",
                value,
                {**self.extra_attributes, "batch_type": batch_type},
            ),
        )

    def batch_pending_reqs(self, value: int, batch_type: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.batch.pending_reqs",
                value,
                {**self.extra_attributes, "batch_type": batch_type},
            ),
        )

    def cache_used_kv_pct(self, ratio: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.cache.used_kv_pct", ratio, self.extra_attributes
            ),
        )

    def cache_used_host_kv_pct(self, ratio: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.cache.used_host_kv_pct",
                ratio,
                self.extra_attributes,
            ),
        )

    def cache_device_blocks_served(self, count: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.cache.device_blocks_served",
                count,
                self.extra_attributes,
            ),
        )

    def cache_h2d_bytes_copied(self, count: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.cache.h2d_bytes_copied",
                count,
                self.extra_attributes,
            ),
        )

    def cache_cross_replica_blocks_copied(self, count: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.cache.cross_replica_blocks_copied",
                count,
                self.extra_attributes,
            ),
        )

    def cache_cross_replica_bytes_copied(self, count: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.cache.cross_replica_bytes_copied",
                count,
                self.extra_attributes,
            ),
        )

    def cache_d2h_bytes_copied(self, count: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.cache.d2h_bytes_copied",
                count,
                self.extra_attributes,
            ),
        )

    def cache_connector_loads_refused(self, count: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.cache.connector_loads_refused",
                count,
                self.extra_attributes,
            ),
        )

    def cache_connector_load_failures(self, count: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.cache.connector_load_failures",
                count,
                self.extra_attributes,
            ),
        )

    def cache_connector_offload_blocks_dropped(self, count: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.cache.connector_offload_blocks_dropped",
                count,
                self.extra_attributes,
            ),
        )

    def cache_disk_bytes_read(self, count: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.cache.disk_bytes_read",
                count,
                self.extra_attributes,
            ),
        )

    def cache_disk_bytes_written(self, count: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.cache.disk_bytes_written",
                count,
                self.extra_attributes,
            ),
        )

    def cache_used_disk_kv_pct(self, ratio: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.cache.used_disk_kv_pct",
                ratio,
                self.extra_attributes,
            ),
        )

    def spec_decode_avg_acceptance_length(self, length: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.spec_decode.avg_acceptance_length",
                length,
                self.extra_attributes,
            ),
        )

    def dkv_nixl_read_gib_per_s(self, gib_per_s: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.dkv.nixl_read_gib_per_s",
                gib_per_s,
                self.extra_attributes,
            ),
        )

    def dkv_nixl_write_gib_per_s(self, gib_per_s: float) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.dkv.nixl_write_gib_per_s",
                gib_per_s,
                self.extra_attributes,
            ),
        )

    def tool_call_responses(self) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.tool_call.responses",
                1,
                self.extra_attributes,
            ),
        )

    def tool_call_tools_per_request(self, count: int) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.tool_call.tools_per_request",
                count,
                self.extra_attributes,
            ),
        )

    def tool_call_requests(self, choice: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.tool_call.requests",
                1,
                {**self.extra_attributes, "choice": choice},
            ),
        )

    def export_zero_structured_output_requests(self) -> None:
        """Exports every kind at zero, so a quiet window reads as zero."""
        for kind in get_args(_StructuredOutputKind):
            self.client.send_measurement(
                MaxMeasurement(
                    "max.structured_output.requests",
                    0,
                    {**self.extra_attributes, "kind": kind},
                ),
            )

    def structured_output_requests(self, kind: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.structured_output.requests",
                1,
                {**self.extra_attributes, "kind": kind},
            ),
        )

    def tokenizer_chat_encoder_requests(self, count: int, outcome: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.tokenizer.chat_encoder_requests",
                count,
                {**self.extra_attributes, "outcome": outcome},
            ),
        )

    def tool_call_conformance_error(self, outcome: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.tool_call.conformance_errors",
                1,
                {**self.extra_attributes, "outcome": outcome},
            ),
        )

    def structured_output_grammar_rejection(self, kind: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.structured_output.grammar_rejections",
                1,
                {**self.extra_attributes, "kind": kind},
            ),
        )

    def structured_output_grammar_build_time(self, ms: float) -> None:
        """Wall-clock time of an off-thread grammar-matcher build."""
        self.client.send_measurement(
            MaxMeasurement(
                "max.structured_output.grammar_build_time",
                ms,
                self.extra_attributes,
            ),
        )

    def response_format_conformance_error(self, outcome: str) -> None:
        self.client.send_measurement(
            MaxMeasurement(
                "max.response_format.conformance_errors",
                1,
                {**self.extra_attributes, "outcome": outcome},
            ),
        )


METRICS = _AsyncMetrics()
