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
from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING

import opentelemetry.trace as otel_trace
from max._core import request_context as _request_context
from max._core.profiler import range_begin_with_id, range_end
from max.pipelines.context import (
    TextAndVisionContext,
    TextContext,
    TextGenerationOutput,
)
from max.pipelines.kv_cache import PagedKVCacheManagerInterface
from max.pipelines.lib import (
    PIPELINE_REGISTRY,
    MemoryPlan,
    OverlapTextGenerationPipeline,
    PipelineConfig,
    TextGenerationPipeline,
)
from max.pipelines.modeling.types import (
    Pipeline,
    RequestID,
    TextGenerationInputs,
)
from max.profiler import Tracer, traced
from max.serve.queue import (
    MAXPullQueue,
    MAXPushQueue,
    drain_queue,
)
from max.serve.scheduler.interface import Scheduler
from max.serve.scheduler_result import SchedulerResult
from max.serve.telemetry.common import (
    _batch_id_ctx,
    _BatchLinks,
    batch_spans_enabled,
)
from opentelemetry import propagate as otel_propagate
from opentelemetry.context import Context as OtelContext

from .base import SchedulerProgress
from .batch_constructor import TextBatchConstructor
from .batch_constructor.text_batch_constructor import BatchSchedulingStrategy
from .config import TokenGenerationSchedulerConfig
from .dp_padding import DPBatchPadder
from .utils import SchedulerLogger, get_cancelled_reqs

if TYPE_CHECKING:
    from max.serve.telemetry._kernel_capture import KernelCapture

logger = logging.getLogger("max.serve")
_tracer = otel_trace.get_tracer("max.serve")

# The OTel SDK's default span link limit; the SDK would build and then drop
# any links beyond it.
_MAX_BATCH_LINKS = 128


def _tracing_enabled() -> bool:
    """Whether a real TracerProvider is configured, vs. the OTel no-op default.

    The spans created below are cheap no-ops when tracing is disabled, but the
    bookkeeping around them (set construction/diffing every batch) is not.
    Callers use this to skip that bookkeeping entirely on the hot path.
    """
    return not isinstance(
        otel_trace.get_tracer_provider(), otel_trace.ProxyTracerProvider
    )


def _parent_trace_context(context: TextContext) -> OtelContext | None:
    """Re-extracts the phase spans' parent from ``context.trace_carrier``.

    That is the request's ``max.request`` span where the handler set one,
    otherwise its HTTP server span. ``trace_carrier`` was serialized by the
    API process (see ``_trace_context.inject_trace_carrier``) since a live
    ``Context`` can't cross the process boundary. Returns None (a root span)
    when neither was present, or the request arrived before this propagation
    existed.
    """
    if context.trace_carrier is None:
        return None
    return otel_propagate.extract(context.trace_carrier)


class TokenGenerationScheduler(Scheduler):
    def __init__(
        self,
        scheduler_config: TokenGenerationSchedulerConfig,
        pipeline: Pipeline[
            TextGenerationInputs[TextContext], TextGenerationOutput
        ],
        *,
        request_queue: MAXPullQueue[TextContext | TextAndVisionContext],
        response_queue: MAXPushQueue[
            dict[RequestID, SchedulerResult[TextGenerationOutput]]
        ],
        cancel_queue: MAXPullQueue[list[RequestID]],
        kv_cache: PagedKVCacheManagerInterface,
        support_empty_batches: bool = False,
        dp_padder: DPBatchPadder | None = None,
        max_pending_requests: int | None = None,
        kernel_capture: KernelCapture | None = None,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.pipeline = pipeline
        # The model worker configures tracing before it builds the scheduler.
        self._tracing = _tracing_enabled()

        self.request_queue = request_queue
        self.response_queue = response_queue
        self.cancel_queue = cancel_queue

        # Cap M on the scheduler's pending (CE/prefill) queue depth. When set,
        # the scheduler stops pulling from the request queue once it already
        # holds this many not-yet-running requests, so excess backlog stays in
        # the bounded request queue and exerts backpressure (the API rejects
        # with HTTP 429) instead of growing this unbounded pending pool. This
        # naturally accounts for long requests holding batch/KV space: when
        # they can't drain into a batch, the pending queue stays full and new
        # admissions are shed sooner.
        self.max_pending_requests = max_pending_requests

        # Parse batch scheduling strategy from environment variable
        batch_strategy = BatchSchedulingStrategy.PER_REPLICA
        env_strategy = os.getenv("MAX_SERVE_BATCH_PRIORITY")
        if env_strategy:
            try:
                batch_strategy = BatchSchedulingStrategy(env_strategy.lower())
            except ValueError:
                logger.warning(
                    f"Invalid MAX_SERVE_BATCH_PRIORITY value '{env_strategy}'. "
                    f"Valid values are: {', '.join([s.value for s in BatchSchedulingStrategy])}. "
                    f"Using default: {BatchSchedulingStrategy.PER_REPLICA.value}"
                )

        self.batch_constructor = TextBatchConstructor(
            scheduler_config=scheduler_config,
            pipeline=pipeline,
            kv_cache=kv_cache,
            batch_scheduling_strategy=batch_strategy,
            dp_padder=dp_padder,
        )
        self.scheduler_logger = SchedulerLogger()
        self.support_empty_batches = support_empty_batches
        self.max_items_per_drain = (
            scheduler_config.max_batch_size
            * scheduler_config.data_parallel_degree
            * 2
        )
        self._prefill_spans: dict[RequestID, otel_trace.Span] = {}
        self._decode_spans: dict[RequestID, otel_trace.Span] = {}
        self._batch_counter: int = 0
        # Also gates the libkineto max.batch range. The model worker
        # configures kernel tracing before it builds the scheduler.
        self._batch_spans_enabled = batch_spans_enabled()
        # None unless requests may arm kernel captures, which needs tracing.
        self._kernel_capture = kernel_capture
        self._batch_links = _BatchLinks(
            self._batch_spans_enabled, _MAX_BATCH_LINKS
        )

    @traced
    def _retrieve_pending_requests(self) -> None:
        """
        Retrieves pending requests from the request queue.

        This method drains the request queue synchronously and processes
        any contexts that are available.

        This method is responsible for ensuring that new requests are continuously
        fetched and made available for batching and scheduling.
        """
        max_items = self.max_items_per_drain
        if self.max_pending_requests is not None:
            # Cap M: only pull enough to keep the pending (CE/prefill) queue at
            # or below max_pending_requests. Anything beyond that stays in the
            # request queue, backing it up so the API can shed load.
            available = self.max_pending_requests - len(
                self.batch_constructor.all_ce_reqs
            )
            if available <= 0:
                return
            max_items = min(max_items, available)

        with Tracer("drain_queue"):
            items = drain_queue(
                self.request_queue,
                max_items=max_items,
            )

        with Tracer(f"adding_to_batch_constructor: {len(items)} items"):
            tracing_enabled = _tracing_enabled()
            for context in items:
                self.batch_constructor.enqueue_new_request(context)
                if tracing_enabled:
                    parent = _parent_trace_context(context)
                    self._prefill_spans[context.request_id] = (
                        _tracer.start_span(
                            "max.phase.prefill",
                            context=parent,
                            attributes={
                                "max.request_id": str(context.request_id)
                            },
                        )
                    )
                    traced = (
                        self._kernel_capture is not None
                        and self._kernel_capture.admit(context)
                    )
                    self._batch_links.admit(context.request_id, parent, traced)

    def _end_request_telemetry(self, request_id: RequestID) -> None:
        """Ends a departing request's phase spans and drops its batch link
        and kernel trace."""
        if (span := self._prefill_spans.pop(request_id, None)) is not None:
            span.end()
        if (span := self._decode_spans.pop(request_id, None)) is not None:
            span.end()
        self._batch_links.drop(request_id)
        if self._kernel_capture is not None:
            self._kernel_capture.untrace(request_id)

    @traced
    def run_iteration(self) -> SchedulerProgress:
        """The Scheduler routine that creates batches and schedules them on GPU

        Returns:
            SchedulerProgress: Indicates whether work was performed in this iteration.
        """
        # Drain the request queue and add to CE requests
        # We are starting the time here to include the time it takes to drain the request queue, in batch creation time.
        t0 = time.monotonic()
        self._retrieve_pending_requests()

        # Construct the batch to execute
        inputs = self.batch_constructor.construct_batch()
        t1 = time.monotonic()
        batch_creation_time_s = t1 - t0

        # Failed requests were never admitted and are already released.
        for failed_id, error in self.batch_constructor.take_grammar_failed():
            self.response_queue.put_nowait(
                {failed_id: SchedulerResult.failed(error)}
            )
            if self._tracing:
                self._end_request_telemetry(failed_id)

        # Skip if there is no work to do.
        has_pending_outputs = (
            isinstance(self.pipeline, OverlapTextGenerationPipeline)
            and self.pipeline.has_pending_outputs()
        )
        if not (inputs or self.support_empty_batches or has_pending_outputs):
            return SchedulerProgress.NO_PROGRESS

        # When the overlap pipeline is actually overlapping, the wall-clock
        # time measured below reflects the previous batch's sync, not the
        # current batch. Flag it so the logger attributes execution time and
        # throughput to the completed-batch stats collected below.
        is_overlap_active = bool(
            getattr(self.pipeline, "overlap_active", False)
        )

        # Schedule the batch
        t0 = time.monotonic()
        if len(inputs.flat_batch) > 0:
            with Tracer(f"_schedule({inputs})"):
                num_terminated_reqs = self._schedule(inputs)
        else:
            num_terminated_reqs = self._schedule(inputs)
        t1 = time.monotonic()
        batch_execution_time_s = t1 - t0

        # Log batch metrics. Under overlap scheduling the wall-clock time
        # measured above describes the previously enqueued batch; the
        # pipeline reports that batch's composition and timing so telemetry
        # is attributed to the correct batch type.
        completed_batch_stats = (
            self.pipeline.take_completed_batch_stats()
            if hasattr(self.pipeline, "take_completed_batch_stats")
            else None
        )
        self.scheduler_logger.log_metrics(
            sch_config=self.scheduler_config,
            inputs=inputs,
            kv_cache=self.batch_constructor.kv_cache,
            batch_creation_time_s=batch_creation_time_s,
            batch_execution_time_s=batch_execution_time_s,
            num_pending_reqs=len(self.batch_constructor.all_ce_reqs),
            num_terminated_reqs=num_terminated_reqs,
            total_preemption_count=self.batch_constructor.total_preemption_count,
            batch_spec_decode_metrics=self.pipeline.batch_spec_decode_metrics()
            if hasattr(self.pipeline, "batch_spec_decode_metrics")
            else None,
            batch_vision_metrics=self.pipeline.batch_vision_metrics()
            if hasattr(self.pipeline, "batch_vision_metrics")
            else None,
            batch_video_metrics=self.pipeline.batch_video_metrics()
            if hasattr(self.pipeline, "batch_video_metrics")
            else None,
            overlap_active=is_overlap_active,
            completed_batch_stats=completed_batch_stats,
        )

        for cancelled_id in get_cancelled_reqs(self.cancel_queue):
            if self.batch_constructor.contains(cancelled_id):
                self.batch_constructor.release_request(cancelled_id)
                self.response_queue.put_nowait(
                    {cancelled_id: SchedulerResult.cancelled()}
                )
            if self._tracing:
                self._end_request_telemetry(cancelled_id)

        return SchedulerProgress.MADE_PROGRESS

    def _schedule(self, inputs: TextGenerationInputs[TextContext]) -> int:
        """Returns the number of terminated requests."""
        tracing_enabled = _tracing_enabled()
        batch_id = self._batch_counter
        self._batch_counter += 1

        # This pass's CE (prefill) requests: those not already decoding, less
        # DP padding. They feed the batch span's counts and the CE→TG
        # prefill-span transition below. all_ce_reqs no longer holds them, as
        # construct_batch() popped them. Skipped when tracing is disabled:
        # building the set costs CPU every pass.
        ce_ids_before = (
            {
                ctx.request_id
                for batch, replica in zip(
                    inputs.batches, self.batch_constructor.replicas, strict=True
                )
                for ctx in batch
                if not ctx._is_padding_ctx
                and ctx.request_id not in replica.tg_reqs
            }
            if tracing_enabled
            else None
        )
        # With tracing off, this test and its twin after execute are all a
        # pass spends on the capture, so its per-pass work stays behind them.
        capture = self._kernel_capture
        traced_level = None
        kernel_range = self._batch_spans_enabled
        if capture is not None:
            traced_level = capture.begin_pass(inputs.batches)
            # While a capture may be recording, every pass opens a range, so
            # the replay can tell an untraced pass's GPU work from a traced
            # one's.
            kernel_range = kernel_range or capture.may_record
            host_start_ns = time.time_ns()
        # INVALID_SPAN is OTel's no-op singleton. Real spans are opt-in: a
        # root span per forward pass escapes parent-based sampling, which is
        # too much export volume to be always-on.
        batch_span: otel_trace.Span = otel_trace.INVALID_SPAN
        if tracing_enabled and (
            traced_level is not None or self._batch_spans_enabled
        ):
            assert ce_ids_before is not None
            batch_span = _tracer.start_span(
                "max.batch",
                links=self._batch_links.for_pass(inputs.flat_batch, capture),
                attributes={
                    "max.batch_id": batch_id,
                    "max.ce_count": len(ce_ids_before),
                    "max.tg_count": inputs.batch_size - len(ce_ids_before),
                },
            )

        try:
            # Execute the batch. C++ telemetry that opts in reads the batch id
            # set here.
            _request_context.set_batch_id(batch_id)
            # Only the JSON log formatters read ``_batch_id_ctx``, and only
            # with tracing on.
            batch_id_token = (
                _batch_id_ctx.set(batch_id) if self._tracing else None
            )
            range_id = (
                range_begin_with_id(batch_id, "max.batch")
                if kernel_range
                else 0
            )
            try:
                responses = self.pipeline.execute(inputs)
            finally:
                if range_id:
                    range_end(range_id)
                if batch_id_token is not None:
                    _batch_id_ctx.reset(batch_id_token)
                _request_context.clear_batch_id()
            if capture is not None:
                # Ahead of the releases below, since releasing the last traced
                # request stops the capture and end_pass then drops this pass.
                capped = capture.end_pass(
                    batch_id, traced_level, batch_span, host_start_ns
                )
                self._batch_links.drop_untraced(capped)

            # Filter out all responses for requests that are already released.
            # We can get a response for a request that is already released due to
            # the quirk of overlap scheduling where the pipeline may produce an extra
            # token after EOS.
            responses = {
                req_id: response
                for req_id, response in responses.items()
                if self.batch_constructor.contains(req_id)
            }

            # Advance the requests: moves completed CE requests into TG.
            self.batch_constructor.advance_requests(inputs)

            # Any request that was CE before and is now TG just completed
            # prefill. Skipped when tracing is disabled: see ce_ids_before.
            if ce_ids_before:
                tg_reqs_after = self.batch_constructor.all_tg_reqs
                for req_id in ce_ids_before & tg_reqs_after.keys():
                    if req_id in self._prefill_spans:
                        span = self._prefill_spans.pop(req_id)
                        span.set_attribute("max.batch_id", batch_id)
                        span.end()
                    if req_id in self._decode_spans:
                        # Preempted back to CE and now decoding again: the
                        # prior decode span was never closed on preemption,
                        # so close it here instead of leaking it via
                        # overwrite below.
                        self._decode_spans.pop(req_id).end()
                    self._decode_spans[req_id] = _tracer.start_span(
                        "max.phase.decode",
                        context=_parent_trace_context(tg_reqs_after[req_id]),
                        attributes={
                            "max.request_id": str(req_id),
                            "max.batch_id": batch_id,
                        },
                    )

            # Release terminated requests from the batch
            num_terminated_requests = 0
            for request_id, response in responses.items():
                if response.is_done:
                    self.batch_constructor.release_request(request_id)
                    num_terminated_requests += 1
                    if self._tracing:
                        self._end_request_telemetry(request_id)

            # send the responses to the API process
            if responses:
                self.response_queue.put_nowait(
                    {
                        req_id: SchedulerResult.create(response, batch_id)
                        for req_id, response in responses.items()
                    }
                )

            batch_span.set_attribute(
                "max.terminated_count", num_terminated_requests
            )
            return num_terminated_requests
        finally:
            batch_span.end()


def load_text_generation_scheduler(
    pipeline: TextGenerationPipeline[TextContext],
    pipeline_config: PipelineConfig,
    request_queue: MAXPullQueue[TextContext | TextAndVisionContext],
    response_queue: MAXPushQueue[
        dict[RequestID, SchedulerResult[TextGenerationOutput]]
    ],
    cancel_queue: MAXPullQueue[list[RequestID]],
    memory_plan: MemoryPlan | None,
    max_pending_requests: int | None = None,
    kernel_capture: KernelCapture | None = None,
) -> TokenGenerationScheduler:
    # Create Scheduler Config.
    scheduler_config = TokenGenerationSchedulerConfig.from_pipeline_config(
        pipeline_config, pipeline.max_batch_size, memory_plan
    )

    # Build DP batch padder when DP > 1 with device graph capture.
    kv_manager = pipeline.kv_manager
    dp_padder: DPBatchPadder | None = None
    assert pipeline_config.model is not None
    if (
        scheduler_config.data_parallel_degree > 1
        and pipeline_config.runtime.device_graph_capture
    ):
        # Padding dummies must match the architecture's concrete context
        # type — for VLMs the overlap pipeline narrows every batched context
        # to TextAndVisionContext, so plain-TextContext dummies would crash
        # the first padded batch.
        context_type = PIPELINE_REGISTRY.retrieve_context_type(pipeline_config)
        assert issubclass(context_type, TextContext)
        assert (
            memory_plan is not None
            and memory_plan.planned_max_length is not None
        ), "DP batch padding requires a memory plan with a max length"
        dp_padder = DPBatchPadder(
            dp_size=scheduler_config.data_parallel_degree,
            kv_manager=kv_manager,
            max_length=memory_plan.planned_max_length,
            model_name=pipeline_config.model.model_name,
            pipeline=pipeline,
            context_type=context_type,
        )

    # Return Scheduler
    return TokenGenerationScheduler(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        # For spec decoding, there may be multiple KVCaches. The scheduler
        # arbitrarily uses either the draft or target one. The other kvcache is
        # hidden from scheduler currently and managed by pipelines.
        kv_cache=kv_manager,
        request_queue=request_queue,
        response_queue=response_queue,
        cancel_queue=cancel_queue,
        support_empty_batches=pipeline_config.runtime.execute_empty_batches,
        dp_padder=dp_padder,
        max_pending_requests=max_pending_requests,
        kernel_capture=kernel_capture,
    )
