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

from max.interfaces import (
    MAXPullQueue,
    MAXPushQueue,
    Pipeline,
    RequestID,
    Scheduler,
    SchedulerResult,
    TextGenerationInputs,
    TextGenerationOutput,
)
from max.interfaces.queue import drain_queue
from max.kv_cache import PagedKVCacheManager
from max.pipelines.core import TextAndVisionContext, TextContext
from max.pipelines.lib import (
    OverlapTextGenerationPipeline,
    PipelineConfig,
    TextGenerationPipeline,
)
from max.profiler import Tracer, traced

from .base import SchedulerProgress
from .batch_constructor import TextBatchConstructor
from .batch_constructor.text_batch_constructor import BatchSchedulingStrategy
from .config import TokenGenerationSchedulerConfig
from .utils import SchedulerLogger, get_cancelled_reqs

logger = logging.getLogger("max.serve")


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
        kv_cache: PagedKVCacheManager,
        support_empty_batches: bool = False,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.pipeline = pipeline

        self.request_queue = request_queue
        self.response_queue = response_queue
        self.cancel_queue = cancel_queue

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
        )
        self.scheduler_logger = SchedulerLogger()
        self.support_empty_batches = support_empty_batches
        self.max_items_per_drain = (
            scheduler_config.max_batch_size
            * scheduler_config.data_parallel_degree
            * 2
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
        with Tracer("drain_queue"):
            items = drain_queue(
                self.request_queue,
                max_items=self.max_items_per_drain,
            )

        with Tracer(f"adding_to_batch_constructor: {len(items)} items"):
            for context in items:
                self.batch_constructor.enqueue_new_request(context)

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

        # Skip if there is no work to do.
        has_pending_outputs = (
            isinstance(self.pipeline, OverlapTextGenerationPipeline)
            and self.pipeline.has_pending_outputs()
        )
        if not (inputs or self.support_empty_batches or has_pending_outputs):
            return SchedulerProgress.NO_PROGRESS

        # Schedule the batch
        t0 = time.monotonic()
        if len(inputs.flat_batch) > 0:
            with Tracer(f"_schedule({inputs})"):
                num_terminated_reqs = self._schedule(inputs)
        else:
            num_terminated_reqs = self._schedule(inputs)
        t1 = time.monotonic()
        batch_execution_time_s = t1 - t0

        # Log batch metrics
        self.scheduler_logger.log_metrics(
            sch_config=self.scheduler_config,
            inputs=inputs,
            kv_cache=self.batch_constructor.kv_cache,
            batch_creation_time_s=batch_creation_time_s,
            batch_execution_time_s=batch_execution_time_s,
            num_pending_reqs=len(self.batch_constructor.all_ce_reqs),
            num_terminated_reqs=num_terminated_reqs,
            total_preemption_count=self.batch_constructor.total_preemption_count,
        )

        for cancelled_id in get_cancelled_reqs(self.cancel_queue):
            if self.batch_constructor.contains(cancelled_id):
                self.batch_constructor.release_request(cancelled_id)
                self.response_queue.put_nowait(
                    {cancelled_id: SchedulerResult.cancelled()}
                )

        return SchedulerProgress.MADE_PROGRESS

    def _schedule(self, inputs: TextGenerationInputs[TextContext]) -> int:
        """Returns the number of terminated requests."""
        # Execute the batch.
        responses = self.pipeline.execute(inputs)

        # Filter out all responses for requests that are already released.
        # We can get a response for a request that is already released due to
        # the quirk of overlap scheduling where the pipeline may produce an extra
        # token after EOS.
        responses = {
            req_id: response
            for req_id, response in responses.items()
            if self.batch_constructor.contains(req_id)
        }

        # Advance the requests and collect the invalid request IDs
        self.batch_constructor.advance_requests(inputs)

        # Release terminated requests from the batch
        num_terminated_requests = 0
        for request_id, response in responses.items():
            if response.is_done:
                self.batch_constructor.release_request(request_id)
                num_terminated_requests += 1

        # send the responses to the API process
        if responses:
            self.response_queue.put_nowait(
                {
                    req_id: SchedulerResult.create(response)
                    for req_id, response in responses.items()
                }
            )

        return num_terminated_requests


def load_text_generation_scheduler(
    pipeline: TextGenerationPipeline[TextContext],
    pipeline_config: PipelineConfig,
    request_queue: MAXPullQueue[TextContext | TextAndVisionContext],
    response_queue: MAXPushQueue[
        dict[RequestID, SchedulerResult[TextGenerationOutput]]
    ],
    cancel_queue: MAXPullQueue[list[RequestID]],
) -> TokenGenerationScheduler:
    # Create Scheduler Config.
    scheduler_config = TokenGenerationSchedulerConfig.from_pipeline_config(
        pipeline_config
    )

    # Return Scheduler
    return TokenGenerationScheduler(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        # For spec decoding, there may be multiple KVCaches. The scheduler
        # arbitrarily uses either the draft or target one. The other kvcache is
        # hidden from scheduler currently and managed by pipelines.
        kv_cache=pipeline.kv_manager,
        request_queue=request_queue,
        response_queue=response_queue,
        cancel_queue=cancel_queue,
        support_empty_batches=pipeline_config.execute_empty_batches,
    )
