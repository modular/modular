# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
import time
import traceback

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
from max.pipelines.lib import PipelineConfig, get_kv_cache
from max.profiler import Tracer, traced

from .base import SchedulerProgress
from .batch_constructor import TextBatchConstructor
from .config import TokenGenerationSchedulerConfig
from .utils import SchedulerLogger, get_cancelled_reqs

logger = logging.getLogger("max.serve")


class OverlapTokenGenerationScheduler(Scheduler):
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
        kv_cache: PagedKVCacheManager | None = None,
        support_empty_batches: bool = False,
    ) -> None:
        logger.warning(
            "Using highly experimental overlap scheduler. It is expected to be buggy and unstable."
        )

        self.scheduler_config = scheduler_config
        self.pipeline = pipeline

        self.request_queue = request_queue
        self.response_queue = response_queue
        self.cancel_queue = cancel_queue

        self.batch_constructor = TextBatchConstructor(
            scheduler_config=scheduler_config,
            pipeline=pipeline,
            kv_cache=kv_cache,
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

        # If the batch is empty, skip
        if not inputs and not self.support_empty_batches:
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
        batch_request_ids = [
            context.request_id for context in inputs.flat_batch
        ]

        # Execute the batch.
        try:
            responses = self.pipeline.execute(inputs)
        except Exception as exc:
            logger.error(
                "Exception during pipeline execution: %s",
                traceback.format_exc(),
            )

            # Send error results to ALL requests in the batch
            self.response_queue.put_nowait(
                {
                    req_id: SchedulerResult.from_error(exc)
                    for req_id in batch_request_ids
                }
            )

            # Release all requests from batch constructor
            for req_id in batch_request_ids:
                if self.batch_constructor.contains(req_id):
                    self.batch_constructor.release_request(req_id)

            return len(batch_request_ids)

        # Advance the requests and collect the invalid request IDs
        for (
            request_id
        ) in self.batch_constructor.advance_requests_and_collect_invalid_ids(
            inputs.batches
        ):
            # The only scenario where the request ID should not be in the responses dictionary, is if the pipeline
            # errored out, this should not happen.
            del responses[request_id]

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


def load_overlap_text_generation_scheduler(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
    pipeline_config: PipelineConfig,
    request_queue: MAXPullQueue[TextContext | TextAndVisionContext],
    response_queue: MAXPushQueue[
        dict[RequestID, SchedulerResult[TextGenerationOutput]]
    ],
    cancel_queue: MAXPullQueue[list[RequestID]],
) -> OverlapTokenGenerationScheduler:
    # Create Scheduler Config.
    scheduler_config = TokenGenerationSchedulerConfig.from_pipeline_config(
        pipeline_config
    )

    # Retrieve Paged Manager
    kv_cache = get_kv_cache(pipeline)

    # Return Scheduler
    return OverlapTokenGenerationScheduler(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_cache,
        request_queue=request_queue,
        response_queue=response_queue,
        cancel_queue=cancel_queue,
        support_empty_batches=pipeline_config.execute_empty_batches,
    )
