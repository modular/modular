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

from max.interfaces import (
    MAXPullQueue,
    MAXPushQueue,
    RequestID,
    Scheduler,
    SchedulerResult,
    TextGenerationInputs,
    TextGenerationOutput,
)
from max.interfaces.queue import BackgroundQueueDrainer, drain_queue
from max.nn.kv_cache import PagedKVCacheManager
from max.pipelines.core import TextAndVisionContext, TextContext
from max.pipelines.lib import PipelineConfig, TextGenerationPipelineType
from max.pipelines.lib.pipeline import get_paged_manager
from max.profiler import Tracer, traced

from .base import SchedulerProgress
from .data_parallelism_utils import split_by_replica_idx
from .text_batch_constructor import (
    TextBatchConstructor,
    TokenGenerationSchedulerConfig,
)
from .utils import (
    SchedulerLogger,
    add_newly_encoded_reqs_to_tg_batch,
    release_cancelled_requests,
    release_terminated_requests,
)

logger = logging.getLogger("max.serve")


class TokenGenerationScheduler(Scheduler):
    def __init__(
        self,
        scheduler_config: TokenGenerationSchedulerConfig,
        pipeline: TextGenerationPipelineType[TextContext],
        *,
        request_queue: MAXPullQueue[TextContext | TextAndVisionContext],
        response_queue: MAXPushQueue[
            dict[RequestID, SchedulerResult[TextGenerationOutput]]
        ],
        cancel_queue: MAXPullQueue[list[RequestID]],
        paged_manager: PagedKVCacheManager | None = None,
        offload_queue_draining: bool = False,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.pipeline = pipeline

        self.request_queue = request_queue
        self.response_queue = response_queue
        self.cancel_queue = cancel_queue

        self.batch_constructor = TextBatchConstructor(
            scheduler_config=scheduler_config,
            pipeline=pipeline,
            paged_cache=paged_manager,
        )
        self.scheduler_logger = SchedulerLogger()

        # We are parameterizing the offload of queue draining to allow for
        # the use case where we want to drain the queue in the main thread.
        # This is useful for debugging and testing purposes.
        self._queue_drainer: (
            BackgroundQueueDrainer[TextContext | TextAndVisionContext] | None
        ) = None
        if offload_queue_draining:
            # I am setting this to drain at max batch size ce * 2, to ensure we do not drain
            # forever, but have more than enough to form full batches.
            self._queue_drainer = BackgroundQueueDrainer[
                TextContext | TextAndVisionContext
            ](
                self.request_queue,
                max_items_per_drain=self.scheduler_config.max_batch_size_ce * 2,
            )

    @traced
    def _retrieve_pending_requests(self) -> None:
        """
        Initiates retrieval of pending requests from the request queue.

        If a background retrieval task is already running, this method returns immediately.
        Otherwise, it submits a background task to drain the request queue and processes
        any contexts that have already been retrieved and are pending.

        This method is responsible for ensuring that new requests are continuously
        fetched and made available for batching and scheduling.
        """
        if self._queue_drainer is not None:
            self._queue_drainer.start_draining()
            items = self._queue_drainer.retrieve_items()
        else:
            items = drain_queue(
                self.request_queue,
                max_items=self.scheduler_config.max_batch_size_ce * 2,
            )

        for context in items:
            self.batch_constructor.ce_reqs[context.request_id] = context

    @traced
    def run_iteration(self) -> SchedulerProgress:
        """The Scheduler routine that creates batches and schedules them on GPU

        Returns:
            SchedulerProgress: Indicates whether work was performed in this iteration.
        """
        # Drain the request queue and add to CE requests
        self._retrieve_pending_requests()

        # Construct the batch to execute
        t0 = time.monotonic()
        inputs = self.batch_constructor.construct_batch()
        t1 = time.monotonic()
        batch_creation_time_s = t1 - t0

        # If the batch is empty, skip
        if not inputs:
            return SchedulerProgress.NO_PROGRESS

        # Schedule the batch
        t0 = time.monotonic()
        with Tracer(f"_schedule({inputs})"):
            num_terminated_reqs = self._schedule(inputs)
        t1 = time.monotonic()
        batch_execution_time_s = t1 - t0

        # Log batch metrics
        self.scheduler_logger.log_metrics(
            sch_config=self.scheduler_config,
            inputs=inputs,
            paged_cache=self.batch_constructor.paged_cache,
            batch_creation_time_s=batch_creation_time_s,
            batch_execution_time_s=batch_execution_time_s,
            num_pending_reqs=len(self.batch_constructor.ce_reqs),
            num_terminated_reqs=num_terminated_reqs,
            total_preemption_count=self.batch_constructor.total_preemption_count,
        )

        release_cancelled_requests(
            self.cancel_queue,
            self.response_queue,
            self.batch_constructor.tg_reqs,
            self.pipeline,
        )

        return SchedulerProgress.MADE_PROGRESS

    def _schedule(self, inputs: TextGenerationInputs[TextContext]) -> int:
        """Returns the number of terminated requests."""
        assert inputs

        # TODO(E2EOPT-399): Add proper data parallelism support. Currently
        # this naively splits the batch onto different devices.
        split_by_replica_idx(
            inputs,
            self.scheduler_config.data_parallel_degree,
            self.batch_constructor.paged_cache,
        )

        # execute the batch
        responses = self.pipeline.execute(inputs)

        # If there is a chunked request, we put it back into the request queue
        add_newly_encoded_reqs_to_tg_batch(
            inputs.batch,
            responses,
            self.batch_constructor,
        )

        # remove terminated requests from the batch
        num_terminated_reqs = release_terminated_requests(
            responses,
            self.pipeline,
            self.batch_constructor.tg_reqs,
        )

        # send the responses to the API process
        if responses:
            self.response_queue.put_nowait(
                {
                    req_id: SchedulerResult.create(response)
                    for req_id, response in responses.items()
                }
            )

        return num_terminated_reqs


def load_text_generation_scheduler(
    pipeline: TextGenerationPipelineType[TextContext],
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

    # Retrieve Paged Manager
    paged_manager = get_paged_manager(pipeline)

    # Return Scheduler
    return TokenGenerationScheduler(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        paged_manager=paged_manager,
        request_queue=request_queue,
        response_queue=response_queue,
        cancel_queue=cancel_queue,
        offload_queue_draining=pipeline_config.experimental_background_queue,
    )
