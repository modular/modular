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

import logging
import queue
from dataclasses import dataclass

from max.interfaces import (
    EmbeddingsContext,
    EmbeddingsGenerationInputs,
    EmbeddingsGenerationOutput,
    MAXPullQueue,
    MAXPushQueue,
    RequestID,
    Scheduler,
    SchedulerResult,
)
from max.interfaces.queue import BackgroundQueueDrainer
from max.pipelines.lib import EmbeddingsPipelineType
from max.profiler import traced

from .base import SchedulerProgress

logger = logging.getLogger("max.serve")


@dataclass
class EmbeddingsSchedulerConfig:
    """Embeddings Scheduler configuration."""

    # The maximum number of requests that can be in the encode batch.
    max_batch_size: int


class EmbeddingsScheduler(Scheduler):
    def __init__(
        self,
        scheduler_config: EmbeddingsSchedulerConfig,
        pipeline: EmbeddingsPipelineType,
        request_queue: MAXPullQueue[EmbeddingsContext],
        response_queue: MAXPushQueue[
            dict[RequestID, SchedulerResult[EmbeddingsGenerationOutput]]
        ],
        cancel_queue: MAXPullQueue[list[RequestID]],
        offload_queue_draining: bool = False,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.pipeline = pipeline
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.cancel_queue = cancel_queue

        # We are parameterizing the offload of queue draining to allow for
        # the use case where we want to drain the queue in the main thread.
        # This is useful for debugging and testing purposes.
        self._queue_drainer: (
            BackgroundQueueDrainer[EmbeddingsContext] | None
        ) = None
        if offload_queue_draining:
            # Initialize the background queue drainer
            self._queue_drainer = BackgroundQueueDrainer[EmbeddingsContext](
                self.request_queue,
                max_items_per_drain=self.scheduler_config.max_batch_size * 2,
            )

    @traced
    def _create_batch_to_execute(
        self,
    ) -> dict[RequestID, EmbeddingsContext]:
        max_batch_size_to_create = self.scheduler_config.max_batch_size

        batch: dict[RequestID, EmbeddingsContext] = {}

        if self._queue_drainer is not None:
            # Start draining the queue in the background
            self._queue_drainer.start_draining()

            # Process items from the drainer
            while True:
                if len(batch) < max_batch_size_to_create:
                    try:
                        item = self._queue_drainer.retrieve_item()
                        batch[item.request_id] = item
                    except queue.Empty:
                        break
                else:
                    break
        else:
            # Synchronous draining
            while True:
                if len(batch) >= max_batch_size_to_create:
                    break

                try:
                    item = self.request_queue.get_nowait()
                except queue.Empty:
                    break

                batch[item.request_id] = item

        return batch

    def run_iteration(self) -> SchedulerProgress:
        """The Scheduler loop that creates batches and schedules them on GPU

        Returns:
            SchedulerProgress: Indicates whether work was performed in this iteration.
        """
        batch_to_execute = self._create_batch_to_execute()
        if len(batch_to_execute) == 0:
            return SchedulerProgress.NO_PROGRESS

        self._schedule_encode(batch_to_execute)
        return SchedulerProgress.MADE_PROGRESS

    @traced
    def _handle_terminated_responses(
        self,
        batch_executed: dict[RequestID, EmbeddingsContext],
        batch_response: dict[RequestID, EmbeddingsGenerationOutput],
    ) -> None:
        """Task that handles responses"""
        already_terminated = set()
        terminated = batch_executed.keys() - batch_response.keys()
        for req_id in terminated:
            if req_id in already_terminated:
                continue
            del batch_executed[req_id]
            already_terminated.add(req_id)

    @traced
    def _schedule_encode(
        self, batch_to_execute: dict[RequestID, EmbeddingsContext]
    ) -> None:
        # execute the batch
        batch_responses = self.pipeline.execute(
            EmbeddingsGenerationInputs(batches=[batch_to_execute])
        )
        # remove terminated requests from the batch
        self._handle_terminated_responses(batch_to_execute, batch_responses)
        # send the responses to the API process
        self.response_queue.put_nowait(
            {
                request_id: SchedulerResult.create(response)
                for request_id, response in batch_responses.items()
            }
        )
