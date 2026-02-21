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
"""OneShotScheduler for non-autoregressive pipelines.

This scheduler is designed for pipelines that process requests in a single pass
without requiring iterative generation (e.g., image generation, non-autoregressive
text models). It processes each request serially, making it simple and suitable
for workloads that don't benefit from batching or continuous generation.
"""

import logging
import queue
from collections.abc import Callable
from typing import Generic

from max.interfaces import (
    BaseContextType,
    MAXPullQueue,
    MAXPushQueue,
    Pipeline,
    PipelineInputsType,
    PipelineOutputType,
    RequestID,
    Scheduler,
    SchedulerResult,
)
from max.profiler import traced

from .base import SchedulerProgress

logger = logging.getLogger("max.serve")


class OneShotScheduler(
    Scheduler,
    Generic[BaseContextType, PipelineInputsType, PipelineOutputType],
):
    """Scheduler for non-autoregressive pipelines that processes requests serially.

    This scheduler is optimized for pipelines that:
    - Complete in a single forward pass (no iterative generation)
    - Don't require KV cache management
    - Process each request independently

    The scheduler pulls one request at a time from the queue, executes the pipeline,
    and returns the result. This simple approach is suitable for image generation,
    embeddings with small batch sizes, and other non-autoregressive workloads.

    Args:
        pipeline: The pipeline to execute requests with
        batch_constructor: Callable that converts a request context into pipeline inputs.
            Takes a BaseContextType and returns a PipelineInputsType.
        request_queue: Queue to pull incoming requests from
        response_queue: Queue to push completed responses to
        cancel_queue: Queue for handling request cancellations
        max_batch_size: Maximum number of requests to process in a single batch.
            Defaults to 1 for serial processing.
    """

    def __init__(
        self,
        pipeline: Pipeline[PipelineInputsType, PipelineOutputType],
        batch_constructor: Callable[[BaseContextType], PipelineInputsType],
        request_queue: MAXPullQueue[BaseContextType],
        response_queue: MAXPushQueue[
            dict[RequestID, SchedulerResult[PipelineOutputType]]
        ],
        cancel_queue: MAXPullQueue[list[RequestID]],
        max_batch_size: int = 1,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.pipeline = pipeline
        self.batch_constructor = batch_constructor
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.cancel_queue = cancel_queue

    @traced
    def _get_next_request(self) -> BaseContextType | None:
        """Pull the next request from the queue.

        Returns:
            The next context to process, or None if the queue is empty.
        """
        try:
            return self.request_queue.get_nowait()
        except queue.Empty:
            return None

    def run_iteration(self) -> SchedulerProgress:
        """Execute one scheduling iteration.

        Pulls a single request from the queue, executes it through the pipeline,
        and sends the response back.

        Returns:
            SchedulerProgress.MADE_PROGRESS if a request was processed,
            SchedulerProgress.NO_PROGRESS if no requests were available.
        """
        # Get the next request
        context = self._get_next_request()
        if context is None:
            return SchedulerProgress.NO_PROGRESS

        logger.info(f"OneShotScheduler: Starting request {context.request_id}")

        try:
            # Convert the context to pipeline inputs using the batch constructor
            pipeline_inputs = self.batch_constructor(context)

            # Execute the pipeline
            responses = self.pipeline.execute(pipeline_inputs)

            logger.info(
                f"OneShotScheduler: Completed request {context.request_id} "
                f"with {len(responses)} response(s)"
            )

            # Send the responses
            self.response_queue.put_nowait(
                {
                    request_id: SchedulerResult.create(response)
                    for request_id, response in responses.items()
                }
            )
        except Exception as exc:
            logger.exception(
                f"OneShotScheduler: Exception during pipeline execution for request {context.request_id}"
            )

            # Send cancelled result (error details are logged above)
            self.response_queue.put_nowait(
                {context.request_id: SchedulerResult.cancelled()}
            )

        return SchedulerProgress.MADE_PROGRESS
