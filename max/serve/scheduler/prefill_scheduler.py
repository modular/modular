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
from typing import Optional

import zmq
from max.nn.kv_cache import PagedKVCacheManager
from max.pipelines.core import InputContext, TokenGenerator
from max.serve.process_control import ProcessControl

from .base import Scheduler
from .zmq_queue import ZmqPullSocket, ZmqPushSocket

logger = logging.getLogger("max.serve")


@dataclass
class PrefillSchedulerConfig:
    """Prefill Specific Scheduler Config."""

    max_batch_size_ce: int
    """The maximum number of requests that can be in the context encoding batch."""

    batch_timeout: Optional[float]
    """The maximum amount of time to wait before creating a context encoding batch."""

    enable_chunked_prefill: bool = True
    """Enables chunked prefill, where the scheduler splits requests into chunks to ensure
    each batch contains exactly `target_tokens_per_batch_ce` tokens."""

    target_tokens_per_batch_ce: int = 4096
    """The target total number of tokens to encode in the context encoding batch."""


class PrefillScheduler(Scheduler):
    def __init__(
        self,
        process_control: ProcessControl,
        pipeline: TokenGenerator,
        scheduler_config: PrefillSchedulerConfig,
        paged_manager: PagedKVCacheManager,
        *,
        zmq_ctx: zmq.Context,
        prefill_zmq_endpoint: str,
        decode_zmq_endpoint: str,
    ):
        self.pc = process_control
        self.pipeline = pipeline
        self.scheduler_config = scheduler_config
        self.paged_manager = paged_manager

        # Initialize Queues for Disaggregation
        logger.info(f"starting prefill queue on: {prefill_zmq_endpoint}")
        logger.info(f"starting decode queue on: {decode_zmq_endpoint}")
        self.prefill_pull_socket = ZmqPullSocket[tuple[str, InputContext]](
            zmq_ctx, prefill_zmq_endpoint
        )
        self.decode_push_socket = ZmqPushSocket[tuple[str, InputContext]](
            zmq_ctx, decode_zmq_endpoint
        )

        # Initialize Scheduler state.
        self.active_batch: dict[str, InputContext] = {}
        self.available_cache_indices = set(
            range(self.scheduler_config.max_batch_size_ce)
        )
        self.preempted_requests: queue.Queue[tuple[str, InputContext]] = (
            queue.Queue()
        )

    def push_to_decode_socket(
        self, request_id: str, data: InputContext
    ) -> None:
        self.decode_push_socket.put_nowait((request_id, data))

    def pull_from_prefill_socket(self) -> tuple[str, InputContext]:
        """Pulls a request from the prefill socket, checking preempted requests first.

        Returns:
            tuple[str, InputContext]: A tuple containing the request ID and input context.

        Raises:
            queue.Empty: If no requests are available.
            zmq.ZMQError: If there is an error receiving from the socket.
        """
        # First try and return from pre-empted requests queue.
        if not self.preempted_requests.empty():
            return self.preempted_requests.get()

        return self.prefill_pull_socket.get_nowait()

    def return_to_prefill_queue(self, req_id: str, data: InputContext) -> None:
        """Releases the cache index back to the available pool and cleans up pipeline
        resources before returning the request to the preempted queue.

        Args:
            req_id: The ID of the request to return
            data: The InputContext containing the request data
        """
        self.available_cache_indices.add(data.cache_seq_id)
        self.pipeline.release(data)
        data.reset()
        self.preempted_requests.put((req_id, data))

    def update_batch(self) -> None:
        """Updates the active batch by pulling requests from the prefill queue.

        Processes requests up to max_batch_size_ce, handling cache assignment and chunking.
        For each request:
        - Assigns cache if needed
        - Attempts to schedule via paged manager
        - Chunks inputs if enabled and batch token length exceeds target
        - Tracks total batch token length
        """
        batch_token_length = 0
        for _ in range(self.scheduler_config.max_batch_size_ce):
            try:
                req_id, data = self.pull_from_prefill_socket()

                if data.start_idx == 0:
                    data.unassign_from_cache()

                if not data.is_assigned_to_cache:
                    data.assign_to_cache(self.available_cache_indices.pop())
                    self.paged_manager.external_claim([data.cache_seq_id])

            except queue.Empty:
                break

            scheduled = self.paged_manager.prefetch(data, 1)

            if not scheduled:
                self.return_to_prefill_queue(req_id, data)
                break

            if self.scheduler_config.enable_chunked_prefill:
                if (
                    batch_token_length + data.active_length
                    >= self.scheduler_config.target_tokens_per_batch_ce
                ):
                    trimmed_tokens = (
                        batch_token_length
                        + data.active_length
                        - self.scheduler_config.target_tokens_per_batch_ce
                    )
                    data.bump_token_indices(active_idx=-trimmed_tokens)

            batch_token_length += data.active_length
            self.active_batch[req_id] = data

    def _handle_chunked_requests(
        self,
    ) -> None:
        """Handles chunked requests by either sending them back to the preempted queue or to decode.

        For the last request in the active batch:
        - If it was chunked (active_idx - start_idx > 1), sends it back to preempted queue
        - If not chunked, resets indices and sends to decode socket
        """

        # Always pop the last item.
        # If its chunked, we should response the associated item from the responses dict.
        # If not, we simple add it back into the dictionary.
        # Both popitem, and putting the same value in a dictionary are O(1)
        # Which should be faster than creating a list to retrieve the last dictionary item
        # and then conditionally popping which is O(n).
        last_request_id, last_request = self.active_batch.popitem()

        # Check if its chunked.
        if last_request.active_idx - last_request.start_idx > 1:
            # If its chunked, add it back to the start of the request queue.
            self.preempted_requests.put((last_request_id, last_request))
        else:
            # Send to decode if not chunked
            last_request.bump_token_indices(start_idx=-last_request.start_idx)
            self.push_to_decode_socket(last_request_id, last_request)

    def schedule(self) -> None:
        """Executes the current batch of requests and sends completed requests to decode.

        Processes the active batch through the pipeline, handles any chunked prefill requests,
        and sends completed requests to the decode queue while resetting their token indices.
        """
        # Execute the Batch
        _ = self.pipeline.next_token(
            self.active_batch,
            num_steps=1,
        )

        if self.scheduler_config.enable_chunked_prefill:
            self._handle_chunked_requests()

        # Send completed requests to decode queue.
        while self.active_batch:
            req_id, input_context = self.active_batch.popitem()
            # Reset this - This is a workaround until we successfully transfer the KV Cache.
            input_context.bump_token_indices(start_idx=-input_context.start_idx)
            self.push_to_decode_socket(req_id, input_context)

    def run(self) -> None:
        """Main scheduling loop that processes prefill requests.

        Continuously receives requests, creates batches, and schedules them for processing
        while handling errors and cancelled requests. The loop continues until the process
        is cancelled.
        """
        i = 0
        while not self.pc.is_canceled():
            # Indicate that the process is still alive.
            self.pc.beat()
            i += 1

            # Try and receive any request from the prefill node.
            try:
                # Create a new batch
                self.update_batch()

                # Break out of loop if batch is empty.
                if not self.active_batch:
                    continue

                self.schedule()

                # Occasionally handle cancelled requests.
                if i % 20 == 0:
                    # TODO: E2EOPT-225 Handle Cancelled Requests
                    pass

            except Exception as e:
                logger.exception("An error occured during scheduling.")
                raise e
