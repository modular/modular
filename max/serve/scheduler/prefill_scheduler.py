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
import tempfile
import uuid
from dataclasses import dataclass
from typing import Optional

import zmq
from max.nn.kv_cache import (
    KVTransferEngine,
    PagedKVCacheManager,
)
from max.pipelines.core import InputContext, TokenGenerator
from max.pipelines.lib.pipeline import get_paged_manager
from max.serve.config import Settings
from max.serve.process_control import ProcessControl
from max.serve.queue.zmq_queue import ZmqPullSocket, ZmqPushSocket

from .base import DecodeRequest, PrefillRequest, Scheduler

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
        transfer_engine_zmq_endpoint: str = f"ipc://{tempfile.gettempdir()}/transfer_engine",
    ):
        self.pc = process_control
        self.pipeline = pipeline
        self.scheduler_config = scheduler_config
        self.paged_manager = paged_manager

        # Initialize Queues for Disaggregation
        logger.info(f"starting prefill queue on: {prefill_zmq_endpoint}")
        logger.info(f"starting decode queue on: {decode_zmq_endpoint}")
        self.prefill_pull_socket = ZmqPullSocket[PrefillRequest](
            zmq_ctx, prefill_zmq_endpoint
        )
        self.decode_push_socket = ZmqPushSocket[tuple[str, InputContext]](
            zmq_ctx, decode_zmq_endpoint
        )

        # Initialize Scheduler state.
        self.pending_transfers: dict[str, tuple[str, list[int]]] = {}
        self.active_batch: dict[str, InputContext] = {}
        self.available_cache_indices = set(
            range(self.scheduler_config.max_batch_size_ce)
        )
        self.preempted_prefill: queue.Queue[PrefillRequest] = queue.Queue()

        # Create Transfer Engine
        self.transfer_engine = KVTransferEngine(
            name=f"prefill_agent_{uuid.uuid4()}",
            listen_port=8047,
            tensor=self.paged_manager.device_tensors[0],
            total_num_pages=self.paged_manager.total_num_pages,
        )

        self.register_remote_transfer_engine(
            transfer_engine_zmq_endpoint, zmq_ctx
        )

    def register_remote_transfer_engine(
        self, transfer_engine_zmq_endpoint: str, zmq_ctx: zmq.Context
    ) -> None:
        """Registers and connects the transfer engine with a remote decode agent.

        This function establishes a ZMQ socket connection with a remote decode agent,
        exchanges transfer engine metadata between the two agents, and sets up the
        connection between them. The metadata exchange allows the agents to communicate
        and transfer data between each other.

        Args:
            zmq_ctx: The ZMQ context used to create the socket connection.
        """
        # Open up the socket.
        logger.debug("connecting to transfer engine socket.")
        socket = zmq_ctx.socket(zmq.REQ)
        socket.connect(transfer_engine_zmq_endpoint)

        # Send Transfer Engine Metadata
        logger.debug("sending prefill transfer engine metadata.")
        socket.send_pyobj(self.transfer_engine.metadata)

        # Wait for Partner Transfer Engine Metadata
        logger.debug("waiting for decode engine metadata.")
        remote_engine_message = socket.recv_pyobj()
        self.transfer_engine.connect(remote_engine_message)

    def push_to_decode_socket(
        self, request_id: str, data: InputContext
    ) -> None:
        self.decode_push_socket.put_nowait(DecodeRequest(request_id, data))

    def pull_from_prefill_socket(
        self,
    ) -> PrefillRequest:
        """Pulls a request from the prefill socket, checking preempted requests first.

        Returns:
            tuple[str, InputContext]: A tuple containing the request ID and input context.

        Raises:
            queue.Empty: If no requests are available.
            zmq.ZMQError: If there is an error receiving from the socket.
        """
        # First try and return from pre-empted requests queue.
        if not self.preempted_prefill.empty():
            return self.preempted_prefill.get()

        return self.prefill_pull_socket.get_nowait()

    def return_to_prefill_queue(
        self,
        prefill_request: PrefillRequest,
    ) -> None:
        """Releases the cache index back to the available pool and cleans up pipeline
        resources before returning the request to the preempted queue.

        Args:
            req_id: The ID of the request to return
            data: The InputContext containing the request data
        """
        self.pipeline.release(prefill_request.context)
        prefill_request.context.reset()
        self.preempted_prefill.put(prefill_request)

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
        while self.available_cache_indices:
            try:
                prefill_request = self.pull_from_prefill_socket()
                logger.info("received from decode node!")

                if prefill_request.context.start_idx == 0:
                    prefill_request.context.unassign_from_cache()

                if not prefill_request.context.is_assigned_to_cache:
                    prefill_request.context.assign_to_cache(
                        self.available_cache_indices.pop()
                    )
                    self.paged_manager.external_claim(
                        [prefill_request.context.cache_seq_id]
                    )

            except queue.Empty:
                break

            scheduled = self.paged_manager.prefetch(prefill_request.context, 1)

            if not scheduled:
                self.return_to_prefill_queue(prefill_request)
                break

            if self.scheduler_config.enable_chunked_prefill:
                if (
                    batch_token_length + prefill_request.context.active_length
                    >= self.scheduler_config.target_tokens_per_batch_ce
                ):
                    trimmed_tokens = (
                        batch_token_length
                        + prefill_request.context.active_length
                        - self.scheduler_config.target_tokens_per_batch_ce
                    )
                    prefill_request.context.bump_token_indices(
                        active_idx=-trimmed_tokens
                    )

            batch_token_length += prefill_request.context.active_length
            self.active_batch[prefill_request.id] = prefill_request.context
            self.pending_transfers[prefill_request.id] = (
                prefill_request.transfer_engine_name,
                prefill_request.block_ids,
            )

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
        remote_name, dst_idx = self.pending_transfers.pop(last_request_id)

        # Check if its chunked.
        if last_request.active_idx - last_request.start_idx > 1:
            # If its chunked, add it back to the start of the request queue.
            self.preempted_prefill.put(
                PrefillRequest(
                    last_request_id, last_request, remote_name, dst_idx
                )
            )
        else:
            # Send to decode if not chunked
            last_request.bump_token_indices(start_idx=-last_request.start_idx)
            self.decode_push_socket.put_nowait(
                DecodeRequest(last_request_id, last_request)
            )

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
            # TODO: E2EOPT-231
            input_context._completion_start_idx -= 1  # type: ignore
            self.decode_push_socket.put_nowait(
                DecodeRequest(req_id, input_context)
            )

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
                logger.exception("An error occurred during scheduling.")
                raise e


def load_prefill_scheduler(
    zmq_ctx: zmq.Context,
    settings: Settings,
    pipeline: TokenGenerator,
    pc: ProcessControl,
    max_batch_size_ce: int,
    target_tokens_per_batch_ce: Optional[int],
    batch_timeout: Optional[float],
    enable_chunked_prefill: bool,
) -> PrefillScheduler:
    if enable_chunked_prefill == True and target_tokens_per_batch_ce is None:
        raise RuntimeError(
            "if enable_chunked_prefill=True, target_tokens_per_batch_ce must be provided"
        )

    if target_tokens_per_batch_ce is None:
        target_tokens_per_batch_ce = -1

    # Create Scheduler Config.
    scheduler_config = PrefillSchedulerConfig(
        max_batch_size_ce=max_batch_size_ce,
        batch_timeout=batch_timeout,
        enable_chunked_prefill=enable_chunked_prefill,
        target_tokens_per_batch_ce=target_tokens_per_batch_ce,
    )

    # Get Paged Manager
    paged_manager = get_paged_manager(pipeline)

    if paged_manager is None:
        raise RuntimeError(
            "A paged KV cache manager must be present to use the DecodeScheduler"
        )

    if (
        settings.prefill_zmq_endpoint is None
        or settings.decode_zmq_endpoint is None
    ):
        raise ValueError(
            "both prefil_zmq_endpoint and decode_zmq_endpoint must be provided in Server settings to run the DecodeScheduler."
        )

    return PrefillScheduler(
        process_control=pc,
        pipeline=pipeline,
        scheduler_config=scheduler_config,
        paged_manager=paged_manager,
        zmq_ctx=zmq_ctx,
        prefill_zmq_endpoint=settings.prefill_zmq_endpoint,
        decode_zmq_endpoint=settings.decode_zmq_endpoint,
    )
