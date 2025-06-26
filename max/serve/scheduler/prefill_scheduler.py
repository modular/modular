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
import uuid
from dataclasses import dataclass
from typing import Optional, Union

import zmq
from max._core import nixl
from max.nn.kv_cache import (
    KVTransferEngine,
    KVTransferEngineMetadata,
    PagedKVCacheManager,
    XferReqData,
)
from max.pipelines.core import TextAndVisionContext, TextContext, TokenGenerator
from max.pipelines.lib.pipeline import get_paged_manager
from max.profiler import traced
from max.serve.config import Settings
from max.serve.kvcache_agent.dispatcher_base import MessageType, ReplyContext
from max.serve.kvcache_agent.dispatcher_client import DispatcherClient
from max.serve.scheduler.base import PrefillRequest, PrefillResponse

from .base import Scheduler

logger = logging.getLogger("max.serve")


@dataclass
class PrefillSchedulerConfig:
    """Prefill Specific Scheduler Config."""

    max_batch_size_ce: int
    """The maximum number of requests that can be in the context encoding batch."""

    enable_chunked_prefill: bool = True
    """Enables chunked prefill, where the scheduler splits requests into chunks to ensure
    each batch contains exactly `target_tokens_per_batch_ce` tokens."""

    target_tokens_per_batch_ce: int = 4096
    """The target total number of tokens to encode in the context encoding batch."""


class PrefillScheduler(Scheduler):
    def __init__(
        self,
        pipeline: TokenGenerator,
        scheduler_config: PrefillSchedulerConfig,
        paged_manager: PagedKVCacheManager,
        *,
        zmq_ctx: zmq.Context,
        dispatcher_client: DispatcherClient,
    ):
        self.pipeline = pipeline
        self.scheduler_config = scheduler_config
        self.paged_manager = paged_manager

        # Initialize Scheduler state.
        self.active_batch: dict[
            str, Union[TextAndVisionContext, TextContext]
        ] = {}
        self.pending_transfers: dict[str, PrefillRequest] = {}
        self.active_transfers: dict[
            str, tuple[Union[TextAndVisionContext, TextContext], XferReqData]
        ] = {}
        self.available_cache_indices = set(
            range(self.scheduler_config.max_batch_size_ce)
        )
        self.preempted_prefill: queue.Queue[PrefillRequest] = queue.Queue()

        self.dispatcher_client = dispatcher_client
        self.dispatcher_client.register_request_handler(
            MessageType.PREFILL_REQUEST, self.handle_prefill_request
        )
        self.dispatcher_client.register_request_handler(
            MessageType.TRANSFER_ENGINE_REQUEST,
            self.handle_transfer_engine_request,
        )

        self.request_id_to_reply_context: dict[str, ReplyContext] = {}
        self.prefill_requests: queue.Queue[PrefillRequest] = queue.Queue()

        # Create Transfer Engine
        self.transfer_engine = KVTransferEngine(
            name=f"prefill_agent_{uuid.uuid4()}",
            listen_port=8047,
            tensor=self.paged_manager.device_tensors[0],
            total_num_pages=self.paged_manager.total_num_pages,
        )

    @traced
    def handle_transfer_engine_request(
        self, message: KVTransferEngineMetadata, reply_context: ReplyContext
    ) -> None:
        """Handles a engine registration request from the dispatcher."""
        logger.info(f"connecting to remote transfer_engine: {message.name}")
        self.transfer_engine.connect(message)

        self.dispatcher_client.send_reply(
            MessageType.TRANSFER_ENGINE_RESPONSE,
            self.transfer_engine.metadata,
            reply_context,
        )

    def handle_prefill_request(
        self, message: PrefillRequest, reply_context: ReplyContext
    ) -> None:
        """Handles a prefill request from the dispatcher."""
        self.prefill_requests.put(message)
        self.request_id_to_reply_context[message.id] = reply_context

    @traced
    def send_prefill_complete_response(
        self,
        request_id: str,
        data: Union[TextAndVisionContext, TextContext],
        xfer_data: XferReqData,
    ) -> None:
        if request_id not in self.request_id_to_reply_context:
            logger.error(
                f"Request ID {request_id} not found in request_id_to_reply_context"
            )
            return
        reply_context = self.request_id_to_reply_context.pop(request_id)

        self.dispatcher_client.send_reply(
            MessageType.PREFILL_RESPONSE,
            PrefillResponse(
                id=request_id, context=data, transfer_metadata=xfer_data
            ),
            reply_context,
        )

    def get_prefill_request(self) -> PrefillRequest:
        """Gets a request from the prefill request queue, checking preempted requests first.

        Returns:
            PrefillRequest: A prefill request.

        Raises:
            queue.Empty: If no requests are available.
            zmq.ZMQError: If there is an error receiving from the socket.
        """
        # First try and return from pre-empted requests queue.
        if not self.preempted_prefill.empty():
            return self.preempted_prefill.get()

        return self.prefill_requests.get_nowait()

    def return_to_prefill_queue(
        self,
        prefill_request: PrefillRequest,
    ) -> None:
        """Releases the cache index back to the available pool and cleans up pipeline
        resources before returning the request to the preempted queue.

        Args:
            req_id: The ID of the request to return
            data: The Union[TextAndVisionContext, TextContext] containing the request data
        """
        self.pipeline.release(prefill_request.context)
        prefill_request.context.reset()
        self.preempted_prefill.put(prefill_request)

    @traced
    def cleanup_active_transfers(self) -> None:
        """Cleans up completed transfers from the active transfers dictionary.

        Checks the status of all active transfers. For any transfer that is no longer in progress:
        - Returns the cache index back to the available pool
        - Releases pipeline resources
        - Removes the transfer from active_transfers
        """
        to_be_deleted = []
        for req_id, (context, transfer) in self.active_transfers.items():
            status = self.transfer_engine.get_transfer_status(transfer)

            if status != nixl.Status.IN_PROG:
                self.available_cache_indices.add(context.cache_seq_id)
                self.pipeline.release(context)
                to_be_deleted.append(req_id)

        for id in to_be_deleted:
            del self.active_transfers[id]

    @traced
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
                prefill_request = self.get_prefill_request()
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
            self.pending_transfers[prefill_request.id] = prefill_request

    @traced
    def schedule(self) -> None:
        """Executes the current batch of requests and sends completed requests to decode.

        Processes the active batch through the pipeline, handles any chunked prefill requests,
        and sends completed requests to the decode queue while resetting their token indices.
        """
        # Execute the Batch
        _ = self.pipeline.next_token(self.active_batch, num_steps=1)

        # Send completed requests to decode queue.
        # TODO: E2EOPT-275 Handle chunked requests.
        while self.active_batch:
            req_id, input_context = self.active_batch.popitem()
            logger.info("received request from decode node.")

            # Get Remote Metadata.
            prefill_request = self.pending_transfers.pop(req_id)
            remote_metadata = self.transfer_engine.remote_connections[
                prefill_request.transfer_engine_name
            ]

            # Retrieve source block ids.
            src_idx = self.paged_manager.block_manager.get_req_blocks(
                prefill_request.context.cache_seq_id,
            )

            # Bump this back, so the token is returned.
            input_context._completion_start_idx -= 1

            logger.info("initiating transfer from prefill worker.")
            xfer_data = self.transfer_engine.initiate_send_xfer(
                remote_metadata,
                src_idx,
                prefill_request.block_ids,
            )
            self.active_transfers[prefill_request.id] = (
                prefill_request.context,
                xfer_data,
            )

            logger.info("returning response to decode node")
            self.send_prefill_complete_response(
                req_id, input_context, xfer_data
            )

    @traced
    def run_iteration(self) -> None:
        """Main scheduling loop that processes prefill requests.

        Receives requests, creates batches, and schedules them for processing
        while handling errors and cancelled requests.
        """
        # Cleanup active transfers.
        self.cleanup_active_transfers()

        # Create a new batch
        self.update_batch()

        # Break out of loop if batch is empty.
        if not self.active_batch:
            return

        self.schedule()

    def needs_dispatcher_client(self) -> bool:
        """Whether the scheduler needs a dispatcher client."""
        return True


def load_prefill_scheduler(
    zmq_ctx: zmq.Context,
    settings: Settings,
    pipeline: TokenGenerator,
    max_batch_size_ce: int,
    target_tokens_per_batch_ce: Optional[int],
    enable_chunked_prefill: bool,
    dispatcher_client: DispatcherClient,
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
        enable_chunked_prefill=enable_chunked_prefill,
        target_tokens_per_batch_ce=target_tokens_per_batch_ce,
    )

    # Get Paged Manager
    paged_manager = get_paged_manager(pipeline)

    if paged_manager is None:
        raise RuntimeError(
            "A paged KV cache manager must be present to use the PrefillScheduler"
        )

    return PrefillScheduler(
        pipeline=pipeline,
        scheduler_config=scheduler_config,
        paged_manager=paged_manager,
        zmq_ctx=zmq_ctx,
        dispatcher_client=dispatcher_client,
    )
