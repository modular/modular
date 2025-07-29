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
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Union

from max._core import nixl
from max.interfaces import TextGenerationInputs, TokenGenerator
from max.nn.kv_cache import (
    KVTransferEngine,
    KVTransferEngineMetadata,
    PagedKVCacheManager,
    XferReqData,
)
from max.pipelines.core import TextAndVisionContext, TextContext
from max.pipelines.lib import PipelineConfig
from max.pipelines.lib.pipeline import get_paged_manager
from max.profiler import traced
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

        self.dispatcher_client = dispatcher_client
        self.dispatcher_client.register_request_handler(
            MessageType.PREFILL_REQUEST, self.handle_prefill_request
        )
        self.dispatcher_client.register_request_handler(
            MessageType.TRANSFER_ENGINE_REQUEST,
            self.handle_transfer_engine_request,
        )

        self.request_id_to_reply_context: dict[str, ReplyContext] = {}
        self.prefill_requests: deque[PrefillRequest] = deque()

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
        self.prefill_requests.append(message)
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

    def return_to_prefill_queue(
        self,
        prefill_request: PrefillRequest,
    ) -> None:
        """Releases pipeline resources and cleans up the request before returning
        it to the preempted queue.

        """
        self.pipeline.release(prefill_request.context.request_id)
        prefill_request.context.reset()
        self.prefill_requests.appendleft(prefill_request)

    def cleanup_active_transfers(self) -> None:
        """Cleans up completed transfers from the active transfers dictionary.

        Checks the status of all active transfers. For any transfer that is no longer in progress:
        - Releases pipeline resources
        - Removes the transfer from active_transfers
        """
        to_be_deleted = []
        for req_id, (context, transfer) in self.active_transfers.items():
            status = self.transfer_engine.get_transfer_status(transfer)

            if status != nixl.Status.IN_PROG:
                self.pipeline.release(context.request_id)
                to_be_deleted.append(req_id)

        for id in to_be_deleted:
            del self.active_transfers[id]

    def update_batch(self) -> None:
        """Updates the active batch by pulling requests from the prefill queue.

        Processes requests up to max_batch_size_ce, handling chunking.
        For each request:
        - Attempts to schedule via paged manager
        - Chunks inputs if enabled and batch token length exceeds target
        - Tracks total batch token length
        """
        batch_token_length = 0
        while (
            len(self.active_batch) < self.scheduler_config.max_batch_size_ce
            and self.prefill_requests
        ):
            prefill_request = self.prefill_requests.popleft()
            prefill_request.context.reset()
            logger.info("received from decode node!")

            if not self.paged_manager.contains(prefill_request.id):
                self.paged_manager.external_claim(prefill_request.id)

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
        inputs = TextGenerationInputs(batch=self.active_batch, num_steps=1)
        _ = self.pipeline.next_token(inputs)

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
                prefill_request.context.request_id,
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


def load_prefill_scheduler(
    pipeline: TokenGenerator,
    pipeline_config: PipelineConfig,
    dispatcher_client: DispatcherClient,
) -> PrefillScheduler:
    enable_chunked_prefill = pipeline_config.enable_chunked_prefill
    target_tokens_per_batch_ce = pipeline_config.target_num_new_tokens
    max_batch_size_ce = pipeline_config.max_ce_batch_size

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
        dispatcher_client=dispatcher_client,
    )
