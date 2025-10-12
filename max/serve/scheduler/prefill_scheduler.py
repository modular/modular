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
import queue
import time
import uuid

from max.interfaces import (
    Pipeline,
    RequestID,
    Scheduler,
    TextGenerationInputs,
    TextGenerationOutput,
)
from max.nn.kv_cache import (
    KVTransferEngine,
    KVTransferEngineMetadata,
    PagedKVCacheManager,
    TransferReqData,
)
from max.pipelines.core import TextAndVisionContext, TextContext
from max.pipelines.lib import (
    PipelineConfig,
)
from max.pipelines.lib.pipeline import get_paged_manager
from max.profiler import Tracer, traced
from max.serve.config import Settings
from max.serve.queue.zmq_queue import ClientIdentity
from max.serve.scheduler.base import (
    CancelRequest,
    PrefillRequest,
    PrefillResponse,
)
from max.serve.scheduler.di_dispatchers import PrefillDispatcherServerV2
from max.serve.scheduler.text_batch_constructor import (
    TextBatchConstructor,
    TokenGenerationSchedulerConfig,
)

from .base import SchedulerProgress
from .utils import SchedulerLogger, add_newly_encoded_reqs_to_tg_batch

logger = logging.getLogger("max.serve")


class PrefillScheduler(Scheduler):
    def __init__(
        self,
        pipeline: Pipeline[
            TextGenerationInputs[TextContext], TextGenerationOutput
        ],
        scheduler_config: TokenGenerationSchedulerConfig,
        paged_cache: PagedKVCacheManager,
        dispatcher: PrefillDispatcherServerV2,
    ) -> None:
        self.pipeline = pipeline
        self.scheduler_config = scheduler_config
        self.paged_cache = paged_cache
        # Initialize Scheduler state.
        self.active_transfers: dict[
            RequestID,
            tuple[TextAndVisionContext | TextContext, TransferReqData],
        ] = {}
        self.request_id_to_reply_context: dict[
            RequestID, tuple[ClientIdentity, str, list[int]]
        ] = {}

        # Create Transfer Engine
        self.transfer_engine = KVTransferEngine(
            name=f"prefill_agent_{uuid.uuid4()}",
            tensors=paged_cache.device_tensors,
            total_num_pages=paged_cache.total_num_pages,
        )

        self.outstanding_cancelled_requests: set[RequestID] = set()

        self.batch_constructor = TextBatchConstructor(
            scheduler_config=scheduler_config,
            pipeline=pipeline,
            paged_cache=paged_cache,
        )
        self.scheduler_logger = SchedulerLogger()
        self.dispatcher = dispatcher

    @traced
    def handle_cancel_request(self, message: CancelRequest) -> None:
        """Handles a cancel request by adding the request ID to the set of outstanding cancelled requests."""
        self.outstanding_cancelled_requests.add(message.id)

    @traced
    def handle_transfer_engine_request(
        self, message: KVTransferEngineMetadata, identity: ClientIdentity
    ) -> None:
        """Handles a engine registration request from the dispatcher."""
        logger.debug(f"connecting to remote transfer_engine: {message.name}")
        if message.name in self.transfer_engine.remote_connections:
            logger.info(f"Transfer engine {message.name} already connected")
            return

        self.transfer_engine.connect(message)

        self.dispatcher.send_reply_nowait(
            self.transfer_engine.metadata,
            identity,
        )

    def handle_prefill_request(
        self, message: PrefillRequest, identity: ClientIdentity
    ) -> None:
        """Handles a prefill request from the dispatcher."""
        logger.debug("received request from decode node.")
        context = message.context
        assert context.needs_ce, (
            f"Expected needs_ce to be True. Invalid context: {context}"
        )
        # It is possible for the context to have a non-zero start_idx due to
        # decode using prefix caching.
        context.reset()
        self.batch_constructor.ce_reqs[message.id] = context
        self.request_id_to_reply_context[message.id] = (
            identity,
            message.transfer_engine_name,
            message.block_ids,
        )

    def cleanup_active_transfers(self) -> None:
        """Cleans up completed transfers from the active transfers dictionary.

        Checks the status of all active transfers. For any transfer that is no longer in progress:
        - Releases pipeline resources
        - Removes the transfer from active_transfers
        """
        to_be_deleted = []
        for req_id, (context, transfer) in self.active_transfers.items():
            if self.transfer_engine.is_complete(transfer):
                self.transfer_engine.cleanup_transfer(transfer)
                self.pipeline.release(context.request_id)
                to_be_deleted.append(req_id)

        for id in to_be_deleted:
            del self.active_transfers[id]

    @traced
    def schedule(self, inputs: TextGenerationInputs[TextContext]) -> int:
        """Executes the current batch of requests and sends completed requests to decode.

        Processes the active batch through the pipeline, handles any chunked prefill requests,
        and sends completed requests to the decode queue while resetting their token indices.
        """
        # Execute the Batch
        assert len(inputs.batches) > 0
        responses = self.pipeline.execute(inputs)

        add_newly_encoded_reqs_to_tg_batch(
            inputs.batch,
            responses,
            self.batch_constructor,
        )

        # Send fully encoded requests to decode queue.
        for req_id, context in self.batch_constructor.tg_reqs.items():
            identity, transfer_engine_name, dst_idxs = (
                self.request_id_to_reply_context.pop(req_id)
            )

            # If cancelled, throw away result.
            if req_id in self.outstanding_cancelled_requests:
                self.outstanding_cancelled_requests.remove(req_id)
                continue

            # Get Remote Metadata.
            remote_metadata = self.transfer_engine.remote_connections[
                transfer_engine_name
            ]

            # Retrieve source block ids.
            src_idxs = self.paged_cache.block_manager.get_req_blocks(
                context.request_id,
            )
            assert len(src_idxs) == len(dst_idxs)

            # Transfer only the blocks that are not already on decode node.
            num_already_cached_blocks = dst_idxs.count(-1)
            src_idxs = src_idxs[num_already_cached_blocks:]
            dst_idxs = dst_idxs[num_already_cached_blocks:]
            assert dst_idxs.count(-1) == 0

            # Initiate the KV transfer
            logger.debug("initiating transfer from prefill worker.")
            transfer_data = self.transfer_engine.initiate_send_transfer(
                remote_metadata,
                src_idxs,
                dst_idxs,
            )
            self.active_transfers[req_id] = (context, transfer_data)

            assert not context.needs_ce, (
                f"Invalid Context: Expected needs_ce to be False. Found: {context}"
            )
            assert context.start_idx > 0, (
                f"Invalid Context: Expected start_idx to be greater than 0. Found: {context}"
            )
            self.dispatcher.send_reply_nowait(
                PrefillResponse(
                    id=req_id,
                    generated_token_id=context.last_generated_token,
                    transfer_metadata=transfer_data,
                ),
                identity,
            )

        # Remove all TG requests from the batch constructor.
        num_terminated_reqs = len(self.batch_constructor.tg_reqs)
        self.batch_constructor.tg_reqs.clear()
        return num_terminated_reqs

    def run_iteration(self) -> SchedulerProgress:
        """Main scheduling loop that processes prefill requests.

        Receives requests, creates batches, and schedules them for processing
        while handling errors and cancelled requests.

        Returns:
            SchedulerProgress: Indicates whether work was performed in this iteration.
        """

        while True:
            try:
                request, identity = self.dispatcher.recv_request_nowait()
            except queue.Empty:
                break
            if isinstance(request, CancelRequest):
                self.handle_cancel_request(request)
            elif isinstance(request, KVTransferEngineMetadata):
                self.handle_transfer_engine_request(request, identity)
            elif isinstance(request, PrefillRequest):
                self.handle_prefill_request(request, identity)
            else:
                raise ValueError(f"Invalid request type: {request}")

        # Cleanup active transfers.
        self.cleanup_active_transfers()

        # Construct the batch to execute
        t0 = time.monotonic()
        inputs = self.batch_constructor.construct_batch()
        t1 = time.monotonic()
        batch_creation_time_s = t1 - t0

        # If the batch is empty, skip
        if len(inputs.batch) == 0:
            return SchedulerProgress.NO_PROGRESS

        # Schedule the batch
        t0 = time.monotonic()
        with Tracer(f"_schedule({inputs})"):
            num_terminated_reqs = self.schedule(inputs)
        t1 = time.monotonic()
        batch_execution_time_s = t1 - t0

        # Log batch metrics
        self.scheduler_logger.log_metrics(
            sch_config=self.scheduler_config,
            inputs=inputs,
            paged_cache=self.paged_cache,
            batch_creation_time_s=batch_creation_time_s,
            batch_execution_time_s=batch_execution_time_s,
            num_pending_reqs=len(self.batch_constructor.ce_reqs),
            num_terminated_reqs=num_terminated_reqs,
            total_preemption_count=self.batch_constructor.total_preemption_count,
        )

        return SchedulerProgress.MADE_PROGRESS


def load_prefill_scheduler(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
    pipeline_config: PipelineConfig,
    settings: Settings,
) -> PrefillScheduler:
    # Create Scheduler Config.
    scheduler_config = TokenGenerationSchedulerConfig.from_pipeline_config(
        pipeline_config
    )

    # Get Paged Manager
    paged_cache = get_paged_manager(pipeline)

    if paged_cache is None:
        raise RuntimeError(
            "A paged KV cache manager must be present to use the PrefillScheduler"
        )

    return PrefillScheduler(
        pipeline=pipeline,
        scheduler_config=scheduler_config,
        paged_cache=paged_cache,
        dispatcher=PrefillDispatcherServerV2(
            bind_addr=settings.dispatcher_config.transport_config.bind_address
        ),
    )
