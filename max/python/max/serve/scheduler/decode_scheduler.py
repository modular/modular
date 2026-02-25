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
import queue
import time
import uuid
from collections import OrderedDict

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
from max.kv_cache import (
    InsufficientBlocksError,
    KVTransferEngine,
    KVTransferEngineMetadata,
    PagedKVCacheManager,
    TransferReqData,
)
from max.pipelines.core import TextAndVisionContext, TextContext
from max.pipelines.lib import PipelineConfig, TextGenerationPipeline
from max.profiler import Tracer, traced
from max.serve.config import Settings
from max.serve.scheduler.base import (
    CancelRequest,
    PrefillRequest,
    PrefillResponse,
)
from max.serve.scheduler.di_dispatchers import DecodeDispatcherClientV2

from .base import SchedulerProgress
from .batch_constructor import TextBatchConstructor
from .batch_constructor.text_batch_constructor import BatchSchedulingStrategy
from .config import TokenGenerationSchedulerConfig
from .utils import SchedulerLogger, get_cancelled_reqs

logger = logging.getLogger("max.serve")


class DecodeScheduler(Scheduler):
    def __init__(
        self,
        pipeline: Pipeline[
            TextGenerationInputs[TextContext], TextGenerationOutput
        ],
        scheduler_config: TokenGenerationSchedulerConfig,
        kv_cache: PagedKVCacheManager,
        *,
        request_queue: MAXPullQueue[TextContext | TextAndVisionContext],
        response_queue: MAXPushQueue[
            dict[RequestID, SchedulerResult[TextGenerationOutput]]
        ],
        cancel_queue: MAXPullQueue[list[RequestID]],
        dispatcher: DecodeDispatcherClientV2,
    ) -> None:
        # Initialize Pipeline and Config
        self.scheduler_config = scheduler_config
        self.pipeline = pipeline
        self.kv_cache = kv_cache

        # Initialize Queues
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.cancel_queue = cancel_queue

        self.dispatcher = dispatcher

        # Initialize Scheduler state.
        self.pending_reqs: OrderedDict[RequestID, TextContext] = OrderedDict()
        self.prefill_reqs: dict[RequestID, tuple[TextContext, int]] = {}
        self.inflight_transfers: dict[RequestID, TransferReqData] = {}
        self.prefill_reqs_per_replica: list[int] = [
            0 for _ in range(self.kv_cache.num_replicas)
        ]

        self.transfer_engine = KVTransferEngine(
            name=f"decode_agent_{uuid.uuid4()}",
            # TODO: Also support scales tensors
            tensors=[
                self.kv_cache.get_device_buffer(replica_idx).values
                for replica_idx in range(self.kv_cache.num_replicas)
            ],
            # Assume all replicas have the same number of pages.
            total_num_pages=self.kv_cache.get_num_pages(replica_idx=0),
        )

        self.batch_constructor = TextBatchConstructor(
            scheduler_config=scheduler_config,
            pipeline=pipeline,
            kv_cache=kv_cache,
            batch_scheduling_strategy=BatchSchedulingStrategy.DECODE_FIRST,
        )
        self.scheduler_logger = SchedulerLogger()
        # None corresponds to the default destination address.
        # TODO: delete the default destination address.
        self.remote_endpoints: set[str] = set()

    @traced
    def handle_transfer_engine_response(
        self, message: KVTransferEngineMetadata
    ) -> None:
        logger.debug(f"connecting to remote transfer engine: {message.name}")
        self.transfer_engine.connect(message)

    def handle_prefill_response(self, message: PrefillResponse) -> None:
        """Handles a prefill response from the dispatcher."""
        # Update the context with the generated token
        request_id = message.id
        context, _ = self.prefill_reqs[request_id]
        context.update(message.generated_token_id)

        # Send singular token to the API process
        output = context.to_generation_output()
        self.response_queue.put_nowait(
            {request_id: SchedulerResult.create(output)}
        )

        self.inflight_transfers[request_id] = message.transfer_metadata

    @traced
    def send_prefill_request(
        self,
        request_id: RequestID,
        data: TextContext,
        dst_idxs: list[int],
        dst_replica_idx: int,
    ) -> None:
        """Pushes a request to the prefill socket.

        Args:
            request_id: The ID of the request to send
            data: The context containing the request data
            dst_idxs: The destination block indices for the request
            replica_idx: The replica the request is on for Decode

        Raises:
            zmq.ZMQError: If there is an error sending on the socket
        """
        # TODO: Do not crash the scheduler if a request does not have a target endpoint.
        #       Instead we should validate this in the frontend.
        if data.target_endpoint is None:
            raise ValueError(
                f"Target endpoint is not specified for the request {request_id}"
            )
        if data.target_endpoint not in self.remote_endpoints:
            self.dispatcher.send_request_nowait(
                self.transfer_engine.metadata,
                data.target_endpoint,
            )
            self.remote_endpoints.add(data.target_endpoint)

        assert data.tokens.generated_length == 0, (
            f"Invalid Context: Expected needs_ce to be True. Found: {data}"
        )

        # Set dst_idx to -1 to denote pages which the decode already has due to
        # prefix caching.
        for i in range(data.tokens.processed_length // self.kv_cache.page_size):
            dst_idxs[i] = -1

        self.dispatcher.send_request_nowait(
            PrefillRequest(
                id=request_id,
                context=data,
                transfer_engine_name=self.transfer_engine.name,
                dst_block_ids=dst_idxs,
                dst_replica_idx=dst_replica_idx,
            ),
            data.target_endpoint,
        )

    def reserve_memory_and_send_to_prefill(self) -> None:
        """Continuously pulls requests from the request queue and forwards them to the prefill node."""
        items = drain_queue(
            self.request_queue,
            max_items=self.scheduler_config.max_batch_size * 2,
        )

        for context in items:
            self.pending_reqs[context.request_id] = context

        while (
            self.pending_reqs
            and (
                len(self.batch_constructor.all_tg_reqs) + len(self.prefill_reqs)
            )
            < self.scheduler_config.max_batch_size
            and (
                self.kv_cache is None
                or any(
                    self.kv_cache.get_num_used_pages(replica_idx)
                    / self.kv_cache.get_num_pages(replica_idx)
                    < 0.9
                    for replica_idx in range(self.kv_cache.num_replicas)
                )
            )
        ):
            # Pop off request queue
            context = next(iter(self.pending_reqs.values()))
            req_id = context.request_id
            del self.pending_reqs[req_id]

            # Claim the slot with the paged manager
            replica_idx = self.batch_constructor.get_next_replica_idx(
                external_requests_per_replica=self.prefill_reqs_per_replica
            )
            self.kv_cache.claim(req_id, replica_idx=replica_idx)

            # Allocate enough memory needed to run the request for one step.
            # The blocks allocated here will be written via a KVCache transfer
            # from prefill -> decode.
            try:
                self.kv_cache.alloc(
                    context, replica_idx=replica_idx, num_steps=1
                )
            except InsufficientBlocksError:
                # If we don't have enough space, we will return this to the request queue.
                self.pending_reqs[req_id] = context
                self.pending_reqs.move_to_end(req_id, last=False)
                self.kv_cache.release(req_id, replica_idx=replica_idx)
                break

            # Send to the Prefill Node
            dst_idxs = self.kv_cache.get_req_blocks(
                req_id, replica_idx=replica_idx
            )
            self.prefill_reqs[req_id] = (context, replica_idx)
            self.prefill_reqs_per_replica[replica_idx] += 1
            self.send_prefill_request(req_id, context, dst_idxs, replica_idx)

    def _handle_cancelled_requests(self) -> None:
        for req_id in get_cancelled_reqs(self.cancel_queue):
            if self.batch_constructor.contains(req_id):
                # Remove it from the active batch.
                self.batch_constructor.release_request(req_id)
                # Send the cancelled result back to the response q
                self.response_queue.put_nowait(
                    {req_id: SchedulerResult.cancelled()}
                )

            # If it is pending prefill, remove the pending request.
            elif req_id in self.prefill_reqs:
                data, dst_replica_idx = self.prefill_reqs[req_id]

                # Remove from pending requests.
                del self.prefill_reqs[req_id]
                self.prefill_reqs_per_replica[dst_replica_idx] -= 1

                # TODO: Do not crash the scheduler if a request does not have a target endpoint.
                #       Instead we should validate this in the frontend.
                if data.target_endpoint is None:
                    raise ValueError(
                        f"Target endpoint is not specified for the request {req_id}."
                    )
                # Send a cancel request to the prefill node
                self.dispatcher.send_request_nowait(
                    CancelRequest(id=req_id), data.target_endpoint
                )

                # Send the cancelled result back to the response q
                self.response_queue.put_nowait(
                    {req_id: SchedulerResult.cancelled()}
                )

            else:
                logger.debug(
                    f"cancel request received on decode node for {req_id} not in pending or active batch."
                )

    def check_for_completed_transfers(self) -> None:
        """Checks for the completion of KVCache transfers.

        For transfers that have been completed, we will mark the request as ready
        for token generation by enqueuing it into the text batch constructor.

        We also ensure that the metadata is cleaned up in the transfer engine.
        """

        request_ids = list(self.inflight_transfers.keys())
        for request_id in request_ids:
            transfer_metadata = self.inflight_transfers[request_id]

            # Transfer is not complete, skip.
            if not self.transfer_engine.is_complete(transfer_metadata):
                continue

            # Cleanup the transfer.
            del self.inflight_transfers[request_id]
            self.transfer_engine.cleanup_transfer(transfer_metadata)

            # When cancelled, the request is removed from prefill_reqs
            # therefore the request should only be added to the active_batch
            # if it is still in prefill_reqs.
            if request_id not in self.prefill_reqs:
                continue

            # Remove from pending prefill requests and add to TG requests.
            context, dst_replica_idx = self.prefill_reqs.pop(request_id)
            self.batch_constructor.enqueue_new_request(context, dst_replica_idx)

        # Manage for cancelled requests
        self._handle_cancelled_requests()

    @traced
    def schedule(self, inputs: TextGenerationInputs[TextContext]) -> int:
        """Schedules a batch of requests for token generation and handles the responses.

        Args:
            inputs: The inputs containing the batch of requests to schedule.
        """
        assert len(inputs.batches) > 0
        responses = self.pipeline.execute(inputs)

        self.batch_constructor.advance_requests(inputs)

        # Release terminated requests
        num_terminated_requests = 0
        for request_id, response in responses.items():
            if response.is_done:
                self.batch_constructor.release_request(request_id)
                num_terminated_requests += 1

        # send the responses to the API process
        self.response_queue.put_nowait(
            {
                req_id: SchedulerResult.create(response)
                for req_id, response in responses.items()
            }
        )

        return num_terminated_requests

    def run_iteration(self) -> SchedulerProgress:
        """Main scheduling loop that processes decode requests.

        Receives requests, updates batches, and schedules them for processing
        while handling memory management.

        Returns:
            SchedulerProgress: Indicates whether work was performed in this iteration.
        """
        while True:
            try:
                reply = self.dispatcher.recv_reply_nowait()
            except queue.Empty:
                break
            if isinstance(reply, KVTransferEngineMetadata):
                self.handle_transfer_engine_response(reply)
            elif isinstance(reply, PrefillResponse):
                self.handle_prefill_response(reply)
            else:
                raise ValueError(f"Invalid reply type: {reply}")

        # Eagerly reserve memory and send to prefill worker
        self.reserve_memory_and_send_to_prefill()

        # Update the active decode batch
        self.check_for_completed_transfers()

        # Construct the batch to execute
        t0 = time.monotonic()
        inputs = self.batch_constructor.construct_batch()
        t1 = time.monotonic()
        batch_creation_time_s = t1 - t0

        # If the batch is empty, skip
        if len(inputs.flat_batch) == 0:
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
            kv_cache=self.kv_cache,
            batch_creation_time_s=batch_creation_time_s,
            batch_execution_time_s=batch_execution_time_s,
            num_pending_reqs=len(self.pending_reqs) + len(self.prefill_reqs),
            num_terminated_reqs=num_terminated_reqs,
            total_preemption_count=self.batch_constructor.total_preemption_count,
        )

        return SchedulerProgress.MADE_PROGRESS


def load_decode_scheduler(
    pipeline: TextGenerationPipeline[TextContext],
    pipeline_config: PipelineConfig,
    request_queue: MAXPullQueue[TextContext | TextAndVisionContext],
    response_queue: MAXPushQueue[
        dict[RequestID, SchedulerResult[TextGenerationOutput]]
    ],
    cancel_queue: MAXPullQueue[list[RequestID]],
    settings: Settings,
) -> DecodeScheduler:
    # Create Scheduler Config.
    scheduler_config = TokenGenerationSchedulerConfig.from_pipeline_config(
        pipeline_config
    )

    return DecodeScheduler(
        pipeline=pipeline,
        scheduler_config=scheduler_config,
        kv_cache=pipeline.kv_manager,
        request_queue=request_queue,
        response_queue=response_queue,
        cancel_queue=cancel_queue,
        dispatcher=DecodeDispatcherClientV2(bind_addr=settings.di_bind_address),
    )
