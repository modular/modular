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
from typing import Union

from max.interfaces import (
    Pipeline,
    RequestID,
    SchedulerResult,
    TextGenerationInputs,
    TextGenerationOutput,
    msgpack_numpy_decoder,
    msgpack_numpy_encoder,
)
from max.nn.kv_cache import (
    KVTransferEngine,
    KVTransferEngineMetadata,
    PagedKVCacheManager,
)
from max.pipelines.core import TextAndVisionContext, TextContext
from max.pipelines.lib import PipelineConfig
from max.pipelines.lib.pipeline import get_paged_manager
from max.profiler import Tracer, traced
from max.serve.config import Settings
from max.serve.kvcache_agent.dispatcher_base import MessageType
from max.serve.kvcache_agent.dispatcher_client import DispatcherClient
from max.serve.queue.zmq_queue import ZmqPullSocket, ZmqPushSocket
from max.serve.scheduler.base import PrefillRequest, PrefillResponse

from .base import Scheduler
from .text_batch_constructor import (
    TextBatchConstructor,
    TokenGenerationSchedulerConfig,
)
from .utils import (
    OrderedDict,
    SchedulerLogger,
    SchedulerOutput,
    maybe_restore_chunked_request,
    release_terminated_requests,
)

logger = logging.getLogger("max.serve")


class DecodeScheduler(Scheduler):
    def __init__(
        self,
        pipeline: Pipeline[
            TextGenerationInputs[Union[TextContext, TextAndVisionContext]],
            TextGenerationOutput,
        ],
        scheduler_config: TokenGenerationSchedulerConfig,
        paged_manager: PagedKVCacheManager,
        *,
        request_zmq_endpoint: str,
        response_zmq_endpoint: str,
        cancel_zmq_endpoint: str,
        dispatcher_client: DispatcherClient,
    ) -> None:
        # Initialize Pipeline and Config
        self.scheduler_config = scheduler_config
        self.pipeline = pipeline
        self.paged_manager = paged_manager

        # Initialize Queues
        self.request_pull_socket = ZmqPullSocket[
            tuple[str, Union[TextContext, TextAndVisionContext]]
        ](
            zmq_endpoint=request_zmq_endpoint,
            deserialize=msgpack_numpy_decoder(
                tuple[str, Union[TextContext, TextAndVisionContext]]
            ),
        )
        self.response_push_socket = ZmqPushSocket[
            dict[str, SchedulerResult[TextGenerationOutput]]
        ](
            zmq_endpoint=response_zmq_endpoint,
            serialize=msgpack_numpy_encoder(),
        )
        self.cancel_pull_socket = ZmqPullSocket[list[RequestID]](
            zmq_endpoint=cancel_zmq_endpoint,
            deserialize=msgpack_numpy_decoder(
                list[RequestID],
            ),
        )

        self.dispatcher_client = dispatcher_client
        self.dispatcher_client.register_reply_handler(
            MessageType.PREFILL_RESPONSE, self.handle_prefill_response
        )
        self.dispatcher_client.register_reply_handler(
            MessageType.TRANSFER_ENGINE_RESPONSE,
            self.handle_transfer_engine_response,
        )

        self.prefill_responses: dict[str, PrefillResponse] = {}

        # Initialize Scheduler state.
        self.pending_reqs: OrderedDict[
            RequestID, Union[TextContext, TextAndVisionContext]
        ] = OrderedDict()
        self.pending_prefill_requests: set[RequestID] = set()

        # Create Transfer Engine
        self.transfer_engine = KVTransferEngine(
            name=f"decode_agent_{uuid.uuid4()}",
            tensors=self.paged_manager.device_tensors,
            total_num_pages=self.paged_manager.total_num_pages,
        )

        self.batch_constructor = TextBatchConstructor(
            scheduler_config=scheduler_config,
            pipeline=pipeline,
            paged_cache=paged_manager,
        )
        self.scheduler_logger = SchedulerLogger()
        # None corresponds to the default destination address.
        # TODO: delete the default destination address.
        self.remote_endpoints: set[str | None] = set()

    @traced
    def handle_transfer_engine_response(
        self, message: KVTransferEngineMetadata
    ) -> None:
        logger.debug(f"connecting to remote transfer engine: {message.name}")
        self.transfer_engine.connect(message)

    def handle_prefill_response(self, message: PrefillResponse) -> None:
        """Handles a prefill response from the dispatcher."""
        # Send singular token to the API process
        context = message.context
        output = context.to_generation_output()
        self.response_push_socket.put_nowait(
            {message.id: SchedulerResult.create(output)}
        )

        # Add to prefill responses afterwards to avoid race condition
        self.prefill_responses[message.transfer_metadata.xfer_name] = message

    @traced
    def send_prefill_request(
        self,
        request_id: RequestID,
        data: Union[TextContext, TextAndVisionContext],
        dst_idx: list[int],
    ) -> None:
        """Pushes a request to the prefill socket.

        Args:
            request_id: The ID of the request to send
            data: The Union[TextContext, TextAndVisionContext] containing the request data

        Raises:
            zmq.ZMQError: If there is an error sending on the socket
        """

        if data.target_endpoint not in self.remote_endpoints:
            self.dispatcher_client.send(
                MessageType.TRANSFER_ENGINE_REQUEST,
                self.transfer_engine.metadata,
                destination_address=data.target_endpoint,
            )
            self.remote_endpoints.add(data.target_endpoint)

        self.dispatcher_client.send(
            MessageType.PREFILL_REQUEST,
            PrefillRequest(
                id=request_id,
                context=data,
                transfer_engine_name=self.transfer_engine.name,
                block_ids=dst_idx,
            ),
            destination_address=data.target_endpoint,
        )

    def reserve_memory_and_send_to_prefill(self) -> None:
        """Continuously pulls requests from the request queue and forwards them to the prefill node.

        Breaks when the request queue is empty. Memory reservation is pending implementation.
        """
        self.pending_reqs |= dict(self.request_pull_socket.drain_nowait())

        while (
            self.pending_reqs
            and (
                len(self.batch_constructor.tg_reqs)
                + len(self.pending_prefill_requests)
            )
            < self.scheduler_config.max_batch_size_tg
        ):
            # Pop off request queue
            req_id, context = self.pending_reqs.popitem(last=False)

            # Claim the slot with the paged manager
            if not self.paged_manager.contains(req_id):
                self.paged_manager.external_claim(req_id)

            # Prefetch memory for Context Encoding eagerly, this only needs to be
            # for one step.
            if not self.paged_manager.prefetch(context, 1):
                # If we don't have enough space in the paged manager
                # return this to the request queue.
                self.pending_reqs[req_id] = context
                self.pending_reqs.move_to_end(req_id, last=False)
                self.paged_manager.release(req_id)
                break

            # Send to the Prefill Node
            dst_idxs = self.paged_manager.block_manager.get_req_blocks(req_id)
            self.pending_prefill_requests.add(req_id)
            self.send_prefill_request(req_id, context, dst_idxs)

    def _handle_cancelled_requests(self):
        while True:
            try:
                for request_id in self.cancel_pull_socket.get_nowait():
                    # Remove it from the active batch.
                    if request_id in self.batch_constructor.tg_reqs:
                        del self.batch_constructor.tg_reqs[request_id]

                        # Send the cancelled result back to the response q
                        self.response_push_socket.put_nowait(
                            {request_id: SchedulerResult.cancelled()}
                        )

                    # If it is pending prefill, remove the pending request.
                    elif request_id in self.pending_prefill_requests:
                        # Remove from pending requests.
                        self.pending_prefill_requests.remove(request_id)

                        # Send a cancel request to the prefill node
                        self.dispatcher_client.send(
                            MessageType.CANCEL_REQUEST, request_id
                        )

                        # Send the cancelled result back to the response q
                        self.response_push_socket.put_nowait(
                            {request_id: SchedulerResult.cancelled()}
                        )

                    else:
                        logger.debug(
                            f"cancel request received on decode node for {request_id} not in pending or active batch."
                        )

            except queue.Empty:
                break

    def check_for_completed_transfers(self) -> None:
        """Updates the active batch by adding new requests from the decode queue and managing memory prefetching.

        Adds new requests to the batch up to the maximum batch size. For each request, attempts to prefetch
        required memory. If prefetch fails, handles preemption by returning newer requests to the decode queue.
        """

        transfer_names = list(self.prefill_responses.keys())
        for transfer_name in transfer_names:
            prefill_response = self.prefill_responses[transfer_name]
            transfer_metadata = prefill_response.transfer_metadata

            # Transfer is not complete, skip.
            if not self.transfer_engine.is_complete(transfer_metadata):
                continue

            # Cleanup the transfer.
            del self.prefill_responses[transfer_name]
            self.transfer_engine.cleanup_transfer(transfer_metadata)

            # When cancelled, the request is removed from prefill_requests
            # therefore the request should only be added to the active_batch
            # if it is still in prefill_requests.
            if prefill_response.id not in self.pending_prefill_requests:
                continue

            # Remove from pending prefill requests and add to TG requests.
            self.pending_prefill_requests.remove(prefill_response.id)
            context = prefill_response.context
            self.batch_constructor.tg_reqs[prefill_response.id] = context

        # Manage for cancelled requests
        self._handle_cancelled_requests()

    @traced
    def schedule(self, sch_output: SchedulerOutput) -> None:
        """Schedules a batch of requests for token generation and handles the responses.

        Args:
            sch_output: The scheduler output containing the batch of requests to schedule.
        """
        assert sch_output.batch_size > 0
        batch = sch_output.batch_inputs
        responses = self.pipeline.execute(
            TextGenerationInputs([batch], num_steps=sch_output.num_steps)
        )

        # Even though this is CE specific, it is possible for decode_scheduler
        # to execute CE if a request is preempted. We do not send such a preempted
        # request back to prefill. Instead the decode worker just runs CE on the
        # preempted request.
        self.batch_constructor.tg_reqs |= batch
        maybe_restore_chunked_request(
            batch,
            responses,
            self.batch_constructor.ce_reqs,
        )

        # remove terminated requests from the batch
        release_terminated_requests(
            sch_output,
            responses,
            self.pipeline,
            self.batch_constructor.tg_reqs,
        )

        # send the responses to the API process
        self.response_push_socket.put_nowait(
            {
                req_id: SchedulerResult.create(response)
                for req_id, response in responses.items()
            }
        )

    def run_iteration(self) -> None:
        """Main scheduling loop that processes decode requests.

        Receives requests, updates batches, and schedules them for processing
        while handling memory management.
        """
        # Eagerly reserve memory and send to prefill worker
        self.reserve_memory_and_send_to_prefill()

        # Update the active decode batch
        self.check_for_completed_transfers()

        # Construct the batch to execute
        t0 = time.monotonic()
        batch_to_execute = self.batch_constructor.construct_batch()
        t1 = time.monotonic()
        batch_creation_time_s = t1 - t0

        # If the batch is empty, skip
        batch_size = batch_to_execute.batch_size
        if batch_size == 0:
            return

        # Schedule the batch
        t0 = time.monotonic()
        with Tracer(f"_schedule({batch_to_execute})"):
            self.schedule(batch_to_execute)
        t1 = time.monotonic()
        batch_execution_time_s = t1 - t0

        # Log batch metrics
        self.scheduler_logger.log_metrics(
            sch_config=self.scheduler_config,
            sch_output=batch_to_execute,
            paged_cache=self.paged_manager,
            batch_creation_time_s=batch_creation_time_s,
            batch_execution_time_s=batch_execution_time_s,
            num_pending_reqs=len(self.pending_reqs)
            + len(self.pending_prefill_requests),
            total_preemption_count=self.batch_constructor.total_preemption_count,
        )


def load_decode_scheduler(
    settings: Settings,
    pipeline: Pipeline[
        TextGenerationInputs[Union[TextContext, TextAndVisionContext]],
        TextGenerationOutput,
    ],
    pipeline_config: PipelineConfig,
    dispatcher_client: DispatcherClient,
) -> DecodeScheduler:
    # Create Scheduler Config.
    scheduler_config = TokenGenerationSchedulerConfig.from_pipeline_config(
        pipeline_config
    )

    # Retrieve Paged Manager
    paged_manager = get_paged_manager(pipeline)

    if paged_manager is None:
        raise RuntimeError(
            "A paged KV cache manager must be present to use the DecodeScheduler"
        )

    return DecodeScheduler(
        pipeline=pipeline,
        scheduler_config=scheduler_config,
        paged_manager=paged_manager,
        request_zmq_endpoint=settings.request_zmq_endpoint,
        response_zmq_endpoint=settings.response_zmq_endpoint,
        cancel_zmq_endpoint=settings.cancel_zmq_endpoint,
        dispatcher_client=dispatcher_client,
    )
