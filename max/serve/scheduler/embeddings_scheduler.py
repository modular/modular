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
from typing import Any

import zmq
from max.pipelines.core import (
    EmbeddingsGenerator,
    TextContext,
    msgpack_numpy_decoder,
)
from max.profiler import traced
from max.serve.queue.zmq_queue import ZmqPullSocket, ZmqPushSocket
from max.serve.scheduler import Scheduler
from max.serve.scheduler.queues import STOP_STREAM

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
        pipeline: EmbeddingsGenerator,
        request_zmq_endpoint: str,
        response_zmq_endpoint: str,
        cancel_zmq_endpoint: str,
        zmq_ctx: zmq.Context,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.pipeline = pipeline

        self.request_q = ZmqPullSocket[tuple[str, TextContext]](
            zmq_ctx=zmq_ctx,
            zmq_endpoint=request_zmq_endpoint,
            deserialize=msgpack_numpy_decoder(tuple[str, TextContext]),
        )
        self.response_q = ZmqPushSocket[Any](
            zmq_ctx=zmq_ctx, zmq_endpoint=response_zmq_endpoint
        )

    @traced
    def _create_batch_to_execute(self):
        max_batch_size_to_create = self.scheduler_config.max_batch_size

        batch = {}
        try:
            while max_batch_size_to_create > 0:
                req_id, data = self.request_q.get_nowait()
                batch[req_id] = data
                max_batch_size_to_create -= 1
        except queue.Empty:
            pass

        return batch

    def run_iteration(self) -> None:
        """The Scheduler loop that creates batches and schedules them on GPU"""
        batch_to_execute = self._create_batch_to_execute()
        if len(batch_to_execute) == 0:
            return

        self._schedule_encode(batch_to_execute)

    @traced
    def _handle_terminated_responses(
        self,
        batch_executed: dict[str, Any],
        batch_response: dict[str, Any],
    ) -> None:
        """Task that handles responses"""
        already_terminated = set()
        terminated = batch_executed.keys() - batch_response.keys()
        for req_id in terminated:
            if req_id in already_terminated:
                continue
            del batch_executed[req_id]
            batch_response[req_id] = STOP_STREAM
            already_terminated.add(req_id)

    @traced
    def _schedule_encode(self, batch_to_execute) -> None:
        # execute the batch
        batch_responses = self.pipeline.encode(batch_to_execute)
        # remove terminated requests from the batch
        self._handle_terminated_responses(batch_to_execute, batch_responses)
        # send the responses to the API process
        self.response_q.put_nowait([batch_responses])
