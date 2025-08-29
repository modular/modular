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
    EmbeddingsGenerator,
    EmbeddingsOutput,
    SchedulerResult,
    msgpack_numpy_decoder,
    msgpack_numpy_encoder,
)
from max.pipelines.core import TextContext
from max.profiler import traced
from max.serve.queue.zmq_queue import ZmqPullSocket, ZmqPushSocket
from max.serve.scheduler import Scheduler

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
    ) -> None:
        self.scheduler_config = scheduler_config
        self.pipeline = pipeline

        self.request_q = ZmqPullSocket[tuple[str, TextContext]](
            zmq_endpoint=request_zmq_endpoint,
            deserialize=msgpack_numpy_decoder(tuple[str, TextContext]),
        )
        self.response_q = ZmqPushSocket[
            dict[str, SchedulerResult[EmbeddingsOutput]]
        ](
            zmq_endpoint=response_zmq_endpoint,
            serialize=msgpack_numpy_encoder(),
        )

    @traced
    def _create_batch_to_execute(self) -> dict[str, TextContext]:
        max_batch_size_to_create = self.scheduler_config.max_batch_size

        batch: dict[str, TextContext] = {}
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
        batch_executed: dict[str, TextContext],
        batch_response: dict[str, SchedulerResult[EmbeddingsOutput]],
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
        self, batch_to_execute: dict[str, TextContext]
    ) -> None:
        # execute the batch
        batch_responses = self.pipeline.encode(batch_to_execute)
        # remove terminated requests from the batch
        self._handle_terminated_responses(batch_to_execute, batch_responses)
        # send the responses to the API process
        self.response_q.put_nowait(
            {
                request_id: SchedulerResult.create(response)
                for request_id, response in batch_responses.items()
            }
        )
