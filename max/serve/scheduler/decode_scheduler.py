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
import time
from dataclasses import dataclass

import zmq
from max.nn.kv_cache import PagedKVCacheManager
from max.pipelines.core import InputContext, TextResponse, TokenGenerator
from max.serve.process_control import ProcessControl

from .base import Scheduler
from .zmq_queue import ZmqPullSocket, ZmqPushSocket

logger = logging.getLogger("max.serve")


@dataclass
class DecodeSchedulerConfig:
    """Decode Specific Scheduler Config."""

    max_batch_size_tg: int
    """The maximum number of requests that can be in the token generation batch."""

    max_forward_steps_tg: int
    """The number of tokens to generate for each request in the token generation iteration."""


class DecodeScheduler(Scheduler):
    def __init__(
        self,
        process_control: ProcessControl,
        pipeline: TokenGenerator,
        scheduler_config: DecodeSchedulerConfig,
        paged_manager: PagedKVCacheManager,
        *,
        request_zmq_endpoint: str,
        response_zmq_endpoint: str,
        cancel_zmq_endpoint: str,
        prefill_zmq_endpoint: str,
        decode_zmq_endpoint: str,
        zmq_ctx: zmq.Context,
    ):
        # Initialize Pipeline and Config
        self.scheduler_config = DecodeSchedulerConfig
        self.pipeline = pipeline

        # Multiprocessing resources.
        self.pc = process_control

        # Initialize Queues
        self.request_pull_socket = ZmqPullSocket[tuple[str, InputContext]](
            zmq_ctx, zmq_endpoint=request_zmq_endpoint
        )
        self.response_push_socket = ZmqPushSocket[tuple[str, TextResponse]](
            zmq_ctx=zmq_ctx, zmq_endpoint=response_zmq_endpoint
        )
        self.cancel_pull_socket = ZmqPullSocket[tuple[str, InputContext]](
            zmq_ctx=zmq_ctx, zmq_endpoint=cancel_zmq_endpoint
        )

        # Initialize Queues for Disaggregation
        self.decode_pull_socket = ZmqPullSocket[tuple[str, InputContext]](
            zmq_ctx=zmq_ctx, zmq_endpoint=decode_zmq_endpoint
        )
        self.prefill_push_socket = ZmqPushSocket[tuple[str, InputContext]](
            zmq_ctx=zmq_ctx, zmq_endpoint=prefill_zmq_endpoint
        )

    def run(self) -> None:
        while True:
            # Indicate that the process is still alive.
            self.pc.beat()

            # Try and Receive from the request queue and send to Prefill
            try:
                new_request = self.request_pull_socket.get_nowait()
                logger.debug(f"sending from the decode_node: {new_request}")
                self.prefill_push_socket.put_nowait(new_request)
            except queue.Empty:
                logger.debug("nothing in the request queue.")

            try:
                new_decode = self.decode_pull_socket.get_nowait()
                logger.debug(
                    f"received new request on the decode node: {new_decode}"
                )
                logger.debug("sending back to prefill.")
                self.prefill_push_socket.put_nowait(new_decode)
                logger.debug("sent from the decode node.")

            except queue.Empty:
                logger.debug("nothing in the decode queue.")

            time.sleep(5)
