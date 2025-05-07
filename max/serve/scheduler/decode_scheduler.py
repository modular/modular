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

import time
from collections.abc import Mapping
from dataclasses import dataclass

from max.nn.kv_cache import PagedKVCacheManager
from max.pipelines.core import TokenGenerator
from max.serve.process_control import ProcessControl

from .base import Scheduler
from .zmq_queue import ZmqPullSocket, ZmqSocket


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
        queues: Mapping[str, ZmqSocket],
        paged_manager: PagedKVCacheManager,
    ):
        # Initialize Pipeline and Config
        self.scheduler_config = DecodeSchedulerConfig
        self.pipeline = pipeline

        # Multiprocessing resources.
        self.pc = process_control

        # Check that all queues are provided.
        if "REQUEST" not in queues:
            raise RuntimeError(
                "REQUEST queue must be provided to Decode Scheduler."
            )

        if "RESPONSE" not in queues:
            raise RuntimeError(
                "RESPONSE queue must be provided to Decode Scheduler."
            )

        if "CANCEL" not in queues:
            raise RuntimeError(
                "CANCEL queue must be provided to Decode Scheduler."
            )

        # TODO: Initialize ZmqSocket as Client/Server

        # Initialize Queues
        self.request_queue = ZmqPullSocket(queues["REQUEST"])
        self.response_q = queues["RESPONSE"]
        self.cancel_q = queues["CANCEL"]

    def run(self) -> None:
        while True:
            # Indicate that the process is still alive.
            self.pc.beat()
            time.sleep(5)
