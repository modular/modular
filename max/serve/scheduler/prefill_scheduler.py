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
import time
from dataclasses import dataclass
from typing import Optional

from max.nn.kv_cache import PagedKVCacheManager
from max.pipelines.core import TokenGenerator
from max.serve.process_control import ProcessControl

from .base import Scheduler

logger = logging.getLogger("max.serve")


@dataclass
class PrefillSchedulerConfig:
    """Prefill Specific Scheduler Config."""

    max_batch_size_ce: int
    """The maximum number of requests that can be in the context encoding batch."""

    target_tokens_per_batch_ce: Optional[int]
    """The target total number of tokens to encode in the context encoding batch."""

    batch_timeout: Optional[float]
    """The maximum amount of time to wait before creating a context encoding batch."""

    enable_chunked_prefill: bool = True
    """Enables chunked prefill, where the scheduler splits requests into chunks to ensure
    each batch contains exactly `target_tokens_per_batch_ce` tokens."""

    def __post_init__(self) -> None:
        if (
            self.enable_chunked_prefill
            and self.target_tokens_per_batch_ce is None
        ):
            msg = "Need set `target_tokens_per_batch_ce` for the scheduler to enable chunked prefill."
            raise ValueError(msg)


class PrefillScheduler(Scheduler):
    def __init__(
        self,
        process_control: ProcessControl,
        pipeline: TokenGenerator,
        scheduler_config: PrefillSchedulerConfig,
        paged_manager: PagedKVCacheManager,
        *,
        prefill_zmq_endpoint: str,
        decode_zmq_endpoint: str,
    ):
        self.pc = process_control
        self.pipeline = pipeline
        self.scheduler_config = PrefillSchedulerConfig

    def run(self) -> None:
        while True:
            # Indicate that the process is still alive.
            self.pc.beat()
            time.sleep(5)
