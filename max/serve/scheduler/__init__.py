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

from typing import Optional

import zmq
from max.nn.kv_cache import PagedKVCacheManager
from max.pipelines.core import TokenGenerator
from max.pipelines.lib import PipelineRole
from max.pipelines.lib.pipeline import KVCacheMixin, TextGenerationPipeline
from max.serve.config import Settings
from max.serve.process_control import ProcessControl

from .base import Scheduler
from .config import TokenGeneratorSchedulerConfig
from .decode_scheduler import load_decode_scheduler
from .prefill_scheduler import load_prefill_scheduler
from .text_generation_scheduler import load_text_generation_scheduler
from .zmq_queue import ZmqPullSocket, ZmqPushSocket

__all__ = [
    "Scheduler",
    "ZmqPushSocket",
    "ZmqPullSocket",
    "load_scheduler",
    "TokenGeneratorSchedulerConfig",
]


def load_scheduler(
    zmq_ctx: zmq.Context,
    settings: Settings,
    pipeline: TokenGenerator,
    pipeline_role: PipelineRole,
    pc: ProcessControl,
    max_batch_size_tg: int,
    max_forward_steps_tg: int,
    target_tokens_per_batch_tg: Optional[int],
    max_batch_size_ce: int,
    max_forward_steps_ce: int,
    target_tokens_per_batch_ce: Optional[int],
    batch_timeout: Optional[float],
    enable_chunked_prefill: bool = True,
    enable_in_flight_batching: bool = False,
) -> Scheduler:
    if pipeline_role == PipelineRole.PrefillAndDecode:
        return load_text_generation_scheduler(
            zmq_ctx,
            settings,
            pipeline,
            pc,
            max_batch_size_tg,
            max_forward_steps_tg,
            target_tokens_per_batch_tg,
            max_batch_size_ce,
            max_forward_steps_ce,
            target_tokens_per_batch_ce,
            batch_timeout,
            enable_chunked_prefill,
            enable_in_flight_batching,
        )
    elif pipeline_role == PipelineRole.DecodeOnly:
        return load_decode_scheduler(
            zmq_ctx,
            settings,
            pipeline,
            pc,
            max_batch_size_tg,
            max_forward_steps_tg,
        )
    elif pipeline_role == PipelineRole.PrefillOnly:
        return load_prefill_scheduler(
            zmq_ctx,
            settings,
            pipeline,
            pc=pc,
            max_batch_size_ce=max_batch_size_ce,
            target_tokens_per_batch_ce=target_tokens_per_batch_ce,
            batch_timeout=batch_timeout,
            enable_chunked_prefill=enable_chunked_prefill,
        )
    else:
        raise ValueError(
            f"no scheduler support for pipeline_role ({pipeline_role})."
        )
