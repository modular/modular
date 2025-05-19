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

import zmq
from max.pipelines.core import EmbeddingsGenerator, TokenGenerator
from max.pipelines.lib import PipelineRole
from max.serve.config import Settings
from max.serve.process_control import ProcessControl

from .base import Scheduler
from .config import TokenGeneratorSchedulerConfig
from .decode_scheduler import load_decode_scheduler
from .embeddings_scheduler import EmbeddingsScheduler, EmbeddingsSchedulerConfig
from .prefill_scheduler import load_prefill_scheduler
from .text_generation_scheduler import load_text_generation_scheduler
from .zmq_queue import ZmqPullSocket, ZmqPushSocket

__all__ = [
    "Scheduler",
    "ZmqPushSocket",
    "ZmqPullSocket",
    "load_scheduler",
    "TokenGeneratorSchedulerConfig",
    "EmbeddingsScheduler",
    "EmbeddingsSchedulerConfig",
]


def load_scheduler(
    pc: ProcessControl,
    pipeline: TokenGenerator | EmbeddingsGenerator,
    zmq_ctx: zmq.Context,
    settings: Settings,
    config: TokenGeneratorSchedulerConfig,
) -> Scheduler:
    if isinstance(pipeline, EmbeddingsGenerator):
        embeddings_scheduler_config = EmbeddingsSchedulerConfig(
            max_batch_size=config.token_generation.size,
        )
        return EmbeddingsScheduler(
            process_control=pc,
            scheduler_config=embeddings_scheduler_config,
            pipeline=pipeline,
            request_zmq_endpoint=settings.request_zmq_endpoint,
            response_zmq_endpoint=settings.response_zmq_endpoint,
            cancel_zmq_endpoint=settings.cancel_zmq_endpoint,
            zmq_ctx=zmq_ctx,
        )
    elif config.pipeline_role == PipelineRole.PrefillAndDecode:
        return load_text_generation_scheduler(
            zmq_ctx,
            settings,
            pipeline,
            pc,
            max_batch_size_tg=config.max_batch_size_tg,
            max_forward_steps_tg=config.max_forward_steps_tg,
            target_tokens_per_batch_tg=config.target_tokens_per_batch_tg,
            max_batch_size_ce=config.max_batch_size_ce,
            max_forward_steps_ce=config.max_forward_steps_ce,
            target_tokens_per_batch_ce=config.target_tokens_per_batch_ce,
            batch_timeout=config.batch_timeout,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_in_flight_batching=config.enable_in_flight_batching,
        )
    elif config.pipeline_role == PipelineRole.DecodeOnly:
        return load_decode_scheduler(
            zmq_ctx,
            settings,
            pipeline,
            pc,
            max_batch_size_tg=config.max_batch_size_tg,
            max_forward_steps_tg=config.max_forward_steps_tg,
        )
    elif config.pipeline_role == PipelineRole.PrefillOnly:
        return load_prefill_scheduler(
            zmq_ctx,
            settings,
            pipeline,
            pc=pc,
            max_batch_size_ce=config.max_batch_size_ce,
            target_tokens_per_batch_ce=config.target_tokens_per_batch_ce,
            batch_timeout=config.batch_timeout,
            enable_chunked_prefill=config.enable_chunked_prefill,
        )
    else:
        raise ValueError(
            f"no scheduler support for pipeline_role ({config.pipeline_role})."
        )
