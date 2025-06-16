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
from max.nn.kv_cache import PagedKVCacheManager
from max.pipelines.core import (
    AudioGenerator,
    EmbeddingsGenerator,
    TokenGenerator,
)
from max.pipelines.lib import PipelineRole
from max.serve.config import Settings
from max.serve.kvcache_agent.dispatcher_client import DispatcherClient
from max.serve.process_control import ProcessControl

from .audio_generation_scheduler import (
    AudioGenerationScheduler,
    AudioGenerationSchedulerConfig,
)
from .base import PrefillRequest, PrefillResponse, Scheduler
from .config import TokenGeneratorSchedulerConfig
from .decode_scheduler import load_decode_scheduler
from .embeddings_scheduler import EmbeddingsScheduler, EmbeddingsSchedulerConfig
from .prefill_scheduler import load_prefill_scheduler
from .text_generation_scheduler import load_text_generation_scheduler

__all__ = [
    "Scheduler",
    "load_scheduler",
    "EmbeddingsScheduler",
    "EmbeddingsSchedulerConfig",
    "TokenGeneratorSchedulerConfig",
    "AudioGenerationScheduler",
    "AudioGenerationSchedulerConfig",
    "PrefillRequest",
    "PrefillResponse",
]


def load_scheduler(
    pc: ProcessControl,
    pipeline: TokenGenerator | EmbeddingsGenerator | AudioGenerator,
    zmq_ctx: zmq.Context,
    settings: Settings,
    config: TokenGeneratorSchedulerConfig,
    dispatcher_client: DispatcherClient,
) -> Scheduler:
    if isinstance(pipeline, EmbeddingsGenerator):
        embeddings_scheduler_config = EmbeddingsSchedulerConfig(
            max_batch_size=config.token_generation.size
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
    elif pipeline.__class__.__name__ == "AudioGeneratorPipeline":
        assert isinstance(pipeline, AudioGenerator)
        paged_manager = pipeline.speech_lm_pipeline._pipeline_model.kv_manager  # type: ignore
        assert isinstance(paged_manager, PagedKVCacheManager)

        assert config.ce_delay_ms is not None
        assert config.enable_prioritize_first_decode is not None

        token_gen_config = AudioGenerationSchedulerConfig(
            max_batch_size_tg=config.max_batch_size_tg,
            max_forward_steps_tg=config.max_forward_steps_tg,
            target_tokens_per_batch_tg=config.target_tokens_per_batch_tg,
            max_batch_size_ce=config.max_batch_size_ce,
            max_forward_steps_ce=config.max_forward_steps_ce,
            target_tokens_per_batch_ce=config.target_tokens_per_batch_ce,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_in_flight_batching=config.enable_in_flight_batching,
            max_queue_size_tg=config.max_queue_size_tg,
            min_batch_size_tg=config.min_batch_size_tg,
            ce_delay_ms=config.ce_delay_ms,
            enable_prioritize_first_decode=config.enable_prioritize_first_decode,
        )

        return AudioGenerationScheduler(
            process_control=pc,
            scheduler_config=token_gen_config,
            pipeline=pipeline,
            request_zmq_endpoint=settings.request_zmq_endpoint,
            response_zmq_endpoint=settings.response_zmq_endpoint,
            cancel_zmq_endpoint=settings.cancel_zmq_endpoint,
            zmq_ctx=zmq_ctx,
            paged_manager=paged_manager,
        )
    elif config.pipeline_role == PipelineRole.PrefillAndDecode:
        assert isinstance(pipeline, TokenGenerator)
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
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_in_flight_batching=config.enable_in_flight_batching,
        )
    elif config.pipeline_role == PipelineRole.DecodeOnly:
        assert isinstance(pipeline, TokenGenerator)
        return load_decode_scheduler(
            zmq_ctx,
            settings,
            pipeline,
            pc,
            max_batch_size_tg=config.max_batch_size_tg,
            max_forward_steps_tg=config.max_forward_steps_tg,
            dispatcher_client=dispatcher_client,
        )
    elif config.pipeline_role == PipelineRole.PrefillOnly:
        assert isinstance(pipeline, TokenGenerator)
        return load_prefill_scheduler(
            zmq_ctx,
            settings,
            pipeline,
            pc=pc,
            max_batch_size_ce=config.max_batch_size_ce,
            target_tokens_per_batch_ce=config.target_tokens_per_batch_ce,
            enable_chunked_prefill=config.enable_chunked_prefill,
            dispatcher_client=dispatcher_client,
        )
    else:
        raise ValueError(
            f"no scheduler support for pipeline_role ({config.pipeline_role})."
        )
