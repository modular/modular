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

from typing import TypeVar

import zmq
from max.interfaces import AudioGenerator, EmbeddingsGenerator, Pipeline
from max.nn.kv_cache import PagedKVCacheManager
from max.pipelines.core import TTSContext
from max.pipelines.lib import PipelineConfig, PipelineRole
from max.serve.config import Settings
from max.serve.kvcache_agent.dispatcher_client import DispatcherClient

from .audio_generation_scheduler import (
    AudioGenerationScheduler,
    AudioGenerationSchedulerConfig,
)
from .base import PrefillRequest, PrefillResponse, Scheduler
from .decode_scheduler import load_decode_scheduler
from .embeddings_scheduler import EmbeddingsScheduler, EmbeddingsSchedulerConfig
from .prefill_scheduler import load_prefill_scheduler
from .text_generation_scheduler import load_text_generation_scheduler

__all__ = [
    "AudioGenerationScheduler",
    "AudioGenerationSchedulerConfig",
    "EmbeddingsScheduler",
    "EmbeddingsSchedulerConfig",
    "PrefillRequest",
    "PrefillResponse",
    "Scheduler",
    "load_scheduler",
]

T = TypeVar("T")


def load_scheduler(
    pipeline: Pipeline | EmbeddingsGenerator | AudioGenerator[TTSContext],
    pipeline_config: PipelineConfig,
    settings: Settings,
    dispatcher_client: DispatcherClient | None = None,
) -> Scheduler:
    if isinstance(pipeline, EmbeddingsGenerator):
        embeddings_scheduler_config = EmbeddingsSchedulerConfig(
            max_batch_size=pipeline_config.max_batch_size
            if pipeline_config.max_batch_size is not None
            else 1
        )
        return EmbeddingsScheduler(
            scheduler_config=embeddings_scheduler_config,
            pipeline=pipeline,
            request_zmq_endpoint=settings.request_zmq_endpoint,
            response_zmq_endpoint=settings.response_zmq_endpoint,
            cancel_zmq_endpoint=settings.cancel_zmq_endpoint,
        )
    elif pipeline.__class__.__name__ == "AudioGeneratorPipeline":
        assert isinstance(pipeline, AudioGenerator)
        paged_manager = pipeline.speech_lm_pipeline._pipeline_model.kv_manager  # type: ignore
        assert isinstance(paged_manager, PagedKVCacheManager)

        assert pipeline_config.ce_delay_ms is not None
        assert pipeline_config.enable_prioritize_first_decode is not None

        token_gen_config = AudioGenerationSchedulerConfig(
            max_batch_size_tg=pipeline_config.max_batch_size,
            max_forward_steps_tg=pipeline_config.max_num_steps
            if pipeline_config.max_num_steps != -1
            else 1,
            max_batch_size_ce=pipeline_config.max_batch_size,
            target_tokens_per_batch_ce=pipeline_config.target_num_new_tokens,
            enable_chunked_prefill=pipeline_config.enable_chunked_prefill,
            enable_in_flight_batching=pipeline_config.enable_in_flight_batching,
            max_queue_size_tg=pipeline_config.max_queue_size_tg,
            min_batch_size_tg=pipeline_config.min_batch_size_tg,
            ce_delay_ms=pipeline_config.ce_delay_ms,
            enable_prioritize_first_decode=pipeline_config.enable_prioritize_first_decode,
        )

        return AudioGenerationScheduler(
            scheduler_config=token_gen_config,
            pipeline=pipeline,
            request_zmq_endpoint=settings.request_zmq_endpoint,
            response_zmq_endpoint=settings.response_zmq_endpoint,
            cancel_zmq_endpoint=settings.cancel_zmq_endpoint,
            paged_manager=paged_manager,
        )
    elif pipeline_config.pipeline_role == PipelineRole.PrefillAndDecode:
        assert isinstance(pipeline, Pipeline)
        return load_text_generation_scheduler(
            settings,
            pipeline,
            pipeline_config,
        )
    elif pipeline_config.pipeline_role == PipelineRole.DecodeOnly:
        assert isinstance(pipeline, Pipeline)
        if dispatcher_client is None:
            raise ValueError(
                "Dispatcher client is required for decode scheduler"
            )
        return load_decode_scheduler(
            settings,
            pipeline,
            pipeline_config,
            dispatcher_client=dispatcher_client,
        )
    elif pipeline_config.pipeline_role == PipelineRole.PrefillOnly:
        assert isinstance(pipeline, Pipeline)
        if dispatcher_client is None:
            raise ValueError(
                "Dispatcher client is required for prefill scheduler"
            )
        return load_prefill_scheduler(
            pipeline,
            pipeline_config,
            dispatcher_client=dispatcher_client,
        )
    else:
        raise ValueError(
            f"No scheduler support for pipeline_role ({pipeline_config.pipeline_role})."
        )
