# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

import contextlib
from collections.abc import AsyncGenerator
from typing import Any, cast

from max.interfaces import (
    AudioGenerationOutput,
    BaseContext,
    BaseContextType,
    EmbeddingsContext,
    EmbeddingsGenerationOutput,
    MAXPullQueue,
    MAXPushQueue,
    Pipeline,
    PipelineInputsType,
    PipelineOutput,
    PipelineOutputType,
    PixelGenerationContext,
    PixelGenerationInputs,
    RequestID,
    Scheduler,
    SchedulerResult,
    TextGenerationOutput,
)
from max.interfaces.generation import GenerationOutput
from max.kv_cache import PagedKVCacheManager
from max.pipelines.core import TextContext, TTSContext
from max.pipelines.lib import (
    EmbeddingsPipelineType,
    PipelineConfig,
    TextGenerationPipeline,
)
from max.pipelines.lib.audio_generator_pipeline import (
    AudioGeneratorPipelineType,
)
from max.pipelines.lib.pipeline_variants.pixel_generation import (
    PixelGenerationPipeline,
)
from max.serve.config import Settings
from max.serve.worker_interface import WorkerQueues

from .audio_generation_scheduler import (
    AudioGenerationScheduler,
    AudioGenerationSchedulerConfig,
)
from .base import CancelRequest, PrefillRequest, PrefillResponse
from .config import TokenGenerationSchedulerConfig
from .decode_scheduler import load_decode_scheduler
from .embeddings_scheduler import EmbeddingsScheduler, EmbeddingsSchedulerConfig
from .one_shot_scheduler import OneShotScheduler
from .prefill_scheduler import load_prefill_scheduler
from .text_generation_scheduler import load_text_generation_scheduler

__all__ = [
    "AudioGenerationScheduler",
    "AudioGenerationSchedulerConfig",
    "CancelRequest",
    "EmbeddingsScheduler",
    "EmbeddingsSchedulerConfig",
    "OneShotScheduler",
    "PrefillRequest",
    "PrefillResponse",
    "TokenGenerationSchedulerConfig",
    "load_scheduler",
]


def load_scheduler(
    pipeline: Pipeline[PipelineInputsType, PipelineOutputType],
    pipeline_config: PipelineConfig,
    settings: Settings,
    worker_queues: WorkerQueues[BaseContextType, PipelineOutputType],
) -> Scheduler:
    request_queue = worker_queues.request_queue
    response_queue = worker_queues.response_queue
    cancel_queue = worker_queues.cancel_queue

    if pipeline.__class__.__name__ == "PixelGenerationPipeline":
        pixel_pipeline = cast(PixelGenerationPipeline[Any], pipeline)

        def batch_constructor(
            context: PixelGenerationContext,
        ) -> PixelGenerationInputs[Any]:
            """Convert a single PixelGenerationContext into PixelGenerationInputs."""
            return PixelGenerationInputs(batch={context.request_id: context})

        return OneShotScheduler[
            PixelGenerationContext, PixelGenerationInputs[Any], GenerationOutput
        ](
            pipeline=pixel_pipeline,
            batch_constructor=batch_constructor,
            request_queue=cast(
                MAXPullQueue[PixelGenerationContext],
                request_queue,
            ),
            response_queue=cast(
                MAXPushQueue[
                    dict[RequestID, SchedulerResult[GenerationOutput]]
                ],
                response_queue,
            ),
            cancel_queue=cancel_queue,
            max_batch_size=pipeline_config.max_batch_size
            if pipeline_config.max_batch_size is not None
            else 1,
        )
    elif pipeline.__class__.__name__ == "EmbeddingsPipeline":
        embeddings_scheduler_config = EmbeddingsSchedulerConfig(
            max_batch_size=pipeline_config.max_batch_size
            if pipeline_config.max_batch_size is not None
            else 1
        )
        emb_pipeline = cast(EmbeddingsPipelineType, pipeline)
        return EmbeddingsScheduler(
            scheduler_config=embeddings_scheduler_config,
            pipeline=emb_pipeline,
            request_queue=cast(
                MAXPullQueue[EmbeddingsContext],
                request_queue,
            ),
            response_queue=cast(
                MAXPushQueue[
                    dict[RequestID, SchedulerResult[EmbeddingsGenerationOutput]]
                ],
                response_queue,
            ),
            cancel_queue=cancel_queue,
        )
    elif pipeline.__class__.__name__ == "AudioGeneratorPipeline":
        assert hasattr(pipeline, "kv_manager")
        kv_cache = pipeline.kv_manager

        assert pipeline_config.runtime.ce_delay_ms is not None
        assert (
            pipeline_config.runtime.enable_prioritize_first_decode is not None
        )
        assert pipeline_config.model.max_length is not None

        token_gen_config = AudioGenerationSchedulerConfig(
            max_batch_size=pipeline_config.max_batch_size,
            max_forward_steps_tg=pipeline_config.max_num_steps
            if pipeline_config.max_num_steps != -1
            else 1,
            max_seq_len=pipeline_config.model.max_length,
            target_tokens_per_batch_ce=pipeline_config.max_batch_input_tokens,
            enable_chunked_prefill=pipeline_config.enable_chunked_prefill,
            enable_in_flight_batching=pipeline_config.enable_in_flight_batching,
            max_queue_size_tg=pipeline_config.max_queue_size_tg,
            min_batch_size_tg=pipeline_config.min_batch_size_tg,
            ce_delay_ms=pipeline_config.runtime.ce_delay_ms,
            enable_prioritize_first_decode=pipeline_config.runtime.enable_prioritize_first_decode,
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
        )
        audio_pipeline = cast(AudioGeneratorPipelineType, pipeline)
        return AudioGenerationScheduler(
            scheduler_config=token_gen_config,
            pipeline=audio_pipeline,
            request_queue=cast(MAXPullQueue[TTSContext], request_queue),
            response_queue=cast(
                MAXPushQueue[
                    dict[RequestID, SchedulerResult[AudioGenerationOutput]]
                ],
                response_queue,
            ),
            cancel_queue=cancel_queue,
            kv_cache=kv_cache,
        )
    elif pipeline_config.pipeline_role == "prefill_and_decode":
        text_pipeline = cast(TextGenerationPipeline[TextContext], pipeline)
        return load_text_generation_scheduler(
            text_pipeline,
            pipeline_config,
            request_queue=cast(MAXPullQueue[TextContext], request_queue),
            response_queue=cast(
                MAXPushQueue[
                    dict[RequestID, SchedulerResult[TextGenerationOutput]]
                ],
                response_queue,
            ),
            cancel_queue=cancel_queue,
        )
    elif pipeline_config.pipeline_role == "decode_only":
        text_pipeline = cast(TextGenerationPipeline[TextContext], pipeline)
        return load_decode_scheduler(
            text_pipeline,
            pipeline_config,
            request_queue=cast(MAXPullQueue[TextContext], request_queue),
            response_queue=cast(
                MAXPushQueue[
                    dict[RequestID, SchedulerResult[TextGenerationOutput]]
                ],
                response_queue,
            ),
            cancel_queue=cancel_queue,
            settings=settings,
        )
    elif pipeline_config.pipeline_role == "prefill_only":
        text_pipeline = cast(TextGenerationPipeline[TextContext], pipeline)
        return load_prefill_scheduler(text_pipeline, pipeline_config, settings)
    else:
        raise ValueError(
            f"No scheduler support for pipeline_role ({pipeline_config.pipeline_role})."
        )
