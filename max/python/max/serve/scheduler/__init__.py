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

from typing import cast

from max.interfaces import (
    EmbeddingsContext,
    MAXPullQueue,
    Pipeline,
    PipelineInputsType,
    PipelineOutputType,
    Scheduler,
    TextGenerationInputs,
    TextGenerationOutput,
)
from max.nn.kv_cache import PagedKVCacheManager
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    EmbeddingsPipelineType,
    PipelineConfig,
    PipelineRole,
)
from max.pipelines.lib.audio_generator_pipeline import (
    AudioGeneratorPipelineType,
)
from max.serve.config import Settings
from max.serve.scheduler.queues import SchedulerZmqConfigs

from .audio_generation_scheduler import (
    AudioGenerationScheduler,
    AudioGenerationSchedulerConfig,
)
from .base import CancelRequest, PrefillRequest, PrefillResponse
from .decode_scheduler import load_decode_scheduler
from .embeddings_scheduler import EmbeddingsScheduler, EmbeddingsSchedulerConfig
from .prefill_scheduler import load_prefill_scheduler
from .text_generation_scheduler import load_text_generation_scheduler

__all__ = [
    "AudioGenerationScheduler",
    "AudioGenerationSchedulerConfig",
    "CancelRequest",
    "EmbeddingsScheduler",
    "EmbeddingsSchedulerConfig",
    "PrefillRequest",
    "PrefillResponse",
    "load_scheduler",
]


def load_scheduler(
    pipeline: Pipeline[PipelineInputsType, PipelineOutputType],
    pipeline_config: PipelineConfig,
    settings: Settings,
    scheduler_zmq_configs: SchedulerZmqConfigs,
) -> Scheduler:
    request_queue, response_queue, cancel_queue = (
        scheduler_zmq_configs.model_worker_queues()
    )

    if pipeline.__class__.__name__ == "EmbeddingsPipeline":
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
            response_queue=response_queue,
            cancel_queue=cancel_queue,
            offload_queue_draining=pipeline_config.experimental_background_queue,
        )
    elif pipeline.__class__.__name__ == "AudioGeneratorPipeline":
        assert hasattr(pipeline, "speech_lm_pipeline")
        paged_manager = pipeline.speech_lm_pipeline._pipeline_model.kv_manager
        assert isinstance(paged_manager, PagedKVCacheManager)

        assert pipeline_config.ce_delay_ms is not None
        assert pipeline_config.enable_prioritize_first_decode is not None

        token_gen_config = AudioGenerationSchedulerConfig(
            max_batch_size_tg=pipeline_config.max_batch_size,
            max_forward_steps_tg=pipeline_config.max_num_steps
            if pipeline_config.max_num_steps != -1
            else 1,
            max_batch_size_ce=pipeline_config.max_batch_size,
            target_tokens_per_batch_ce=pipeline_config.prefill_chunk_size,
            enable_chunked_prefill=pipeline_config.enable_chunked_prefill,
            enable_in_flight_batching=pipeline_config.enable_in_flight_batching,
            max_queue_size_tg=pipeline_config.max_queue_size_tg,
            min_batch_size_tg=pipeline_config.min_batch_size_tg,
            ce_delay_ms=pipeline_config.ce_delay_ms,
            enable_prioritize_first_decode=pipeline_config.enable_prioritize_first_decode,
            data_parallel_degree=pipeline_config.model_config.data_parallel_degree,
        )
        audio_pipeline = cast(AudioGeneratorPipelineType, pipeline)

        return AudioGenerationScheduler(
            scheduler_config=token_gen_config,
            pipeline=audio_pipeline,
            request_queue=request_queue,
            response_queue=response_queue,
            cancel_queue=cancel_queue,
            paged_manager=paged_manager,
            offload_queue_draining=pipeline_config.experimental_background_queue,
        )
    elif pipeline_config.pipeline_role == PipelineRole.PrefillAndDecode:
        assert isinstance(pipeline, Pipeline)
        text_pipeline = cast(
            Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
            pipeline,
        )
        return load_text_generation_scheduler(
            text_pipeline,
            pipeline_config,
            request_queue=cast(MAXPullQueue[TextContext], request_queue),
            response_queue=response_queue,
            cancel_queue=cancel_queue,
        )
    elif pipeline_config.pipeline_role == PipelineRole.DecodeOnly:
        assert isinstance(pipeline, Pipeline)
        text_pipeline = cast(
            Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
            pipeline,
        )
        return load_decode_scheduler(
            text_pipeline,
            pipeline_config,
            request_queue=cast(MAXPullQueue[TextContext], request_queue),
            response_queue=response_queue,
            cancel_queue=cancel_queue,
            settings=settings,
        )
    elif pipeline_config.pipeline_role == PipelineRole.PrefillOnly:
        assert isinstance(pipeline, Pipeline)
        text_pipeline = cast(
            Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
            pipeline,
        )
        return load_prefill_scheduler(text_pipeline, pipeline_config, settings)
    else:
        raise ValueError(
            f"No scheduler support for pipeline_role ({pipeline_config.pipeline_role})."
        )
