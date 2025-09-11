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

from typing import Any, Union

from max.interfaces import (
    AudioGenerator,
    AudioGeneratorOutput,
    EmbeddingsGenerator,
    EmbeddingsOutput,
    Pipeline,
    RequestID,
    Scheduler,
    SchedulerResult,
    TextGenerationOutput,
)
from max.nn.kv_cache import PagedKVCacheManager
from max.pipelines.core import TextAndVisionContext, TextContext, TTSContext
from max.pipelines.lib import PipelineConfig, PipelineRole
from max.serve.config import Settings
from max.serve.queue.zmq_queue import create_zmq_push_pull_queues

from .audio_generation_scheduler import (
    AudioGenerationScheduler,
    AudioGenerationSchedulerConfig,
)
from .base import PrefillRequest, PrefillResponse
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
    "load_scheduler",
]


def load_scheduler(
    pipeline: Pipeline[Any, Any]
    | EmbeddingsGenerator[TextContext]
    | AudioGenerator[TTSContext],
    pipeline_config: PipelineConfig,
    settings: Settings,
) -> Scheduler:
    if isinstance(pipeline, EmbeddingsGenerator):
        embeddings_scheduler_config = EmbeddingsSchedulerConfig(
            max_batch_size=pipeline_config.max_batch_size
            if pipeline_config.max_batch_size is not None
            else 1
        )

        _, eb_request_pull_queue = create_zmq_push_pull_queues(
            endpoint=settings.request_zmq_endpoint,
            payload_type=tuple[RequestID, TextContext],
        )

        eb_response_push_queue, _ = create_zmq_push_pull_queues(
            endpoint=settings.response_zmq_endpoint,
            payload_type=dict[RequestID, SchedulerResult[EmbeddingsOutput]],
        )

        _, eb_cancel_pull_queue = create_zmq_push_pull_queues(
            endpoint=settings.cancel_zmq_endpoint, payload_type=list[RequestID]
        )
        return EmbeddingsScheduler(
            scheduler_config=embeddings_scheduler_config,
            pipeline=pipeline,
            request_queue=eb_request_pull_queue,
            response_queue=eb_response_push_queue,
            cancel_queue=eb_cancel_pull_queue,
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
            target_tokens_per_batch_ce=pipeline_config.prefill_chunk_size,
            enable_chunked_prefill=pipeline_config.enable_chunked_prefill,
            enable_in_flight_batching=pipeline_config.enable_in_flight_batching,
            max_queue_size_tg=pipeline_config.max_queue_size_tg,
            min_batch_size_tg=pipeline_config.min_batch_size_tg,
            ce_delay_ms=pipeline_config.ce_delay_ms,
            enable_prioritize_first_decode=pipeline_config.enable_prioritize_first_decode,
            data_parallel_degree=pipeline_config.model_config.data_parallel_degree,
        )

        _, ag_request_pull_queue = create_zmq_push_pull_queues(
            endpoint=settings.request_zmq_endpoint,
            payload_type=tuple[RequestID, TTSContext],
        )

        ag_response_push_queue, _ = create_zmq_push_pull_queues(
            endpoint=settings.response_zmq_endpoint,
            payload_type=dict[RequestID, SchedulerResult[AudioGeneratorOutput]],
        )

        _, ag_cancel_pull_queue = create_zmq_push_pull_queues(
            endpoint=settings.cancel_zmq_endpoint, payload_type=list[RequestID]
        )

        return AudioGenerationScheduler(
            scheduler_config=token_gen_config,
            pipeline=pipeline,
            request_queue=ag_request_pull_queue,
            response_queue=ag_response_push_queue,
            cancel_queue=ag_cancel_pull_queue,
            paged_manager=paged_manager,
        )
    elif pipeline_config.pipeline_role == PipelineRole.PrefillAndDecode:
        assert isinstance(pipeline, Pipeline)
        _, pd_request_pull_queue = create_zmq_push_pull_queues(
            endpoint=settings.request_zmq_endpoint,
            payload_type=tuple[
                RequestID, Union[TextContext, TextAndVisionContext]
            ],
        )

        pd_response_push_queue, _ = create_zmq_push_pull_queues(
            endpoint=settings.response_zmq_endpoint,
            payload_type=dict[RequestID, SchedulerResult[TextGenerationOutput]],
        )

        _, pd_cancel_pull_queue = create_zmq_push_pull_queues(
            endpoint=settings.cancel_zmq_endpoint, payload_type=list[RequestID]
        )
        return load_text_generation_scheduler(
            pipeline,
            pipeline_config,
            request_queue=pd_request_pull_queue,
            response_queue=pd_response_push_queue,
            cancel_queue=pd_cancel_pull_queue,
        )
    elif pipeline_config.pipeline_role == PipelineRole.DecodeOnly:
        assert isinstance(pipeline, Pipeline)
        _, ds_request_pull_queue = create_zmq_push_pull_queues(
            endpoint=settings.request_zmq_endpoint,
            payload_type=tuple[
                RequestID, Union[TextContext, TextAndVisionContext]
            ],
        )

        ds_response_push_queue, _ = create_zmq_push_pull_queues(
            endpoint=settings.response_zmq_endpoint,
            payload_type=dict[RequestID, SchedulerResult[TextGenerationOutput]],
        )

        _, ds_cancel_pull_queue = create_zmq_push_pull_queues(
            endpoint=settings.cancel_zmq_endpoint, payload_type=list[RequestID]
        )

        return load_decode_scheduler(
            pipeline,
            pipeline_config,
            request_queue=ds_request_pull_queue,
            response_queue=ds_response_push_queue,
            cancel_queue=ds_cancel_pull_queue,
            settings=settings,
        )
    elif pipeline_config.pipeline_role == PipelineRole.PrefillOnly:
        assert isinstance(pipeline, Pipeline)
        return load_prefill_scheduler(pipeline, pipeline_config, settings)
    else:
        raise ValueError(
            f"No scheduler support for pipeline_role ({pipeline_config.pipeline_role})."
        )
