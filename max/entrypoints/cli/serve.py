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


"""Utilities for serving cli."""

import functools
import logging
import os
import signal
from typing import Optional, Union

import uvloop
from max.nn.kv_cache import KVCacheStrategy
from max.pipelines import (
    PIPELINE_REGISTRY,
    AudioGenerationConfig,
    PipelineConfig,
)
from max.pipelines.core import PipelineTask
from max.profiler import Tracer
from max.serve.api_server import (
    ServingTokenGeneratorSettings,
    fastapi_app,
    fastapi_config,
)
from max.serve.config import Settings
from max.serve.pipelines.llm import batch_config_from_pipeline_config
from max.serve.pipelines.performance_fake import (
    PerformanceFakingPipelineTokenizer,
    get_performance_fake,
)
from transformers import AutoTokenizer
from uvicorn import Server

logger = logging.getLogger(__name__)


def sigterm_handler(sig, frame):
    # We handle SIGINT gracefully, so piggyback on that
    logger.info("Got SIGTERM, terminating...")
    os.kill(os.getpid(), signal.SIGINT)


def serve_pipeline(
    pipeline_config: PipelineConfig,
    performance_fake: str = "none",
    profile: bool = False,
    model_name: Union[str, None] = None,
    failure_percentage: Optional[int] = None,
    experimental_enable_kvcache_agent: bool = False,
    port: Optional[int] = None,
    pipeline_task: PipelineTask = PipelineTask.TEXT_GENERATION,
):
    # Initialize settings
    settings = Settings(MAX_SERVE_USE_HEARTBEAT=False)

    if port is not None:
        settings.port = port

    settings.experimental_enable_kvcache_agent = (
        experimental_enable_kvcache_agent
    )

    override_architecture: Optional[str] = None
    # TODO: This is a workaround to support embeddings generation until the
    # changes to tie pipelines to tasks is complete. This will be removed.
    if (
        pipeline_config.model_config.model_path
        == "sentence-transformers/all-mpnet-base-v2"
    ):
        pipeline_task = PipelineTask.EMBEDDINGS_GENERATION

    # Use the audio decoder architecture for the audio generation pipeline.
    if pipeline_task == PipelineTask.AUDIO_GENERATION:
        assert isinstance(pipeline_config, AudioGenerationConfig)
        override_architecture = pipeline_config.audio_decoder

    if performance_fake == "none":
        logger.info(
            f"Starting server using {pipeline_config.model_config.model_path}"
        )
        # Load tokenizer and pipeline from PIPELINE_REGISTRY.
        tokenizer, pipeline_factory = PIPELINE_REGISTRY.retrieve_factory(
            pipeline_config,
            task=pipeline_task,
            override_architecture=override_architecture,
        )
    else:
        logger.info(
            f"Starting server using performance fake {performance_fake}."
        )
        tokenizer = PerformanceFakingPipelineTokenizer(
            AutoTokenizer.from_pretrained(
                pipeline_config.model_config.model_path
            )
        )
        pipeline_factory = functools.partial(
            get_performance_fake,
            performance_fake,  # type: ignore
            failure_percentage,
        )

        # TODO(AITLIB-320): Figure out a way to avoid monkey patching PipelineConfig
        # here.
        pipeline_config.model_config.kv_cache_config.cache_strategy = (
            KVCacheStrategy.CONTINUOUS
        )

    # Load batch config.
    batch_config = batch_config_from_pipeline_config(
        pipeline_config=pipeline_config, pipeline_task=pipeline_task
    )

    # If explicit model name is not provided, set to model_path.
    if model_name is None:
        model_name = pipeline_config.model_config.model_path

    pipeline_settings = ServingTokenGeneratorSettings(
        model_name=model_name,
        model_factory=pipeline_factory,
        pipeline_config=batch_config,
        tokenizer=tokenizer,
        pipeline_task=pipeline_task,
    )

    # Initialize and serve webserver.
    app = fastapi_app(settings, pipeline_settings)
    config = fastapi_config(app=app, server_settings=settings)

    signal.signal(signal.SIGTERM, sigterm_handler)

    server = Server(config)
    with Tracer("openai_compatible_frontend_server"):
        uvloop.run(server.serve())
