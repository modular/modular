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
from typing import Optional, Union

import uvloop
from max.pipelines import PIPELINE_REGISTRY, PipelineConfig, PipelineTask
from max.pipelines.kv_cache import KVCacheStrategy
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


def serve_pipeline(
    pipeline_config: PipelineConfig,
    performance_fake: str = "none",
    profile: bool = False,
    batch_timeout: float = 0.0,
    model_name: Union[str, None] = None,
    failure_percentage: Optional[int] = None,
    experimental_enable_kvcache_agent: bool = False,
):
    # Initialize settings
    settings = Settings(MAX_SERVE_USE_HEARTBEAT=False)
    settings.experimental_enable_kvcache_agent = (
        experimental_enable_kvcache_agent
    )

    # TODO: This is a workaround to support embeddings generation until the
    # changes to tie pipelines to tasks is complete. This will be removed.
    pipeline_task = PipelineTask.TEXT_GENERATION
    if (
        pipeline_config.model_config.model_path
        == "sentence-transformers/all-mpnet-base-v2"
    ):
        pipeline_task = PipelineTask.EMBEDDINGS_GENERATION

    if performance_fake == "none":
        logger.info(
            f"Starting server using {pipeline_config.model_config.model_path}"
        )
        # Load tokenizer and pipeline from PIPELINE_REGISTRY.
        tokenizer, pipeline_factory = PIPELINE_REGISTRY.retrieve_factory(
            pipeline_config,
            task=pipeline_task,
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

        pipeline_config.model_config.kv_cache_config.cache_strategy = (
            KVCacheStrategy.CONTINUOUS
        )

    # Load batch config.
    batch_config = batch_config_from_pipeline_config(
        pipeline_config=pipeline_config,
        pipeline_task=pipeline_task,
        batch_timeout=batch_timeout,
    )

    # If explicit model name is not provided, set to model_path.
    if model_name is None:
        model_name = pipeline_config.model_config.model_path
        assert model_name is not None

    pipeline_settings = ServingTokenGeneratorSettings(
        model_name=model_name,
        model_factory=pipeline_factory,
        pipeline_config=batch_config,
        tokenizer=tokenizer,
    )

    # Intialize and serve webserver.
    app = fastapi_app(
        settings,
        pipeline_settings,
    )
    config = fastapi_config(app=app, server_settings=settings)

    server = Server(config)
    uvloop.run(server.serve())
