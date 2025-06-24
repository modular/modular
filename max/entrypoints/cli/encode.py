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


"""Utilities for encoding text in the cli."""

import asyncio
import logging
import uuid
from typing import Optional

from max.pipelines import (
    PIPELINE_REGISTRY,
    PipelineConfig,
)
from max.pipelines.core import (
    EmbeddingsGenerator,
    EmbeddingsResponse,
    PipelineTask,
    PipelineTokenizer,
    TokenGeneratorRequest,
)

from .metrics import EmbeddingsMetrics

logger = logging.getLogger("max.entrypoints")

MODEL_NAME = "model"


async def _run_pipeline_encode(
    pipeline: EmbeddingsGenerator,
    tokenizer: PipelineTokenizer,
    prompt: str,
    metrics: Optional[EmbeddingsMetrics] = None,
) -> EmbeddingsResponse:
    req_id = str(uuid.uuid4())
    context = await tokenizer.new_context(
        TokenGeneratorRequest(
            id=req_id, index=0, prompt=prompt, model_name=MODEL_NAME
        )
    )
    pipeline_request = {req_id: context}

    if metrics:
        metrics.prompt_size = context.current_length
        metrics.signpost("begin_encoding")

    response = pipeline.encode(pipeline_request)

    if metrics:
        metrics.signpost("end_encoding")
    return response[req_id]


def pipeline_encode(
    pipeline_config: PipelineConfig,
    prompt: str,
    num_warmups: int = 0,
) -> None:
    # Run timed run & print results.
    with EmbeddingsMetrics(print_report=True) as metrics:
        tokenizer, pipeline = PIPELINE_REGISTRY.retrieve(
            pipeline_config, task=PipelineTask.EMBEDDINGS_GENERATION
        )
        assert isinstance(pipeline, EmbeddingsGenerator)

        if num_warmups > 0:
            logger.info("Running warmup")
            for _ in range(num_warmups):
                asyncio.run(
                    _run_pipeline_encode(
                        pipeline, tokenizer, prompt, metrics=None
                    )
                )

        # Run and print results.
        logger.info("Running model...")
        print("Encoding:", prompt)

        pipeline_output = asyncio.run(
            _run_pipeline_encode(pipeline, tokenizer, prompt, metrics=metrics)
        )
        print("Embeddings:", pipeline_output.embeddings)
