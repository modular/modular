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

"""Utilities for generating text in the cli."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import uuid
from collections.abc import Iterable
from typing import Any

import requests
from max.interfaces import (
    InputContext,
    LogitsProcessor,
    Pipeline,
    PipelineTokenizer,
    ProcessorInputs,
    SamplingParams,
    TextGenerationRequest,
)
from max.pipelines import (
    PIPELINE_REGISTRY,
    GenerateMixin,
    PipelineConfig,
    TextTokenizer,
)

from .metrics import TextGenerationMetrics

logger = logging.getLogger("max.entrypoints")

MODEL_NAME = "model"


class TrackMetrics:
    def __init__(self, metrics: TextGenerationMetrics):
        self.metrics = metrics
        self.first_token = True

    def __call__(self, inputs: ProcessorInputs) -> None:
        if self.first_token:
            self.first_token = False
            self.metrics.signpost("first_token")
            self.metrics.prompt_size = inputs.context.current_length
        self.metrics.new_token()


async def stream_text_to_console(
    pipeline: Pipeline[Any, Any],
    tokenizer: PipelineTokenizer[InputContext, Any, TextGenerationRequest],
    prompt: str,
    images: list[bytes] | None,
    sampling_params: SamplingParams,
    metrics: TextGenerationMetrics | None = None,
    print_tokens: bool = True,
) -> None:
    assert isinstance(tokenizer, TextTokenizer)
    logits_processors: list[LogitsProcessor] = []
    if metrics:
        logits_processors.append(TrackMetrics(metrics))

    sampling_params = dataclasses.replace(
        sampling_params, logits_processors=logits_processors
    )

    request = TextGenerationRequest(
        request_id=str(uuid.uuid4()),
        prompt=prompt,
        images=images,
        model_name=MODEL_NAME,
        sampling_params=sampling_params,
    )

    if metrics:
        metrics.signpost("begin_generation")
    try:
        assert isinstance(pipeline, GenerateMixin)
        async for outputs in pipeline.generate_async(request):
            if print_tokens:
                decoded = tokenizer.delegate.decode(outputs[0].tokens)
                print(decoded, end="", flush=True)
    finally:
        if metrics:
            metrics.signpost("end_generation")


def generate_text_for_pipeline(
    pipeline_config: PipelineConfig,
    sampling_params: SamplingParams,
    prompt: str,
    image_urls: Iterable[str] = (),
    num_warmups: int = 0,
) -> None:
    # Run timed run & print results.
    with TextGenerationMetrics(print_report=True) as metrics:
        tokenizer, pipeline = PIPELINE_REGISTRY.retrieve(pipeline_config)
        assert isinstance(pipeline, Pipeline)
        if image_urls:
            logger.info("Downloading images")
            images = [requests.get(url).content for url in image_urls]
        else:
            images = None

        if num_warmups > 0:
            logger.info("Running warmup")
            warmup_params = dataclasses.replace(
                sampling_params, max_new_tokens=num_warmups
            )
            asyncio.run(
                stream_text_to_console(
                    pipeline,
                    tokenizer,
                    prompt,
                    images,
                    sampling_params=warmup_params,
                    metrics=None,
                    print_tokens=False,
                )
            )

        # Run and print results.
        logger.info("Beginning text generation")
        asyncio.run(
            stream_text_to_console(
                pipeline,
                tokenizer,
                prompt,
                images,
                sampling_params=sampling_params,
                metrics=metrics,
                print_tokens=True,
            )
        )
