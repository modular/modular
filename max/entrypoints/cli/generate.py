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
import logging
import uuid
from collections.abc import Iterable
from typing import Optional

import requests
from max.pipelines import (
    PIPELINE_REGISTRY,
    PipelineConfig,
)
from max.pipelines.core import (
    PipelineTokenizer,
    TokenGenerator,
    TokenGeneratorRequest,
)

from .metrics import TextGenerationMetrics

logger = logging.getLogger("max.entrypoints")

MODEL_NAME = "model"


async def stream_text_to_console(
    pipeline: TokenGenerator,
    tokenizer: PipelineTokenizer,
    prompt: str,
    images: Optional[list[bytes]],
    num_steps: int,
    metrics: Optional[TextGenerationMetrics] = None,
    print_tokens: bool = True,
) -> None:
    req_id = str(uuid.uuid4())
    context = await tokenizer.new_context(
        TokenGeneratorRequest(
            id=req_id,
            index=0,
            prompt=prompt,
            images=images,
            model_name=MODEL_NAME,
        )
    )
    pipeline_request = {req_id: context}
    if print_tokens:
        print(prompt, end="", flush=True)

    prompt_size = context.current_length
    if metrics:
        metrics.prompt_size = prompt_size
        metrics.signpost("begin_generation")

    try:
        first_token = True
        generate_again = True
        while generate_again:
            responses = pipeline.next_token(
                pipeline_request, num_steps=num_steps
            )

            for request_idx, response in responses.items():
                if response.is_done:
                    generate_again = False

                for text_response in response.tokens:
                    encoded_text = text_response.next_token
                    response_text = await tokenizer.decode(
                        context, encoded_text
                    )
                    if metrics:
                        if first_token:
                            first_token = False
                            metrics.signpost("first_token")
                        metrics.new_token()
                    if print_tokens:
                        print(response_text, end="", flush=True)

            # Yield to the event loop.  If at no other point (e.g.
            # tokenizer.decode which we await earlier does not yield to the
            # event loop), it will be at this point that we'll receive a
            # CancelledError if our future was canceled (e.g., we received a
            # SIGINT).
            await asyncio.sleep(0)

    finally:
        if metrics:
            metrics.signpost("end_generation")

        pipeline.release(context)

    if print_tokens:
        print()


def generate_text_for_pipeline(
    pipeline_config: PipelineConfig,
    prompt: str,
    image_urls: Iterable[str] = (),
    num_warmups: int = 0,
) -> None:
    # Run timed run & print results.
    with TextGenerationMetrics(print_report=True) as metrics:
        tokenizer, pipeline = PIPELINE_REGISTRY.retrieve(pipeline_config)
        assert isinstance(pipeline, TokenGenerator)
        if image_urls:
            logger.info("Downloading images")
            images = [requests.get(url).content for url in image_urls]
        else:
            images = None

        if num_warmups > 0:
            logger.info("Running warmup")
            for _ in range(num_warmups):
                asyncio.run(
                    stream_text_to_console(
                        pipeline,
                        tokenizer,
                        prompt,
                        images,
                        num_steps=pipeline_config.max_num_steps,
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
                num_steps=pipeline_config.max_num_steps,
                metrics=metrics,
                print_tokens=True,
            )
        )
