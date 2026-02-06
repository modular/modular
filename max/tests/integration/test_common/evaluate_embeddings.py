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
"""Functions for running embeddings pipelines."""

import asyncio
from collections.abc import Iterable
from typing import Any

from max.interfaces import (
    EmbeddingsGenerationInputs,
    PipelineTokenizer,
    RequestID,
    TextGenerationRequest,
)
from max.pipelines import EmbeddingsPipeline


def encode(  # noqa: ANN201
    pipeline: EmbeddingsPipeline,
    tokenizer: PipelineTokenizer,
    prompts: Iterable[str],
    batch_size: int = 1,
):
    """Runs the model for N steps on each prompt provide."""
    return asyncio.run(
        encode_async(
            pipeline, tokenizer, prompts=prompts, batch_size=batch_size
        )
    )


async def encode_async(
    pipeline: EmbeddingsPipeline,
    tokenizer: PipelineTokenizer,
    prompts: Iterable[str],
    batch_size: int,
) -> list[dict[str, Any]]:
    """Runs the model for each prompt provided."""

    results: list[dict[str, Any]] = []

    def _encode_batch(
        batch_prompts: dict[RequestID, str],
        batch_contexts: dict[RequestID, Any],
    ) -> None:
        model_outputs = pipeline.execute(
            EmbeddingsGenerationInputs([batch_contexts])
        )
        for req_id, prompt in batch_prompts.items():
            results.append(
                {
                    "prompt": prompt,
                    "embeddings": model_outputs[req_id].embeddings,
                }
            )

    # Evaluate prompts.
    batch_prompts = {}
    batch_contexts = {}
    for prompt in prompts:
        curr_req_id = RequestID()
        context = await tokenizer.new_context(
            TextGenerationRequest(
                request_id=curr_req_id,
                prompt=prompt,
                model_name=type(pipeline).__name__,
            )
        )
        # Set up model inputs
        batch_prompts[curr_req_id] = prompt
        batch_contexts[curr_req_id] = context
        if len(batch_contexts) == batch_size:
            _encode_batch(batch_prompts, batch_contexts)
            batch_prompts.clear()
            batch_contexts.clear()
    if batch_contexts:
        _encode_batch(batch_prompts, batch_contexts)

    return results
