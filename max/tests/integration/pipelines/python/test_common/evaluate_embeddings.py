# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Functions for running embeddings pipelines."""

import asyncio
import uuid
from collections.abc import Iterable
from typing import Any

from max.pipelines import EmbeddingsPipeline
from max.pipelines.core import PipelineTokenizer, TokenGeneratorRequest


def encode(
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
        batch_prompts: dict[str, str], batch_contexts: dict[str, Any]
    ) -> None:
        model_outputs = pipeline.encode(batch_contexts)
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
        curr_req_id = str(uuid.uuid4())
        context = await tokenizer.new_context(
            TokenGeneratorRequest(
                id="",
                index=0,
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
