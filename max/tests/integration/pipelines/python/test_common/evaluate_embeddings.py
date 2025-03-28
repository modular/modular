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
from max.pipelines.interfaces import PipelineTokenizer, TokenGeneratorRequest


def encode(
    pipeline: EmbeddingsPipeline,
    tokenizer: PipelineTokenizer,
    prompts: Iterable[str],
):
    """Runs the model for N steps on each prompt provide."""
    return asyncio.run(
        encode_async(
            pipeline,
            tokenizer,
            prompts=prompts,
        )
    )


async def encode_async(
    pipeline: EmbeddingsPipeline,
    tokenizer: PipelineTokenizer,
    prompts: Iterable[str],
) -> list[dict[str, Any]]:
    """Runs the model for each prompt provided."""
    results = []
    # Evaluate prompts individually (not batched).
    # TODO: add batched version of encode.
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
        values: dict[str, list[Any]] = {curr_req_id: []}

        # Set up model inputs
        context_batch = {"": context}
        model_outputs = pipeline.encode(context_batch)[""]
        results.append(
            {"prompt": prompt, "embeddings": model_outputs.embeddings}
        )
    return results
