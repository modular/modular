# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import logging
from typing import Any

import hf_repo_lock
import pytest
from max.pipelines import (
    PIPELINE_REGISTRY,
    PipelineConfig,
    SupportedEncoding,
    TextGenerationPipeline,
    TextTokenizer,
)
from max.pipelines.core import TokenGeneratorRequest
from max.pipelines.lib import generate_local_model_path
from test_common.evaluate import PROMPTS, next_token_with_logits

REPO_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
REVISION = hf_repo_lock.revision_for_hf_repo(REPO_ID)

logger = logging.getLogger("max.pipelines")


@pytest.fixture(scope="session")
def pipeline_config() -> PipelineConfig:
    try:
        model_path = generate_local_model_path(REPO_ID, REVISION)
    except FileNotFoundError as e:
        logger.warning(f"Failed to generate local model path: {str(e)}")
        logger.warning(
            f"Falling back to repo_id: {REPO_ID} as config to PipelineConfig"
        )
        model_path = REPO_ID
    return PipelineConfig(
        model_path=model_path,
        quantization_encoding=SupportedEncoding.float32,
        max_batch_size=4,
    )


@pytest.fixture(scope="session")
def pipeline_tokenizer(pipeline_config: PipelineConfig) -> TextTokenizer:
    try:
        model_path = generate_local_model_path(REPO_ID, REVISION)
    except FileNotFoundError as e:
        logger.warning(f"Failed to generate local model path: {str(e)}")
        logger.warning(
            f"Falling back to repo_id: {REPO_ID} as config to PipelineConfig"
        )
        model_path = REPO_ID
    pipeline_config = PipelineConfig(
        model_path=model_path,
        quantization_encoding=SupportedEncoding.float32,
    )
    return TextTokenizer(
        pipeline_config.model_config.model_path,
        revision=pipeline_config.model_config.huggingface_model_revision,
        max_length=pipeline_config.max_length,
        max_new_tokens=pipeline_config.max_new_tokens,
        trust_remote_code=pipeline_config.model_config.trust_remote_code,
    )


@pytest.fixture(scope="session")
def pipeline(
    pipeline_config: PipelineConfig, pipeline_tokenizer: TextTokenizer
) -> TextGenerationPipeline:
    _, pipeline = PIPELINE_REGISTRY.retrieve(pipeline_config)
    assert isinstance(pipeline, TextGenerationPipeline)
    return pipeline


@pytest.mark.skip("Disabling tempoarily to update llama3 in a separate commit")
@pytest.mark.asyncio
async def test_pipeline_static_batch_same_prompt_same_output(
    pipeline, pipeline_tokenizer
):
    """Execute a batch which matches the batch-size of the model
    Expects tokens to be generated for all contexts in lock-step
    All batches should complete at the same time.

    This should be expected to run with both the naive cache and the continuous
    batching cache.

    """
    prompt = "Repeat this sentence forever and forever."
    batch_size = pipeline._pipeline_config.max_batch_size
    context_batch = {}
    for i in range(batch_size):
        context = await pipeline_tokenizer.new_context(
            TokenGeneratorRequest(
                id="", index=i, prompt=prompt, model_name="llama3"
            )
        )
        batch_id = str(i)
        context_batch[batch_id] = context

    # Execute these batches until they are complete
    for _ in range(context.current_length, context.max_length):
        response = pipeline.next_token(context_batch, num_steps=1)[0]
        assert context_batch.keys() == response.keys()
        response_tokens = list(response.values())
        assert all(response_tokens[0] == t for t in response_tokens)
        # for id, next_token in response.items():
        #     print(id, next_token)

    # The last execution must complete all batches
    assert len(context_batch) == batch_size
    last = pipeline.next_token(context_batch, num_steps=1)[0]
    assert not last

    # We should be resetting the cache
    for context in context_batch.values():
        pipeline.release(context)


@pytest.mark.skip("flaky")
@pytest.mark.asyncio
async def test_pipeline_static_batch_same_prompt_different_max_new_tokens(
    pipeline, pipeline_tokenizer
):
    """Execute a batch which matches the batch-size of the model
    HOWEVER, we set different max-new-tokens for each batch
    For N batches, Batch1 = MaxTokens / N, BatchN = MaxTokens
    We expect batches to complete one by one until the last one is done.

    This should not be expected to run with the naive/contiguous cache. As such, the only
    encodings we should expect this test to run with is tinyllama/fp32.
    """
    prompt = "Repeat this sentence forever and forever."
    batch_size = pipeline._pipeline_config.max_batch_size

    print(batch_size)
    context_batch = {}
    for i in range(batch_size):
        max_new_tokens = int(
            (pipeline._pipeline_config.max_length / batch_size) * (i + 1)
        )
        print(f"Batch: {i}, MaxNewTokens: {max_new_tokens}")
        context = await pipeline_tokenizer.new_context(
            TokenGeneratorRequest(
                id="",
                index=i,
                prompt=prompt,
                model_name="llama3",
                max_new_tokens=max_new_tokens,
            )
        )
        print(f"Context: {i}, MaxNewTokens: {context.max_new_tokens}")
        batch_id = str(i)
        context_batch[batch_id] = context

    max_tokens = max(c.max_length for c in context_batch.values())
    print(
        f"CurTokens: {context.current_length}, MaxTokensAcrossAllBatches: {max_tokens}"
    )

    # Execute batches until they are complete
    for i in range(context.current_length, max_tokens):
        batch_ids_with_lengths = {
            batch_id: c.current_length for batch_id, c in context_batch.items()
        }
        print(f"{i}-Input: {batch_ids_with_lengths}")
        response = pipeline.next_token(context_batch, num_steps=1)[0]
        print(f"{i}-Output: {response}")
        completed_batch_ids = context_batch.keys() - response.keys()
        for batch_id in completed_batch_ids:
            context = context_batch[batch_id]
            print(
                f"Completed {batch_id}, Tokens: {context.current_length}, Max:"
                f" {context.max_length}"
            )
            assert context.current_length > context.max_length
            del context_batch[batch_id]

    # The last execution must complete all batches
    # print(f"Remaining: {context_batch.keys()}")
    assert len(context_batch) == 1
    last = pipeline.next_token(context_batch, num_steps=1)[0]
    assert not last

    for context in context_batch.values():
        pipeline.release(context)


@pytest.fixture(scope="session")
def batch_sizes(request):
    return request.param


@pytest.mark.skip("Disabling tempoarily to update llama3 in a separate commit")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "batch_sizes",
    [
        ([3, 1, 2, 4],),
        ([4, 7, 1, 8],),
    ],
    ids=lambda x: str(x),
    indirect=True,
)
async def test_pipeline_dynamic_batch_same_prompt_same_output(
    pipeline, pipeline_tokenizer, batch_sizes
):
    """Execute a batch which matches the batch-size of the model
    Expects tokens to be generated for all contexts in lock-step
    All batches should complete at the same time.

    This should be expected to run for both naive and continuous
    batching.
    """
    prompt = "Repeat this sentence forever and forever."
    max_batch_size = pipeline._pipeline_config.max_batch_size
    print(f"MaxBatchSize: {max_batch_size}")

    for batch_size in batch_sizes:
        assert batch_size > 0
        assert batch_size <= max_batch_size
        print(f"Batch: {batch_size} - Started")

        context_batch = {}
        for i in range(batch_size):
            context = await pipeline_tokenizer.new_context(
                TokenGeneratorRequest(
                    id="", index=i, prompt=prompt, model_name="llama3"
                )
            )
            batch_id = str(i)
            context_batch[batch_id] = context

        max_tokens = context.max_length
        assert all(c.max_length == max_tokens for c in context_batch.values())
        print(
            f"Batch: {batch_size} - CurTokens: {context.current_length}, MaxTokens:"
            f" {max_tokens}"
        )

        # Execute these batches until they are complete
        for _ in range(context.current_length, max_tokens):
            response = pipeline.next_token(context_batch, num_steps=1)[0]
            assert context_batch.keys() == response.keys()
            response_tokens = list(response.values())
            assert all(response_tokens[0] == t for t in response_tokens)
            # for id, next_token in response.items():
            #     print(id, next_token)

        # The last execution must complete all batches
        assert len(context_batch) == batch_size
        last = pipeline.next_token(context_batch, num_steps=1)[0]
        assert not last

        # for batch_id, batch_context in context_batch.items():
        #     print(f"{batch_id}: {batch_context.decoded}")

        print(
            f"Batch: {batch_size} - Completed with"
            f" {context.current_length} tokens."
        )

        for context in context_batch.values():
            pipeline.release(context)


@pytest.mark.asyncio
@pytest.mark.skip("AITLIB-336: Bug that causes a mismatch in matmul dimensions")
async def test_pipeline_heterogeneous_batch_logits(
    pipeline, pipeline_tokenizer
):
    """Execute a batch with prompts with different lengths and validates the
    logits.

    This should only be expected to run with the continuous batching kv cache.
    As such, it should only work with tinyllama/fp32 on CPU.
    """

    llama3 = pipeline._pipeline_model
    prompt_a = PROMPTS[0]
    prompt_b = PROMPTS[1]
    prompt_c = PROMPTS[2]

    stored_logits: dict[str, list[Any]] = {"A": [], "B": [], "C": []}

    # Send in A for context encoding.
    context_a = await pipeline_tokenizer.new_context(
        TokenGeneratorRequest(
            id="", index=0, prompt=prompt_a, model_name="llama3"
        )
    )
    next_token_with_logits(llama3, {"A": context_a}, stored_logits)

    # Send in B for context encoding
    context_b = await pipeline_tokenizer.new_context(
        TokenGeneratorRequest(
            id="", index=1, prompt=prompt_b, model_name="llama3"
        )
    )
    next_token_with_logits(llama3, {"B": context_b}, stored_logits)

    # Send in both A and B for token generation
    next_token_with_logits(
        llama3, {"A": context_a, "B": context_b}, stored_logits
    )

    # Send in C for context encoding
    context_c = await pipeline_tokenizer.new_context(
        TokenGeneratorRequest(
            id="", index=2, prompt=prompt_c, model_name="llama3"
        )
    )
    next_token_with_logits(llama3, {"C": context_c}, stored_logits)

    # Send in both B and C for token generation
    next_token_with_logits(
        llama3, {"B": context_b, "C": context_c}, stored_logits
    )

    # Send in A, B, C out of order
    # This evaluates if the order of the batch can be mutated
    next_token_with_logits(
        llama3,
        {"C": context_c, "B": context_b, "A": context_a},
        stored_logits,
    )

    pipeline.release(context_a)
    pipeline.release(context_b)
    pipeline.release(context_c)
