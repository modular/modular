# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from typing import Any

import pytest
from max.interfaces import (
    SamplingParams,
    TextGenerationInputs,
    TextGenerationRequest,
)
from max.pipelines import (
    PIPELINE_REGISTRY,
    PipelineConfig,
    SupportedEncoding,
    TextGenerationPipeline,
    TextTokenizer,
)
from test_common.evaluate import next_token_with_logits
from test_common.test_data import DEFAULT_PROMPTS


@pytest.fixture
def pipeline_config(smollm2_135m_local_path: str) -> PipelineConfig:
    return PipelineConfig(
        model_path=smollm2_135m_local_path,
        quantization_encoding=SupportedEncoding.bfloat16,
        max_batch_size=4,
        allow_safetensors_weights_fp32_bf6_bidirectional_cast=True,
    )


@pytest.fixture
def pipeline_tokenizer(pipeline_config: PipelineConfig) -> TextTokenizer:
    return TextTokenizer(
        pipeline_config.model_config.model_path,
        revision=pipeline_config.model_config.huggingface_model_revision,
        max_length=pipeline_config.max_length,
        max_new_tokens=pipeline_config.max_new_tokens,
        trust_remote_code=pipeline_config.model_config.trust_remote_code,
    )


@pytest.fixture
def pipeline(pipeline_config: PipelineConfig) -> TextGenerationPipeline:
    _, pipeline = PIPELINE_REGISTRY.retrieve(pipeline_config)
    assert isinstance(pipeline, TextGenerationPipeline)
    return pipeline


# Please also fix the typing TODO when re-enabling!
@pytest.mark.skip("Disabling temporarily to update llama3 in a separate commit")
@pytest.mark.asyncio
async def test_pipeline_static_batch_same_prompt_same_output(
    pipeline: TextGenerationPipeline, pipeline_tokenizer: TextTokenizer
) -> None:
    """Execute a batch which matches the batch-size of the model
    Expects tokens to be generated for all contexts in lock-step
    All batches should complete at the same time.

    This should be expected to run with both the naive cache and the continuous
    batching cache.

    """
    prompt = "Repeat this sentence forever and forever."
    batch_size = pipeline._pipeline_config.max_batch_size
    assert batch_size is not None
    context_batch = {}
    for i in range(batch_size):
        context = await pipeline_tokenizer.new_context(
            TextGenerationRequest(
                request_id="", prompt=prompt, model_name="llama3"
            )
        )
        batch_id = str(i)
        context_batch[batch_id] = context

    # Execute these batches until they are complete
    for _ in range(context.current_length, context.max_length):
        inputs = TextGenerationInputs(batches=[context_batch], num_steps=1)
        # TODO: Fix typing when this test is re-enabled!
        response = pipeline.execute(inputs)[0]  # type: ignore
        assert context_batch.keys() == response.keys()  # type: ignore
        response_tokens = list(response.values())  # type: ignore
        assert all(response_tokens[0] == t for t in response_tokens)

    # The last execution must complete all batches
    assert len(context_batch) == batch_size
    inputs = TextGenerationInputs(batches=[context_batch], num_steps=1)
    last = pipeline.execute(inputs)[0]  # type: ignore
    assert not last

    # We should be resetting the cache
    for context in context_batch.values():
        pipeline.release(context.request_id)


@pytest.mark.skip("flaky")
@pytest.mark.asyncio
async def test_pipeline_static_batch_same_prompt_different_max_new_tokens(
    pipeline: TextGenerationPipeline, pipeline_tokenizer: TextTokenizer
) -> None:
    """Execute a batch which matches the batch-size of the model
    HOWEVER, we set different max-new-tokens for each batch
    For N batches, Batch1 = MaxTokens / N, BatchN = MaxTokens
    We expect batches to complete one by one until the last one is done.

    This should not be expected to run with the naive/contiguous cache. As such, the only
    encodings we should expect this test to run with is tinyllama/fp32.
    """
    prompt = "Repeat this sentence forever and forever."
    batch_size = pipeline._pipeline_config.max_batch_size
    assert batch_size is not None
    max_length = pipeline._pipeline_config.max_length
    assert max_length is not None

    context_batch = {}
    for i in range(batch_size):
        max_new_tokens = int((max_length / batch_size) * (i + 1))
        sampling_params = SamplingParams(max_new_tokens=max_new_tokens)
        context = await pipeline_tokenizer.new_context(
            TextGenerationRequest(
                request_id="",
                prompt=prompt,
                model_name="llama3",
                sampling_params=sampling_params,
            )
        )
        batch_id = str(i)
        context_batch[batch_id] = context

    max_tokens = max(c.max_length for c in context_batch.values())

    # Execute batches until they are complete
    for _ in range(context.current_length, max_tokens):
        batch_ids_with_lengths = {
            batch_id: c.current_length for batch_id, c in context_batch.items()
        }
        inputs = TextGenerationInputs(batches=[context_batch], num_steps=1)
        response = pipeline.execute(inputs)[0]  # type: ignore
        completed_batch_ids = context_batch.keys() - response.keys()  # type: ignore
        for batch_id in completed_batch_ids:
            context = context_batch[batch_id]
            assert context.current_length > context.max_length
            del context_batch[batch_id]

    # The last execution must complete all batches
    assert len(context_batch) == 1
    inputs = TextGenerationInputs(batches=[context_batch], num_steps=1)
    last = pipeline.execute(inputs)[0]  # type: ignore
    assert not last

    for context in context_batch.values():
        pipeline.release(context.request_id)


@pytest.mark.skip("Disabling temporarily to update llama3 in a separate commit")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "batch_sizes",
    [
        ([3, 1, 2, 4],),
        ([4, 7, 1, 8],),
    ],
    ids=lambda x: str(x),
)
async def test_pipeline_dynamic_batch_same_prompt_same_output(
    pipeline: TextGenerationPipeline,
    pipeline_tokenizer: TextTokenizer,
    batch_sizes: list[int],
) -> None:
    """Execute a batch which matches the batch-size of the model
    Expects tokens to be generated for all contexts in lock-step
    All batches should complete at the same time.

    This should be expected to run for both naive and continuous
    batching.
    """
    prompt = "Repeat this sentence forever and forever."
    max_batch_size = pipeline._pipeline_config.max_batch_size
    assert max_batch_size is not None

    for batch_size in batch_sizes:
        assert batch_size > 0
        assert batch_size <= max_batch_size

        context_batch = {}
        for i in range(batch_size):
            context = await pipeline_tokenizer.new_context(
                TextGenerationRequest(
                    request_id="", prompt=prompt, model_name="llama3"
                )
            )
            batch_id = str(i)
            context_batch[batch_id] = context

        max_tokens = context.max_length
        assert all(c.max_length == max_tokens for c in context_batch.values())

        # Execute these batches until they are complete
        for _ in range(context.current_length, max_tokens):
            inputs = TextGenerationInputs(batches=[context_batch], num_steps=1)
            response = pipeline.execute(inputs)[0]  # type: ignore
            assert context_batch.keys() == response.keys()  # type: ignore
            response_tokens = list(response.values())  # type: ignore
            assert all(response_tokens[0] == t for t in response_tokens)

        # The last execution must complete all batches
        assert len(context_batch) == batch_size
        inputs = TextGenerationInputs(batches=[context_batch], num_steps=1)
        last = pipeline.execute(inputs)[0]  # type: ignore
        assert not last

        for context in context_batch.values():
            pipeline.release(context.request_id)


@pytest.mark.skip(
    "AITLIB-336: Attempted to release request ID  but it is not claimed"
)
@pytest.mark.asyncio
async def test_pipeline_heterogeneous_batch_logits(
    pipeline: TextGenerationPipeline, pipeline_tokenizer: TextTokenizer
) -> None:
    """Execute a batch with prompts with different lengths and validates the
    logits.

    This should only be expected to run with the continuous batching kv cache.
    As such, it should only work with tinyllama/fp32 on CPU.
    """

    llama3 = pipeline._pipeline_model
    prompt_a = DEFAULT_PROMPTS[0]
    prompt_b = DEFAULT_PROMPTS[1]
    prompt_c = DEFAULT_PROMPTS[2]

    stored_logits: dict[str, list[Any]] = {"A": [], "B": [], "C": []}

    # Send in A for context encoding.
    context_a = await pipeline_tokenizer.new_context(
        TextGenerationRequest(
            request_id="", prompt=prompt_a, model_name="llama3"
        )
    )
    next_token_with_logits(llama3, {"A": context_a}, stored_logits)

    # Send in B for context encoding
    context_b = await pipeline_tokenizer.new_context(
        TextGenerationRequest(
            request_id="", prompt=prompt_b, model_name="llama3"
        )
    )
    next_token_with_logits(llama3, {"B": context_b}, stored_logits)

    # Send in both A and B for token generation
    next_token_with_logits(
        llama3, {"A": context_a, "B": context_b}, stored_logits
    )

    # Send in C for context encoding
    context_c = await pipeline_tokenizer.new_context(
        TextGenerationRequest(
            request_id="", prompt=prompt_c, model_name="llama3"
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

    pipeline.release(context_a.request_id)
    pipeline.release(context_b.request_id)
    pipeline.release(context_c.request_id)
