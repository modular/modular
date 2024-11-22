# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from dataclasses import dataclass
from typing import Any, Literal

import pytest
from evaluate_llama import SupportedTestModels
from llama3.llama3_token_gen import Llama3TokenGenerator
from max.pipelines import PipelineConfig, SupportedEncoding, TextTokenizer
from max.pipelines.interfaces import TokenGeneratorRequest
from max.pipelines.kv_cache import KVCacheStrategy
from test_common.evaluate import PROMPTS, next_token_with_logits


@dataclass(frozen=True)
class PipelineModelParams:
    name: Literal["tinyllama", "llama3_1"]
    encoding: SupportedEncoding
    max_length: int
    max_new_tokens: int = -1
    max_batch_size: int = 1

    """Whether to include a print hook. This is generally for debugging
    purposes, but it's also helping to avoid a segfault in the heterogeneous
    test."""

    def __str__(self):
        return f"{self.name}-{self.encoding}-{self.max_length}-{self.max_batch_size}"


@pytest.fixture(scope="session")
def pipeline_config(testdata_directory, request):
    model_params: PipelineModelParams = request.param
    print(f"\nPipelineModel: {model_params}")
    encoding = model_params.encoding
    test_model = SupportedTestModels.get(model_params.name, encoding)

    if encoding in [SupportedEncoding.float32, SupportedEncoding.bfloat16]:
        print("using continuous batching caching strategy")
        cache_strategy = KVCacheStrategy.CONTINUOUS
    else:
        print("using naive caching strategy")
        cache_strategy = KVCacheStrategy.NAIVE

    return test_model.build_config(
        testdata_directory=testdata_directory,
        max_length=model_params.max_length,
        max_new_tokens=model_params.max_new_tokens,
        max_cache_batch_size=model_params.max_batch_size,
        cache_strategy=cache_strategy,
    )


@pytest.fixture(scope="session")
def pipeline_tokenizer(pipeline_config):
    return TextTokenizer(pipeline_config)


@pytest.fixture(scope="session")
def pipeline_model(
    pipeline_config: PipelineConfig, pipeline_tokenizer: TextTokenizer
):
    return Llama3TokenGenerator(
        pipeline_config,
        pipeline_tokenizer.delegate.eos_token_id,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pipeline_config",
    [
        PipelineModelParams("llama3_1", SupportedEncoding.q4_k, 10, -1, 2),
        PipelineModelParams("llama3_1", SupportedEncoding.q4_k, 10, -1, 4),
    ],
    ids=PipelineModelParams.__str__,
    indirect=True,
)
async def test_pipeline_static_batch_same_prompt_same_output(
    pipeline_model, pipeline_tokenizer
):
    """Execute a batch which matches the batch-size of the model
    Expects tokens to be generated for all contexts in lock-step
    All batches should complete at the same time.

    This should be expected to run with both the naive cache and the continuous
    batching cache.

    """
    prompt = "Repeat this sentence forever and forever."
    batch_size = pipeline_model.config.max_cache_batch_size
    context_batch = {}
    for i in range(batch_size):
        context = await pipeline_tokenizer.new_context(
            TokenGeneratorRequest(
                id="", index=i, prompt=prompt, model_name="llama3"
            )
        )
        batch_id = str(i)
        context_batch[batch_id] = context

    cur_tokens = len(context.tokens)
    max_tokens = context.max_tokens
    assert all(len(c.tokens) == cur_tokens for c in context_batch.values())
    assert all(c.max_tokens == max_tokens for c in context_batch.values())

    # Execute these batches until they are complete
    for _ in range(cur_tokens, max_tokens):
        response = pipeline_model.next_token(context_batch)[0]
        assert context_batch.keys() == response.keys()
        response_tokens = list(response.values())
        assert all(response_tokens[0] == t for t in response_tokens)
        # for id, next_token in response.items():
        #     print(id, next_token)

    # The last execution must complete all batches
    assert len(context_batch) == batch_size
    last = pipeline_model.next_token(context_batch)[0]
    assert not last

    # We should be resetting the cache
    for context in context_batch.values():
        pipeline_model.release(context)


@pytest.mark.skip("flaky")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pipeline_config",
    [
        PipelineModelParams("tinyllama", SupportedEncoding.float32, 128, -1, 2),
        PipelineModelParams("tinyllama", SupportedEncoding.float32, 128, -1, 4),
    ],
    ids=PipelineModelParams.__str__,
    indirect=True,
)
async def test_pipeline_static_batch_same_prompt_different_max_new_tokens(
    pipeline_model, pipeline_tokenizer
):
    """Execute a batch which matches the batch-size of the model
    HOWEVER, we set different max-new-tokens for each batch
    For N batches, Batch1 = MaxTokens / N, BatchN = MaxTokens
    We expect batches to complete one by one until the last one is done.

    This should not be expected to run with the naive/contiguous cache. As such, the only
    encodings we should expect this test to run with is tinyllama/fp32.
    """
    prompt = "Repeat this sentence forever and forever."
    batch_size = pipeline_model.config.max_cache_batch_size

    print(batch_size)
    context_batch = {}
    for i in range(batch_size):
        max_new_tokens = int(
            (pipeline_model.config.max_length / batch_size) * (i + 1)
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

    cur_tokens = len(context.tokens)
    assert all(len(c.tokens) == cur_tokens for c in context_batch.values())
    max_tokens = max(c.max_tokens for c in context_batch.values())
    print(f"CurTokens: {cur_tokens}, MaxTokensAcrossAllBatches: {max_tokens}")

    # Execute batches until they are complete
    for i in range(cur_tokens, max_tokens):
        batch_ids_with_lengths = {
            batch_id: len(c.tokens) for batch_id, c in context_batch.items()
        }
        print(f"{i}-Input: {batch_ids_with_lengths}")
        response = pipeline_model.next_token(context_batch)[0]
        print(f"{i}-Output: {response}")
        completed_batch_ids = context_batch.keys() - response.keys()
        for batch_id in completed_batch_ids:
            context = context_batch[batch_id]
            print(
                f"Completed {batch_id}, Tokens: {len(context.tokens)}, Max:"
                f" {context.max_tokens}"
            )
            assert len(context.tokens) > context.max_tokens
            del context_batch[batch_id]

    # The last execution must complete all batches
    # print(f"Remaining: {context_batch.keys()}")
    assert len(context_batch) == 1
    last = pipeline_model.next_token(context_batch)[0]
    assert not last

    for context in context_batch.values():
        pipeline_model.release(context)


@pytest.fixture(scope="session")
def batch_sizes(request):
    return request.param


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pipeline_config, batch_sizes",
    [
        (
            PipelineModelParams("llama3_1", SupportedEncoding.q4_k, 12, -1, 4),
            [3, 1, 2, 4],
        ),
        (
            PipelineModelParams("llama3_1", SupportedEncoding.q4_k, 12, -1, 8),
            [4, 7, 1, 8],
        ),
    ],
    ids=lambda x: str(x),
    indirect=True,
)
async def test_pipeline_dynamic_batch_same_prompt_same_output(
    pipeline_model, pipeline_tokenizer, batch_sizes
):
    """Execute a batch which matches the batch-size of the model
    Expects tokens to be generated for all contexts in lock-step
    All batches should complete at the same time.

    This should be expected to run for both naive and continuous
    batching.
    """
    prompt = "Repeat this sentence forever and forever."
    max_batch_size = pipeline_model.config.max_cache_batch_size
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

        cur_tokens = len(context.tokens)
        max_tokens = context.max_tokens
        assert all(len(c.tokens) == cur_tokens for c in context_batch.values())
        assert all(c.max_tokens == max_tokens for c in context_batch.values())
        print(
            f"Batch: {batch_size} - CurTokens: {cur_tokens}, MaxTokens:"
            f" {max_tokens}"
        )

        # Execute these batches until they are complete
        for _ in range(cur_tokens, max_tokens):
            response = pipeline_model.next_token(context_batch)[0]
            assert context_batch.keys() == response.keys()
            response_tokens = list(response.values())
            assert all(response_tokens[0] == t for t in response_tokens)
            # for id, next_token in response.items():
            #     print(id, next_token)

        # The last execution must complete all batches
        assert len(context_batch) == batch_size
        last = pipeline_model.next_token(context_batch)[0]
        assert not last

        # for batch_id, batch_context in context_batch.items():
        #     print(f"{batch_id}: {batch_context.decoded}")

        print(
            f"Batch: {batch_size} - Completed with"
            f" {len(context.tokens)} tokens."
        )

        for context in context_batch.values():
            pipeline_model.release(context)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pipeline_config",
    [
        PipelineModelParams(
            "tinyllama",
            SupportedEncoding.float32,
            512,
            10,
            4,
        ),
    ],
    ids=PipelineModelParams.__str__,
    indirect=True,
)
async def test_pipeline_heterogeneous_batch_logits(
    pipeline_model, pipeline_tokenizer
):
    """Execute a batch with prompts with different lengths and validates the
    logits.

    This should only be expected to run with the continuous batching kv cache.
    As such, it should only work with tinyllama/fp32 on CPU.
    """

    llama3 = pipeline_model.model
    kv_manager = pipeline_model._kv_manager
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

    pipeline_model.release(context_a)
    pipeline_model.release(context_b)
    pipeline_model.release(context_c)
