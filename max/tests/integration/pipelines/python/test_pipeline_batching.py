# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from typing import Literal
import pytest
from dataclasses import dataclass

from llama3 import (
    SupportedEncodings,
    SupportedVersions,
    InferenceConfig,
    Llama3,
)
from evaluate_llama import hf_hub_download


@dataclass(frozen=True)
class PipelineModelParams:
    name: Literal["tinyllama", "llama3_1"]
    encoding: SupportedEncodings
    max_length: int
    max_new_tokens: int = -1
    max_batch_size: int = 1
    version = SupportedVersions.llama3_1

    def __str__(self):
        return f"{self.name}-{self.encoding}-{self.max_length}-{self.max_batch_size}"


@pytest.fixture(scope="session")
def pipeline_model(testdata_directory, request):
    model_params: PipelineModelParams = request.param
    print(f"\nPipelineModel: {model_params}")
    model_encoding = model_params.encoding
    if model_params.name == "tinyllama":
        weight_path = testdata_directory / "tiny_llama.gguf"
    else:
        weights_repo_id = f"modularai/llama-{model_params.version}"
        weights_encoding_file = model_encoding.hf_model_name(
            model_params.version
        )
        weight_path = hf_hub_download(
            repo_id=weights_repo_id,
            filename=weights_encoding_file,
        )
        print(f"- Downloaded: {weight_path}")
    config = InferenceConfig(
        weight_path=weight_path,
        version=model_params.version,
        quantization_encoding=model_encoding,
        max_length=model_params.max_length,
        max_new_tokens=model_params.max_new_tokens,
        batch_size=model_params.max_batch_size,
    )
    print(
        f"- Using config: {config.version}, MaxLength={config.max_length},"
        f" MaxNewTokens={config.max_new_tokens}, BatchSize={config.batch_size}"
    )
    model = Llama3(config)
    return model


@pytest.mark.skip("Expensive")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pipeline_model",
    [
        PipelineModelParams("llama3_1", SupportedEncodings.q4_k, 128, -1, 2),
        PipelineModelParams("llama3_1", SupportedEncodings.q4_k, 128, -1, 4),
    ],
    ids=PipelineModelParams.__str__,
    indirect=True,
)
async def test_pipeline_static_batch_same_prompt_same_output(pipeline_model):
    """Execute a batch which matches the batch-size of the model
    Expects tokens to be generated for all contexts in lock-step
    All batches should complete at the same time.
    """
    prompt = "Repeat this sentence forever and forever."
    batch_size = pipeline_model.config.batch_size
    context_batch = {}
    for i in range(batch_size):
        context = await pipeline_model.new_context(prompt)
        batch_id = str(i)
        context_batch[batch_id] = context

    cur_tokens = len(context.tokens)
    max_tokens = context.max_tokens
    assert all(len(c.tokens) == cur_tokens for c in context_batch.values())
    assert all(c.max_tokens == max_tokens for c in context_batch.values())

    # Execute these batches until they are complete
    for _ in range(cur_tokens, max_tokens):
        response = await pipeline_model.next_token(context_batch)
        assert context_batch.keys() == response.keys()
        response_tokens = list(response.values())
        assert all(response_tokens[0] == t for t in response_tokens)
        # for id, next_token in response.items():
        #     print(id, next_token)

    # The last execution must complete all batches
    assert len(context_batch) == batch_size
    last = await pipeline_model.next_token(context_batch)
    assert not last


@pytest.mark.skip("ExpensiveAndFails")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pipeline_model",
    [
        PipelineModelParams("llama3_1", SupportedEncodings.q4_k, 128, -1, 2),
        PipelineModelParams("llama3_1", SupportedEncodings.q4_k, 128, -1, 4),
    ],
    ids=PipelineModelParams.__str__,
    indirect=True,
)
async def test_pipeline_static_batch_same_prompt_different_max_new_tokens(
    pipeline_model,
):
    """Execute a batch which matches the batch-size of the model
    HOWEVER, we set different max-new-tokens for each batch
    For N batches, Batch1 = MaxTokens / N, BatchN = MaxTokens
    We expect batches to complete one by one until the last one is done.
    """
    prompt = "Repeat this sentence forever and forever."
    batch_size = pipeline_model.config.batch_size
    print(batch_size)
    context_batch = {}
    for i in range(batch_size):
        max_new_tokens = int(
            (pipeline_model.config.max_length / batch_size) * (i + 1)
        )
        print(f"Batch: {i}, MaxNewTokens: {max_new_tokens}")
        context = await pipeline_model.new_context(prompt, max_new_tokens)
        print(f"Context: {i}, MaxTokens: {context.max_tokens}")
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
        response = await pipeline_model.next_token(context_batch)
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
    last = await pipeline_model.next_token(context_batch)
    assert not last


@pytest.fixture(scope="session")
def batch_sizes(request):
    return request.param


@pytest.mark.skip("Expensive")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pipeline_model, batch_sizes",
    [
        (
            PipelineModelParams(
                "llama3_1", SupportedEncodings.q4_k, 128, -1, 4
            ),
            [3, 1, 2, 4],
        ),
        (
            PipelineModelParams(
                "llama3_1", SupportedEncodings.q4_k, 128, -1, 8
            ),
            [4, 7, 1, 8],
        ),
    ],
    ids=lambda x: str(x),
    indirect=True,
)
async def test_pipeline_dynamic_batch_same_prompt_same_output(
    pipeline_model, batch_sizes
):
    """Execute a batch which matches the batch-size of the model
    Expects tokens to be generated for all contexts in lock-step
    All batches should complete at the same time.
    """
    prompt = "Repeat this sentence forever and forever."
    max_batch_size = pipeline_model.config.batch_size
    print(f"MaxBatchSize: {max_batch_size}")

    for batch_size in batch_sizes:
        assert batch_size > 0
        assert batch_size <= max_batch_size
        print(f"Batch: {batch_size} - Started")

        context_batch = {}
        for i in range(batch_size):
            context = await pipeline_model.new_context(prompt)
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
            response = await pipeline_model.next_token(context_batch)
            assert context_batch.keys() == response.keys()
            response_tokens = list(response.values())
            assert all(response_tokens[0] == t for t in response_tokens)
            # for id, next_token in response.items():
            #     print(id, next_token)

        # The last execution must complete all batches
        assert len(context_batch) == batch_size
        last = await pipeline_model.next_token(context_batch)
        assert not last

        # for batch_id, batch_context in context_batch.items():
        #     print(f"{batch_id}: {batch_context.decoded}")

        print(
            f"Batch: {batch_size} - Completed with"
            f" {len(context.tokens)} tokens."
        )
