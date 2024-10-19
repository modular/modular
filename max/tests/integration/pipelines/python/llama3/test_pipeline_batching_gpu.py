# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from dataclasses import dataclass
from typing import Literal

import pytest
from evaluate_llama import (
    PROMPTS,
    NumpyDecoder,
    SupportedTestModels,
    compare_values,
    find_runtime_path,
    next_token_with_logits,
)
from llama3.config import SupportedEncodings, SupportedVersions
from llama3.llama3 import Llama3
from nn.kv_cache import KVCacheStrategy


@dataclass(frozen=True)
class PipelineModelParams:
    name: Literal["tinyllama", "llama3_1"]
    encoding: SupportedEncodings
    max_length: int
    max_new_tokens: int = -1
    max_batch_size: int = 1
    version: SupportedVersions = SupportedVersions.llama3_1

    """Whether to include a print hook. This is generally for debugging
    purposes, but it's also helping to avoid a segfault in the heterogeneous
    test."""

    def __str__(self):
        return f"{self.name}-{self.encoding}-{self.max_length}-{self.max_batch_size}"


@pytest.fixture(scope="session")
def pipeline_model(testdata_directory, request):
    model_params: PipelineModelParams = request.param
    print(f"\nPipelineModel: {model_params}")
    encoding = model_params.encoding
    test_model = SupportedTestModels.get(
        model_params.name, encoding, strict=False
    )

    if encoding in [SupportedEncodings.float32, SupportedEncodings.bfloat16]:
        cache_strategy = KVCacheStrategy.CONTINUOUS
    else:
        cache_strategy = KVCacheStrategy.NAIVE

    config = test_model.build_config(
        testdata_directory=testdata_directory,
        max_length=model_params.max_length,
        max_new_tokens=model_params.max_new_tokens,
        max_cache_batch_size=model_params.max_batch_size,
        cache_strategy=cache_strategy,
        pad_to_multiple_of=2,
    )

    print(
        f"- Using config: {config.version}, MaxLength={config.max_length},"
        f" MaxNewTokens={config.max_new_tokens},"
        f" BatchSize={config.max_cache_batch_size}"
    )
    model = Llama3(config)

    return model


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pipeline_model",
    [
        PipelineModelParams(
            "tinyllama",
            SupportedEncodings.bfloat16,
            512,
            10,
            4,
        ),
    ],
    ids=PipelineModelParams.__str__,
    indirect=True,
)
async def test_pipeline_heterogeneous_batch_logits(
    pipeline_model, testdata_directory
):
    """Execute a batch with prompts with different lengths and validates the
    logits.
    """
    golden_data_path = find_runtime_path(
        SupportedTestModels.TINY_LLAMA_BF16.golden_data_fname(),
        testdata_directory,
    )
    expected_results = NumpyDecoder().decode(golden_data_path.read_text())

    prompt_a = PROMPTS[0]
    prompt_b = PROMPTS[1]
    prompt_c = PROMPTS[2]

    stored_logits = {"A": [], "B": [], "C": []}

    # Send in A for context encoding.
    context_a = await pipeline_model.new_context(prompt_a)
    next_token_with_logits(pipeline_model, {"A": context_a}, stored_logits)

    # Send in B for context encoding
    context_b = await pipeline_model.new_context(prompt_b)
    next_token_with_logits(pipeline_model, {"B": context_b}, stored_logits)

    # Send in both A and B for token generation
    next_token_with_logits(
        pipeline_model, {"A": context_a, "B": context_b}, stored_logits
    )

    # Send in C for context encoding
    context_c = await pipeline_model.new_context(prompt_c)
    next_token_with_logits(pipeline_model, {"C": context_c}, stored_logits)

    # Send in both B and C for token generation
    next_token_with_logits(
        pipeline_model, {"B": context_b, "C": context_c}, stored_logits
    )

    compare_values(
        [
            {"prompt": prompt_a, "values": stored_logits["A"]},
            {"prompt": prompt_b, "values": stored_logits["B"]},
            {"prompt": prompt_c, "values": stored_logits["C"]},
        ],
        expected_results,
        rtol=1e-2,
    )

    await pipeline_model.release(context_a)
    await pipeline_model.release(context_b)
    await pipeline_model.release(context_c)
