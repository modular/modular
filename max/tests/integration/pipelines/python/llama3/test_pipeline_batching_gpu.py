# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from dataclasses import dataclass
from typing import Any, Literal

import pytest
from evaluate_llama import SupportedTestModels
from max.pipelines import (
    PIPELINE_REGISTRY,
    PipelineConfig,
    SupportedEncoding,
    TextGenerationPipeline,
    TextTokenizer,
)
from max.pipelines.architectures import register_all_models
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
    version: str = "3.1"

    """Whether to include a print hook. This is generally for debugging
    purposes, but it's also helping to avoid a segfault in the heterogeneous
    test."""

    def __str__(self):
        return f"{self.name}-{self.encoding}-{self.max_length}-{self.max_batch_size}"


@pytest.fixture(scope="session")
def pipeline_config(testdata_directory, request) -> PipelineConfig:
    print(request)
    model_params: PipelineModelParams = request.param
    print(f"\nPipelineModel: {model_params}")
    encoding = model_params.encoding
    test_model = SupportedTestModels.get(model_params.name, encoding)

    if encoding in [SupportedEncoding.float32, SupportedEncoding.bfloat16]:
        cache_strategy = KVCacheStrategy.CONTINUOUS
    else:
        cache_strategy = KVCacheStrategy.NAIVE

    return test_model.build_config(
        testdata_directory=testdata_directory,
        max_length=model_params.max_length,
        max_new_tokens=model_params.max_new_tokens,
        max_batch_size=model_params.max_batch_size,
        cache_strategy=cache_strategy,
        pad_to_multiple_of=2,
    )


@pytest.fixture(scope="session")
def pipeline_tokenizer(pipeline_config: PipelineConfig) -> TextTokenizer:
    return TextTokenizer(
        pipeline_config.model_path,
        revision=pipeline_config.huggingface_revision,
        max_length=pipeline_config.max_length,
        max_new_tokens=pipeline_config.max_new_tokens,
        trust_remote_code=pipeline_config.trust_remote_code,
    )


@pytest.fixture(scope="session")
def pipeline(
    pipeline_config: PipelineConfig,
) -> TextGenerationPipeline:
    register_all_models()
    _, pipeline = PIPELINE_REGISTRY.retrieve(pipeline_config)
    assert isinstance(pipeline, TextGenerationPipeline)
    return pipeline


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pipeline_config",
    [
        PipelineModelParams(
            "tinyllama",
            SupportedEncoding.bfloat16,
            512,
            10,
            4,
        ),
    ],
    ids=PipelineModelParams.__str__,
    indirect=True,
)
async def test_pipeline_heterogeneous_batch_logits(
    pipeline: TextGenerationPipeline,
    pipeline_tokenizer: TextTokenizer,
) -> None:
    """Executes batch of prompts with different lengths and validates logits.

    NOTE: Intentionally don't compare results with "goldens" because TinyLlama
    weights were randomly initialized.
    """

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
    next_token_with_logits(
        pipeline._pipeline_model, {"A": context_a}, stored_logits
    )

    # Send in B for context encoding
    context_b = await pipeline_tokenizer.new_context(
        TokenGeneratorRequest(
            id="", index=1, prompt=prompt_b, model_name="llama3"
        )
    )
    next_token_with_logits(
        pipeline._pipeline_model, {"B": context_b}, stored_logits
    )

    # Send in both A and B for token generation
    next_token_with_logits(
        pipeline._pipeline_model,
        {"A": context_a, "B": context_b},
        stored_logits,
    )

    # Send in C for context encoding
    context_c = await pipeline_tokenizer.new_context(
        TokenGeneratorRequest(
            id="", index=2, prompt=prompt_c, model_name="llama3"
        )
    )
    next_token_with_logits(
        pipeline._pipeline_model, {"C": context_c}, stored_logits
    )

    # Send in both B and C for token generation
    next_token_with_logits(
        pipeline._pipeline_model,
        {"B": context_b, "C": context_c},
        stored_logits,
    )

    pipeline.release(context_a)
    pipeline.release(context_b)
    pipeline.release(context_c)
