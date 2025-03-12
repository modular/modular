# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from typing import Any

import pytest
from max.pipelines import (
    PIPELINE_REGISTRY,
    PipelineConfig,
    SupportedEncoding,
    TextGenerationPipeline,
    TextTokenizer,
)
from max.pipelines.architectures import register_all_models
from max.pipelines.interfaces import TokenGeneratorRequest
from test_common.evaluate import PROMPTS, next_token_with_logits


@pytest.fixture(scope="session")
def pipeline_config() -> PipelineConfig:
    if not PIPELINE_REGISTRY.architectures:
        register_all_models()

    config = PipelineConfig(
        model_path="HuggingFaceTB/SmolLM-135M",
        quantization_encoding=SupportedEncoding.float32,
        max_cache_batch_size=4,
    )

    return PIPELINE_REGISTRY.validate_pipeline_config(config)


@pytest.fixture(scope="session")
def pipeline_tokenizer(pipeline_config: PipelineConfig) -> TextTokenizer:
    pipeline_config = PipelineConfig(
        model_path="HuggingFaceTB/SmolLM-135M",
        quantization_encoding=SupportedEncoding.float32,
    )
    return TextTokenizer(
        pipeline_config.model_config.model_path,
        revision=pipeline_config.model_config.huggingface_revision,
        max_length=pipeline_config.max_length,
        max_new_tokens=pipeline_config.max_new_tokens,
        trust_remote_code=pipeline_config.model_config.trust_remote_code,
    )


@pytest.fixture(scope="session")
def pipeline(
    pipeline_config: PipelineConfig, pipeline_tokenizer: TextTokenizer
) -> TextGenerationPipeline:
    if not PIPELINE_REGISTRY.architectures:
        register_all_models()

    _, pipeline = PIPELINE_REGISTRY.retrieve(pipeline_config)
    assert isinstance(pipeline, TextGenerationPipeline)
    return pipeline


@pytest.mark.asyncio
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
