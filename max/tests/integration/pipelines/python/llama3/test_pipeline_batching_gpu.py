# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import logging
from typing import Any

import pytest
from max.interfaces import TextGenerationRequest
from max.pipelines import (
    PIPELINE_REGISTRY,
    PipelineConfig,
    SupportedEncoding,
    TextGenerationPipeline,
    TextTokenizer,
)
from test_common.evaluate import next_token_with_logits
from test_common.test_data import DEFAULT_PROMPTS

logger = logging.getLogger("max.pipelines")


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
        trust_remote_code=pipeline_config.model_config.trust_remote_code,
        allow_safetensors_weights_fp32_bf6_bidirectional_cast=True,
    )


@pytest.fixture
def pipeline(pipeline_config: PipelineConfig) -> TextGenerationPipeline:
    _, pipeline = PIPELINE_REGISTRY.retrieve(pipeline_config)
    assert isinstance(pipeline, TextGenerationPipeline)
    return pipeline


@pytest.mark.skip(
    "AITLIB-336: ValueError: Attempted to release request ID but it is not claimed"
)
@pytest.mark.asyncio
async def test_pipeline_heterogeneous_batch_logits(
    pipeline: TextGenerationPipeline,
    pipeline_tokenizer: TextTokenizer,
) -> None:
    """Executes batch of prompts with different lengths and validates logits.

    NOTE: Intentionally don't compare results with "goldens" because TinyLlama
    weights were randomly initialized.
    """

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
    next_token_with_logits(
        pipeline._pipeline_model, {"A": context_a}, stored_logits
    )

    # Send in B for context encoding
    context_b = await pipeline_tokenizer.new_context(
        TextGenerationRequest(
            request_id="", prompt=prompt_b, model_name="llama3"
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
        TextGenerationRequest(
            request_id="", prompt=prompt_c, model_name="llama3"
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

    pipeline.release(context_a.request_id)
    pipeline.release(context_b.request_id)
    pipeline.release(context_c.request_id)
