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
    except FileNotFoundError:
        logger.warning(
            f"Model path does not exist: {REPO_ID}@{REVISION}, falling back to repo_id: {REPO_ID} as config to PipelineConfig"
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


@pytest.mark.skip("AITLIB-336: Bug that causes a mismatch in matmul dimensions")
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
