# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from typing import Any

import hf_repo_lock
import pytest
from max.interfaces import (
    SamplingParams,
    TextGenerationInputs,
    TextGenerationRequest,
)
from max.nn.kv_cache import KVCacheStrategy
from max.pipelines import (
    PIPELINE_REGISTRY,
    PipelineConfig,
    SupportedEncoding,
    TextGenerationPipeline,
    TextTokenizer,
)
from test_common.evaluate import next_token_with_logits
from test_common.test_data import DEFAULT_PROMPTS

LLAMA_3_1_HF_REPO_ID = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA_3_1_HF_REVISION = hf_repo_lock.revision_for_hf_repo(LLAMA_3_1_HF_REPO_ID)

LLAMA_3_1_LORA_HF_REPO_ID = "FinGPT/fingpt-mt_llama3-8b_lora"
LLAMA_3_1_LORA_HF_REVISION = hf_repo_lock.revision_for_hf_repo(
    LLAMA_3_1_LORA_HF_REPO_ID
)


@pytest.fixture
def pipeline_config(llama_3_1_8b_lora_local_path: str) -> PipelineConfig:
    return PipelineConfig(
        model_path=LLAMA_3_1_HF_REPO_ID,
        quantization_encoding=SupportedEncoding.bfloat16,
        max_batch_size=2,
        max_num_loras=1,
        lora_paths=[llama_3_1_8b_lora_local_path],
        max_lora_rank=8,
        allow_safetensors_weights_float32_to_bfloat16_cast=True,
        max_length=2048,
        cache_strategy=KVCacheStrategy.PAGED,
    )


@pytest.fixture
def pipeline_tokenizer(pipeline_config: PipelineConfig) -> TextTokenizer:
    return TextTokenizer(
        pipeline_config.model_config.model_path,
        revision=pipeline_config.model_config.huggingface_model_revision,
        max_length=pipeline_config.max_length,
        max_new_tokens=pipeline_config.max_new_tokens,
        trust_remote_code=True,
    )


@pytest.fixture
def pipeline(pipeline_config: PipelineConfig) -> TextGenerationPipeline:
    _, pipeline = PIPELINE_REGISTRY.retrieve(pipeline_config)
    assert isinstance(pipeline, TextGenerationPipeline)
    return pipeline


@pytest.mark.skip("Add back when a LoRA is uploaded to S3 for CI.")
@pytest.mark.asyncio
async def test_pipeline_lora(
    pipeline: TextGenerationPipeline,
    pipeline_tokenizer: TextTokenizer,
    llama_3_1_8b_lora_local_path: str,
) -> None:
    prompt_a = DEFAULT_PROMPTS[0]
    stored_logits: dict[str, list[Any]] = {"A": []}

    # Send in A for context encoding.
    context_a = await pipeline_tokenizer.new_context(
        TextGenerationRequest(
            request_id="",
            index=0,
            prompt=prompt_a,
            model_name=llama_3_1_8b_lora_local_path,
        )
    )
    next_token_with_logits(
        pipeline._pipeline_model, {"A": context_a}, stored_logits
    )

    # Send in A for token generation
    next_token_with_logits(
        pipeline._pipeline_model,
        {"A": context_a},
        stored_logits,
    )

    pipeline.release(context_a.request_id)


@pytest.mark.skip("Default Prompts produce the same output for LoRA and Base.")
@pytest.mark.asyncio
async def test_lora_vs_base_model_logits(
    pipeline: TextGenerationPipeline,
    pipeline_tokenizer: TextTokenizer,
    llama_3_1_8b_lora_local_path: str,
) -> None:
    """Test that compares logits between base model and LoRA model to validate LoRA is working."""
    prompt = DEFAULT_PROMPTS[0]

    assert pipeline._pipeline_model._lora_manager is not None, (
        "Pipeline should have LoRA manager"
    )
    assert len(pipeline._pipeline_model._lora_manager._loras) > 0, (
        "LoRA manager should have loaded adapters"
    )

    sampling_params = SamplingParams(max_new_tokens=300)

    base_context = await pipeline_tokenizer.new_context(
        TextGenerationRequest(
            request_id="base_test",
            index=0,
            prompt=prompt,
            model_name="llama3",  # Any string
            sampling_params=sampling_params,
        )
    )
    lora_context = await pipeline_tokenizer.new_context(
        TextGenerationRequest(
            request_id="lora_test",
            index=1,  # Reset index since we released the previous context
            prompt=prompt,
            model_name="llama3",
            sampling_params=sampling_params,
        )
    )

    contexts = {"lora": lora_context, "base": base_context}
    tokens: dict[str, list[Any]] = {"lora": [], "base": []}
    done = {}
    while True:
        inputs = TextGenerationInputs(batch=contexts, num_steps=1)
        response = pipeline.next_token(inputs)

        for name, ctx in response.items():
            tokens[name].extend(ctx.tokens)

        for name, ctx in response.items():
            if ctx.is_done:
                done[name] = ctx
                del contexts[name]

        if not contexts:
            break

    pipeline.release(base_context.request_id)
    pipeline.release(lora_context.request_id)

    # Compare tokens between LoRA and base model
    base_tokens = tokens["base"]
    lora_tokens = tokens["lora"]

    print(len(base_tokens))
    print(len(lora_tokens))
    print(prompt)
    # Find the minimum length to compare
    min_len = min(len(base_tokens), len(lora_tokens))

    assert min_len > 0, "Both models should have generated at least one token"

    # Check that at least some tokens are different
    num_different = 0
    for i in range(min_len):
        if base_tokens[i] != lora_tokens[i]:
            num_different += 1

    assert num_different > 0
