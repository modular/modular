# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test Suite for SmolLM2 with LoRA adapters."""

import pytest
from max.interfaces import (
    SamplingParams,
    TextGenerationInputs,
    TextGenerationRequest,
)
from max.pipelines import TextGenerationPipeline
from max.pipelines.core.context import TextContext
from test_common.graph_utils import is_h100_h200
from test_common.lora_utils import (
    create_multiple_test_lora_adapters,
    create_pipeline_with_lora,
    create_test_lora_adapter,
    create_tokenizer,
)


def generate_tokens_from_contexts(
    pipeline: TextGenerationPipeline, contexts: dict[str, TextContext]
) -> dict[str, list[int]]:
    """Generate tokens from multiple contexts using the same pipeline.

    Args:
        pipeline: The text generation pipeline to use
        contexts: Dictionary mapping request_id to TextContext

    Returns:
        Dictionary mapping request_id to list of generated tokens
    """
    all_tokens: dict[str, list[int]] = {req_id: [] for req_id in contexts}
    active_contexts = contexts.copy()

    while active_contexts:
        response = pipeline.execute(
            TextGenerationInputs(active_contexts, num_steps=1)
        )
        for req_id, resp in response.items():
            all_tokens[req_id].extend(resp.tokens)
            if resp.is_done:
                del active_contexts[req_id]

    return all_tokens


@pytest.mark.skipif(is_h100_h200(), reason="LoRA tests fail on H100 and H200")
@pytest.mark.asyncio
async def test_smollm2_with_lora_adapter() -> None:
    """Test SmolLM2 with LoRA adapter loaded."""
    lora_path = create_test_lora_adapter()
    pipeline = create_pipeline_with_lora([lora_path])
    tokenizer = create_tokenizer()

    assert pipeline._pipeline_model._lora_manager is not None
    assert len(pipeline._pipeline_model._lora_manager._loras) == 1
    assert lora_path in pipeline._pipeline_model._lora_manager._loras

    prompt = "The future of AI is"
    sampling_params = SamplingParams(
        max_new_tokens=30,
        temperature=0.0,
        top_k=1,
    )

    base_context = await tokenizer.new_context(
        TextGenerationRequest(
            request_id="base",
            prompt=prompt,
            model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
            sampling_params=sampling_params,
        )
    )

    lora_context = await tokenizer.new_context(
        TextGenerationRequest(
            request_id="lora",
            prompt=prompt,
            model_name=lora_path,
            sampling_params=sampling_params,
        )
    )

    base_tokens = generate_tokens_from_contexts(
        pipeline, {"base": base_context}
    )["base"]
    lora_tokens = generate_tokens_from_contexts(
        pipeline, {"lora": lora_context}
    )["lora"]

    assert len(base_tokens) > 0
    assert len(lora_tokens) > 0
    assert len(lora_tokens) <= 30

    pipeline.release(base_context.request_id)
    pipeline.release(lora_context.request_id)


@pytest.mark.skipif(is_h100_h200(), reason="LoRA tests fail on H100 and H200")
@pytest.mark.asyncio
async def test_lora_vs_base_comparison() -> None:
    """Compare outputs between base model and LoRA-adapted model."""
    lora_path = create_test_lora_adapter()
    pipeline = create_pipeline_with_lora([lora_path])
    tokenizer = create_tokenizer()

    prompt = "The future of AI is"
    sampling_params = SamplingParams(
        max_new_tokens=30,
        temperature=0.0,
        top_k=1,
    )

    base_context = await tokenizer.new_context(
        TextGenerationRequest(
            request_id="base",
            prompt=prompt,
            model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
            sampling_params=sampling_params,
        )
    )

    lora_context = await tokenizer.new_context(
        TextGenerationRequest(
            request_id="lora",
            prompt=prompt,
            model_name=lora_path,
            sampling_params=sampling_params,
        )
    )

    base_tokens = generate_tokens_from_contexts(
        pipeline, {"base": base_context}
    )["base"]
    lora_tokens = generate_tokens_from_contexts(
        pipeline, {"lora": lora_context}
    )["lora"]

    assert len(base_tokens) > 0
    assert len(lora_tokens) > 0

    pipeline.release(base_context.request_id)
    pipeline.release(lora_context.request_id)


@pytest.mark.skipif(is_h100_h200(), reason="LoRA tests fail on H100 and H200")
@pytest.mark.asyncio
async def test_multiple_lora_adapters() -> None:
    """Test loading and using multiple LoRA adapters."""
    lora_paths = create_multiple_test_lora_adapters(num_adapters=2)
    pipeline = create_pipeline_with_lora(lora_paths)
    tokenizer = create_tokenizer()

    assert pipeline._pipeline_model._lora_manager is not None
    assert len(pipeline._pipeline_model._lora_manager._loras) == len(lora_paths)

    prompt = "The future of AI is"
    sampling_params = SamplingParams(
        max_new_tokens=20,
        temperature=0.0,
        top_k=1,
    )

    contexts = {
        "base": await tokenizer.new_context(
            TextGenerationRequest(
                request_id="base",
                prompt=prompt,
                model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
                sampling_params=sampling_params,
            )
        ),
        "lora1": await tokenizer.new_context(
            TextGenerationRequest(
                request_id="lora1",
                prompt=prompt,
                model_name=lora_paths[0],
                sampling_params=sampling_params,
            )
        ),
        "lora2": await tokenizer.new_context(
            TextGenerationRequest(
                request_id="lora2",
                prompt=prompt,
                model_name=lora_paths[1],
                sampling_params=sampling_params,
            )
        ),
    }

    all_tokens = {
        "base": generate_tokens_from_contexts(
            pipeline, {"base": contexts["base"]}
        )["base"],
        "lora1": generate_tokens_from_contexts(
            pipeline, {"lora1": contexts["lora1"]}
        )["lora1"],
        "lora2": generate_tokens_from_contexts(
            pipeline, {"lora2": contexts["lora2"]}
        )["lora2"],
    }

    for name, tokens in all_tokens.items():
        assert len(tokens) > 0, f"No tokens generated for {name}"
        assert len(tokens) <= 20, f"Too many tokens for {name}: {len(tokens)}"

    for context in contexts.values():
        pipeline.release(context.request_id)
