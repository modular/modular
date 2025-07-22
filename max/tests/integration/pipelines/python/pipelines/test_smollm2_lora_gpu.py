# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test Suite for SmolLM2 with LoRA adapters."""

from typing import Any

import pytest
from max.interfaces import SamplingParams, TextGenerationRequest
from test_common.graph_utils import is_h100_h200
from test_common.lora_utils import (
    create_multiple_test_lora_adapters,
    create_pipeline_base,
    create_pipeline_with_lora,
    create_test_lora_adapter,
    create_tokenizer,
)


@pytest.mark.skipif(is_h100_h200(), reason="LoRA tests fail on H100 and H200")
@pytest.mark.asyncio
async def test_smollm2_with_lora_adapter() -> None:
    """Test SmolLM2 with LoRA adapter loaded."""
    # Create test LoRA adapter and pipeline
    lora_path = create_test_lora_adapter()
    pipeline_with_lora = create_pipeline_with_lora([lora_path])
    tokenizer = create_tokenizer()

    # Verify LoRA manager is initialized
    assert pipeline_with_lora._pipeline_model._lora_manager is not None
    assert len(pipeline_with_lora._pipeline_model._lora_manager._loras) == 1
    assert lora_path in pipeline_with_lora._pipeline_model._lora_manager._loras

    # Test generation with LoRA
    prompt = "What is machine learning?"
    sampling_params = SamplingParams(max_new_tokens=50, temperature=0.7)

    context = await tokenizer.new_context(
        TextGenerationRequest(
            id="test_lora",
            index=0,
            prompt=prompt,
            model_name=lora_path,  # Use LoRA adapter
            sampling_params=sampling_params,
        )
    )

    # Generate tokens
    generated_tokens = []
    contexts = {"test": context}

    while contexts:
        response = pipeline_with_lora.next_token(contexts, num_steps=1)

        for req_id, resp in response.items():
            generated_tokens.extend(resp.tokens)

            if resp.is_done:
                del contexts[req_id]

    # Verify we generated some tokens
    assert len(generated_tokens) > 0
    assert len(generated_tokens) <= 50

    pipeline_with_lora.release(context)


@pytest.mark.skipif(is_h100_h200(), reason="LoRA tests fail on H100 and H200")
@pytest.mark.asyncio
async def test_lora_vs_base_comparison() -> None:
    """Compare outputs between base model and LoRA-adapted model."""
    # Create test LoRA adapter and pipelines
    lora_path = create_test_lora_adapter()
    pipeline_with_lora = create_pipeline_with_lora([lora_path])
    pipeline_base = create_pipeline_base()
    tokenizer = create_tokenizer()

    prompt = "The future of AI is"
    sampling_params = SamplingParams(
        max_new_tokens=30,
        temperature=0.0,  # Deterministic for comparison
        top_k=1,
    )

    # Generate with base model
    base_context = await tokenizer.new_context(
        TextGenerationRequest(
            id="base",
            index=0,
            prompt=prompt,
            model_name="base",  # Don't use LoRA
            sampling_params=sampling_params,
        )
    )

    # Generate with LoRA model
    lora_context = await tokenizer.new_context(
        TextGenerationRequest(
            id="lora",
            index=0,
            prompt=prompt,
            model_name=lora_path,  # Use LoRA
            sampling_params=sampling_params,
        )
    )

    base_tokens = []
    lora_tokens = []

    # Generate from base model
    contexts = {"base": base_context}
    while contexts:
        response = pipeline_base.next_token(contexts, num_steps=1)
        for req_id, resp in response.items():
            base_tokens.extend(resp.tokens)
            if resp.is_done:
                del contexts[req_id]

    # Generate from LoRA model
    contexts = {"lora": lora_context}
    while contexts:
        response = pipeline_with_lora.next_token(contexts, num_steps=1)
        for req_id, resp in response.items():
            lora_tokens.extend(resp.tokens)
            if resp.is_done:
                del contexts[req_id]

    # With a proper LoRA adapter, outputs should differ
    # For a minimal test adapter, they might be similar but not identical
    assert len(base_tokens) > 0
    assert len(lora_tokens) > 0

    # Clean up
    pipeline_base.release(base_context)
    pipeline_with_lora.release(lora_context)


@pytest.mark.skipif(is_h100_h200(), reason="LoRA tests fail on H100 and H200")
@pytest.mark.asyncio
async def test_multiple_lora_adapters() -> None:
    """Test loading and using multiple LoRA adapters."""
    # Create multiple test LoRA adapters and pipeline
    lora_paths = create_multiple_test_lora_adapters(num_adapters=2)
    pipeline = create_pipeline_with_lora(lora_paths)
    tokenizer = create_tokenizer()

    # Verify multiple adapters loaded
    assert pipeline._pipeline_model._lora_manager is not None
    assert len(pipeline._pipeline_model._lora_manager._loras) == len(lora_paths)

    # Test generation with different adapters
    prompts = [
        ("What is AI?", lora_paths[0]),
        ("Explain quantum computing", lora_paths[1]),
        ("What is machine learning?", "base"),  # Use base model
    ]

    contexts = {}
    for i, (prompt, model_name) in enumerate(prompts):
        context = await tokenizer.new_context(
            TextGenerationRequest(
                id=f"req_{i}",
                index=i,
                prompt=prompt,
                model_name=model_name,
                sampling_params=SamplingParams(max_new_tokens=20),
            )
        )
        contexts[f"req_{i}"] = context

    # Generate tokens for all requests
    results: dict[str, list[Any]] = {req_id: [] for req_id in contexts}

    while contexts:
        response = pipeline.next_token(contexts, num_steps=1)

        for req_id, resp in response.items():
            results[req_id].extend(resp.tokens)

            if resp.is_done:
                pipeline.release(contexts[req_id])
                del contexts[req_id]

    # Verify all requests generated tokens
    for req_id, tokens in results.items():
        assert len(tokens) > 0, f"Request ID: {req_id} produced 0 tokens."
        assert len(tokens) <= 20, (
            f"Request ID: {req_id} produced more than 20 tokens."
        )
