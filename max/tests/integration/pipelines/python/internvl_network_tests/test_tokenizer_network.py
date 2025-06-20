# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Tests for InternVL tokenizer that require network access."""

import io
import uuid

import pytest
from max import pipelines
from max.pipelines import (
    PipelineConfig,
    TokenGeneratorRequest,
)
from max.pipelines.lib import PipelineEngine
from PIL import Image


@pytest.mark.asyncio
async def test_internvl_tokenizer_with_image():
    """Test InternVL tokenizer adds image tokens correctly."""
    model_id = "OpenGVLab/InternVL3-1B-Instruct"
    test_text = "What is this?"
    image_token_id = 151667  # InternVL's <IMG_CONTEXT> token
    expected_image_tokens = 256  # 256 tokens per 448x448 image patch (after 14x14 patch embeddings and 0.5x downsampling)

    # Create test image
    img_buffer = io.BytesIO()
    Image.new("RGB", (448, 448), color="red").save(img_buffer, format="PNG")
    test_image = img_buffer.getvalue()

    # Get tokenizer
    config = PipelineConfig(
        model_path=model_id, engine=PipelineEngine.MAX, trust_remote_code=True
    )
    max_tokenizer, _ = pipelines.PIPELINE_REGISTRY.retrieve_factory(config)

    # Note: We don't compare with HF tokenizer for image inputs because InternVL's
    # image token handling differs between implementations. In HF, the image tokens
    # are typically inserted by the processor/model during forward pass, not by the
    # tokenizer itself. Our MAX implementation inserts image placeholder tokens
    # directly in the tokenizer for consistency with our pipeline architecture.

    # Compare text-only vs text+image tokenization
    text_context = await max_tokenizer.new_context(
        TokenGeneratorRequest(
            id=str(uuid.uuid4()),
            index=0,
            model_name=model_id,
            prompt=test_text,
        )
    )
    image_context = await max_tokenizer.new_context(
        TokenGeneratorRequest(
            id=str(uuid.uuid4()),
            index=0,
            model_name=model_id,
            prompt=test_text,
            images=[test_image],
        )
    )

    # Verify image tokens were added
    assert len(image_context.all_tokens) > len(text_context.all_tokens)

    num_image_tokens = (image_context.all_tokens == image_token_id).sum()
    assert num_image_tokens == expected_image_tokens
