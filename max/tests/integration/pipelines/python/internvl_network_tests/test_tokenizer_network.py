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
from max.pipelines.architectures.internvl.tokenizer import InternVLProcessor
from max.pipelines.lib import PipelineEngine
from PIL import Image


@pytest.mark.asyncio
async def test_internvl_tokenizer_with_image() -> None:
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


@pytest.mark.asyncio
async def test_internvl_tokenizer_apply_chat_template(mocker) -> None:
    """Test that InternVL tokenizer's apply_chat_template handles multimodal content correctly.

    This test verifies that the InternVL processor can handle messages with multimodal
    content (text + image) without throwing warnings about string concatenation.
    """
    # Create a mock tokenizer with apply_chat_template.
    mock_tokenizer = mocker.MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "User: What is this?"

    # Create a mock config.
    mock_config = mocker.MagicMock()
    mock_config.vision_config.image_size = 448
    mock_config.vision_config.patch_size = 14
    mock_config.max_dynamic_patch = 12
    mock_config.downsample_ratio = 0.5

    # Create processor.
    processor = InternVLProcessor(mock_tokenizer, mock_config)

    # Test with multimodal message (text + image).
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "content": "What is this?"},
                {"type": "image"},
            ],
        }
    ]

    # Mock the warning logger.
    mock_warning = mocker.patch("max.pipelines.lib.tokenizer.logger.warning")

    result = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Verify no warnings were logged.
    mock_warning.assert_not_called()

    # Verify the tokenizer was called with text-only content.
    mock_tokenizer.apply_chat_template.assert_called_once()
    called_messages = mock_tokenizer.apply_chat_template.call_args[0][0]

    # Check that the content was converted to string.
    assert len(called_messages) == 1
    assert called_messages[0]["role"] == "user"
    assert isinstance(called_messages[0]["content"], str)
    assert called_messages[0]["content"] == "What is this?"

    # Verify result.
    assert result == "User: What is this?"

    # Test with text-only message.
    mock_tokenizer.apply_chat_template.reset_mock()
    text_only_messages = [{"role": "user", "content": "Hello world"}]

    result2 = processor.apply_chat_template(
        text_only_messages, tokenize=False, add_generation_prompt=True
    )

    # Verify it still works with text-only.
    called_messages2 = mock_tokenizer.apply_chat_template.call_args[0][0]
    assert called_messages2[0]["content"] == "Hello world"

    # Test with multiple text parts in content.
    mock_tokenizer.apply_chat_template.reset_mock()
    mock_tokenizer.apply_chat_template.return_value = "User: Hello world"

    multi_text_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "content": "Hello"},
                {"type": "image"},
                {"type": "text", "content": "world"},
            ],
        }
    ]

    result3 = processor.apply_chat_template(
        multi_text_messages, tokenize=False, add_generation_prompt=True
    )

    called_messages3 = mock_tokenizer.apply_chat_template.call_args[0][0]
    assert called_messages3[0]["content"] == "Hello world"
