# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Tests for InternVL tokenizer."""

import io
from unittest.mock import MagicMock

import numpy as np
import pytest
from max.pipelines import TokenGeneratorRequest, TokenGeneratorRequestMessage
from max.pipelines.architectures.internvl.tokenizer import InternVLTokenizer
from PIL import Image


@pytest.mark.asyncio
async def test_internvl_tokenizer_new_context_smoke(mocker) -> None:
    """Smoke test to ensure new_context() doesn't raise"""
    # Create minimal mocks
    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token_id = 2
    mock_tokenizer.model_max_length = 2048
    mock_tokenizer.encode.return_value = [1, 2, 3]
    mock_tokenizer.apply_chat_template.return_value = "test prompt"

    mocker.patch(
        "max.pipelines.architectures.internvl.tokenizer.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer,
    )
    mocker.patch(
        "max.pipelines.architectures.internvl.tokenizer.AutoConfig.from_pretrained",
        return_value=MagicMock(),
    )

    tokenizer = InternVLTokenizer("test-model")

    # Mock the processor to return expected format
    tokenizer.processor = MagicMock()
    tokenizer.processor.return_value = {
        "input_ids": [1, 2, 3],
    }

    request = TokenGeneratorRequest(
        messages=[TokenGeneratorRequestMessage(role="user", content="test")],
        index=0,
        id="test-id",
        model_name="test-model",
    )

    context = await tokenizer.new_context(request)

    assert context is not None


@pytest.mark.asyncio
async def test_internvl_tokenizer_image_token_indices(mocker) -> None:
    """Test that the tokenizer correctly computes image token indices."""
    # Create minimal mocks
    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token_id = 2
    mock_tokenizer.model_max_length = 2048
    mock_tokenizer.encode.return_value = [1, 2, 3]
    mock_tokenizer.apply_chat_template.return_value = "test prompt"

    mocker.patch(
        "max.pipelines.architectures.internvl.tokenizer.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer,
    )

    # Mock config with vision config using realistic values from internvl-2B
    mock_config = MagicMock()
    mock_config.vision_config.image_size = 448
    mock_config.vision_config.patch_size = 14
    mock_config.max_dynamic_patch = 12
    mock_config.downsample_ratio = 0.5
    mocker.patch(
        "max.pipelines.architectures.internvl.tokenizer.AutoConfig.from_pretrained",
        return_value=mock_config,
    )

    tokenizer = InternVLTokenizer("test-model")

    # Mock the processor to return input_ids with image token
    # IMAGE_CONTEXT_TOKEN_ID = 151667
    tokenizer.processor = MagicMock()
    tokenizer.processor.return_value = {
        "input_ids": [
            1,
            2,
            151667,
            151667,
            3,
            151667,
            4,
        ],  # 3 image tokens at positions 2, 3, 5
        "pixel_values": [
            [np.zeros((448, 448, 3), dtype=np.float32)]
        ],  # Mock image data
        "image_token_indices": np.array(
            [2, 3, 5], dtype=np.int32
        ),  # Pre-computed indices
    }

    # Create a real image for the test using config dimensions
    img_buffer = io.BytesIO()
    image_size = 448
    Image.new("RGB", (image_size, image_size), color="red").save(
        img_buffer, format="PNG"
    )
    test_image = img_buffer.getvalue()

    # Create request with image
    request = TokenGeneratorRequest(
        messages=[
            TokenGeneratorRequestMessage(
                role="user",
                content=[
                    {"type": "text", "text": "test"},
                    {"type": "image", "content": test_image},
                ],
            )
        ],
        images=[test_image],
        index=0,
        id="test-id",
        model_name="test-model",
    )

    context = await tokenizer.new_context(request)

    assert context is not None
    # Verify that image_token_indices were passed through to extra_model_args
    assert "image_token_indices" in context.extra_model_args


@pytest.mark.asyncio
async def test_internvl_tokenizer_image_placement(mocker) -> None:
    """Test that image tokens are correctly placed in a multi-turn conversation."""
    # Create minimal mocks
    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token_id = 2
    mock_tokenizer.model_max_length = 4096
    mock_tokenizer.return_value = {"input_ids": [1, 2, 3]}

    # Define a fixed, multi-turn prompt that the mocked template will return.
    # This simulates the output of the real chat template.
    multi_turn_prompt = (
        "<|im_start|>user\nWhat is in this image?<|im_end|>\n"
        "<|im_start|>assistant\nI can't see an image.<|im_end|>\n"
        "<|im_start|>user\nSorry, here it is.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    mock_tokenizer.apply_chat_template.return_value = multi_turn_prompt

    mocker.patch(
        "max.pipelines.architectures.internvl.tokenizer.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer,
    )

    # Mock config with vision config
    mock_config = MagicMock()
    mock_config.vision_config.image_size = 448
    mock_config.vision_config.patch_size = 14
    mock_config.max_dynamic_patch = 1
    mock_config.downsample_ratio = 0.5
    mocker.patch(
        "max.pipelines.architectures.internvl.tokenizer.AutoConfig.from_pretrained",
        return_value=mock_config,
    )

    # Use the real tokenizer to exercise the processor logic, but with a mocked delegate.
    tokenizer = InternVLTokenizer("test-model")

    # Create a real image for the test
    img_buffer = io.BytesIO()
    Image.new("RGB", (100, 100), color="red").save(img_buffer, format="PNG")
    test_image = img_buffer.getvalue()

    # Create a multi-turn request with an image in the last turn
    request = TokenGeneratorRequest(
        messages=[
            TokenGeneratorRequestMessage(
                role="user", content="What is in this image?"
            ),
            TokenGeneratorRequestMessage(
                role="assistant", content="I can't see an image."
            ),
            TokenGeneratorRequestMessage(
                role="user",
                content=[
                    {"type": "text", "text": "Sorry, here it is."},
                    {"type": "image", "content": test_image},
                ],
            ),
        ],
        images=[test_image],
        index=0,
        id="test-id",
        model_name="test-model",
    )

    await tokenizer.new_context(request)

    # The processor calls the mocked tokenizer delegate to tokenize the final text.
    # We inspect the arguments of that call to verify the text was modified correctly.
    call_args, _ = mock_tokenizer.call_args
    processed_text = call_args[0]

    expected_last_user_prompt = "user\n<img><IMG_CONTEXT>"

    # Assert that the image tokens were inserted into the last user prompt.
    assert expected_last_user_prompt in processed_text

    # Check that the image was inserted in the right place.
    assert processed_text.rfind("<img>") > processed_text.rfind(
        "I can't see an image."
    )
    assert processed_text.endswith("<|im_start|>assistant\n")
