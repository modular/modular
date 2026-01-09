# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Tests for InternVL tokenizer."""

import io
from unittest.mock import MagicMock

import numpy as np
import pytest
from max.interfaces import (
    RequestID,
    TextGenerationRequest,
    TextGenerationRequestMessage,
)
from max.pipelines.architectures.internvl.tokenizer import InternVLTokenizer
from PIL import Image
from pytest_mock import MockerFixture


@pytest.mark.asyncio
async def test_internvl_tokenizer_new_context_smoke(
    mocker: MockerFixture,
) -> None:
    """Smoke test to ensure new_context() doesn't raise"""
    # Create minimal mocks
    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token_id = 2
    mock_tokenizer.model_max_length = 2048
    mock_tokenizer.encode.return_value = np.array([1, 2, 3], dtype=np.int64)
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
    # InternVL processor returns Python lists for input_ids, not numpy arrays
    tokenizer.processor = MagicMock()
    tokenizer.processor.return_value = {"input_ids": [1, 2, 3]}
    tokenizer.processor.apply_chat_template.return_value = "test prompt"

    request = TextGenerationRequest(
        messages=[TextGenerationRequestMessage(role="user", content="test")],
        request_id=RequestID("test-id"),
        model_name="test-model",
    )

    context = await tokenizer.new_context(request)

    assert context is not None


@pytest.mark.asyncio
async def test_super_long(
    mocker: MockerFixture,
) -> None:
    """Test to ensure new_context() raises if prompt is too long"""
    # Create minimal mocks
    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token_id = 2
    mock_tokenizer.model_max_length = 5
    mock_tokenizer.encode.return_value = np.array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int64
    )
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
    # InternVL processor returns Python lists for input_ids, not numpy arrays
    tokenizer.processor = MagicMock()
    tokenizer.processor.return_value = {
        "input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
    tokenizer.processor.apply_chat_template.return_value = "test prompt"

    request = TextGenerationRequest(
        messages=[TextGenerationRequestMessage(role="user", content="test")],
        request_id=RequestID("test-id"),
        model_name="test-model",
    )

    with pytest.raises(
        ValueError,
        match="encoded_prompt is greater than the max_length of the tokenizer",
    ):
        _ = await tokenizer.new_context(request)


@pytest.mark.asyncio
async def test_internvl_tokenizer_image_token_indices(
    mocker: MockerFixture,
) -> None:
    """Test that the tokenizer correctly computes image token indices."""
    # Create minimal mocks
    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token_id = 2
    mock_tokenizer.model_max_length = 2048
    mock_tokenizer.encode.return_value = np.array([1, 2, 3], dtype=np.int64)
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
    # InternVL processor returns Python lists for input_ids, not numpy arrays
    tokenizer.processor = MagicMock()
    tokenizer.processor.return_value = {
        # 2 image tokens at positions 2, 3
        "input_ids": [1, 2, 151667, 151667, 3, 4],
        # Mock image data
        "pixel_values": [[np.zeros((448, 448, 3), dtype=np.float32)]],
        # Pre-computed indices
        "image_token_indices": np.array([2, 3, 5], dtype=np.int32),
    }
    tokenizer.processor.apply_chat_template.return_value = "test prompt"

    # Create a real image for the test using config dimensions
    img_buffer = io.BytesIO()
    image_size = 448
    Image.new("RGB", (image_size, image_size), color="red").save(
        img_buffer, format="PNG"
    )
    test_image = img_buffer.getvalue()

    # Create request with image
    request = TextGenerationRequest(
        messages=[
            TextGenerationRequestMessage(
                role="user",
                content=[
                    {"type": "text", "text": "test"},
                    {"type": "image", "content": test_image},
                ],
            )
        ],
        images=[test_image],
        request_id=RequestID("test-id"),
        model_name="test-model",
    )

    context = await tokenizer.new_context(request)

    assert context is not None
    # Verify that image_token_indices were passed through to extra_model_args
    assert "image_token_indices" in context.extra_model_args


@pytest.mark.asyncio
async def test_internvl_tokenizer_image_placement(
    mocker: MockerFixture,
) -> None:
    """Test that image tokens are correctly placed in a multi-turn conversation."""
    # Create minimal mocks
    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token_id = 2
    mock_tokenizer.model_max_length = 4096
    # Need one image in input_ids
    # InternVL processor returns Python lists for input_ids, not numpy arrays
    mock_tokenizer.return_value = {"input_ids": [1, 2, 151667, 151667, 3, 4]}

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
    request = TextGenerationRequest(
        messages=[
            TextGenerationRequestMessage(
                role="user", content="What is in this image?"
            ),
            TextGenerationRequestMessage(
                role="assistant", content="I can't see an image."
            ),
            TextGenerationRequestMessage(
                role="user",
                content=[
                    {"type": "text", "text": "Sorry, here it is."},
                    {"type": "image", "content": test_image},
                ],
            ),
        ],
        images=[test_image],
        request_id=RequestID("test-id"),
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
