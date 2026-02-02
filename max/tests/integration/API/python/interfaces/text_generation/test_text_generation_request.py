# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

import pytest
from max.interfaces import (
    RequestID,
    TextGenerationRequest,
    TextGenerationRequestMessage,
)


def test_text_generation_request_init() -> None:
    # Prompt and messages cannot be provided concurrently.
    with pytest.raises(ValueError):
        _ = TextGenerationRequest(
            request_id=RequestID(),
            model_name="test",
            prompt="hello world",
            messages=[
                TextGenerationRequestMessage(
                    role="user",
                    content=[{"type": "text", "text": "hello world"}],
                )
            ],
        )

    _ = TextGenerationRequest(
        request_id=RequestID(),
        model_name="test",
        prompt="hello world",
    )

    _ = TextGenerationRequest(
        request_id=RequestID(),
        model_name="test",
        prompt=None,
        messages=[
            TextGenerationRequestMessage(
                role="user",
                content=[{"type": "text", "text": "hello world"}],
            )
        ],
    )

    # String prompts with images provided are not accepted.
    with pytest.raises(ValueError):
        _ = TextGenerationRequest(
            request_id=RequestID(),
            model_name="test",
            prompt="hello world",
            messages=[],
            images=[b""],
        )

    # If images are provided, we should verify there is an appropriate message for each.
    with pytest.raises(ValueError):
        _ = TextGenerationRequest(
            request_id=RequestID(),
            model_name="test",
            prompt=None,
            messages=[
                TextGenerationRequestMessage(
                    role="user",
                    content=[
                        {"type": "text", "text": "hello world"},
                        {"type": "image"},
                        {"type": "image"},
                    ],
                )
            ],
            images=[b""],
        )

    with pytest.raises(ValueError):
        _ = TextGenerationRequest(
            request_id=RequestID(),
            model_name="test",
            prompt=None,
            messages=[
                TextGenerationRequestMessage(
                    role="user",
                    content=[{"type": "text", "text": "hello world"}],
                )
            ],
            images=[b"", b""],
        )

    _ = TextGenerationRequest(
        request_id=RequestID(),
        model_name="test",
        prompt=None,
        messages=[
            TextGenerationRequestMessage(
                role="user",
                content=[
                    {"type": "text", "text": "hello world"},
                    {"type": "image"},
                    {"type": "image"},
                ],
            )
        ],
        images=[b"", b""],
    )

    # role not user is not supported.
    with pytest.raises(ValueError):
        _ = TextGenerationRequest(
            request_id=RequestID(),
            model_name="test",
            prompt=None,
            messages=[
                TextGenerationRequestMessage(
                    role="not_user",
                    content=[{"type": "text", "text": "hello world"}],
                )
            ],
        )

    # image_url content type is not supported in internal format.
    with pytest.raises(ValueError):
        _ = TextGenerationRequest(
            request_id=RequestID(),
            model_name="test",
            prompt=None,
            messages=[
                TextGenerationRequestMessage(
                    role="user",
                    content=[
                        {"type": "text", "text": "hello world"},
                        {
                            "type": "image_url",
                            "image_url": "https://example.com/image.jpg",
                        },
                    ],
                )
            ],
        )


def test_text_generation_request_message_dict_roundtrip() -> None:
    # Test that messages_dict == dict(TextGenerationRequestMessage(**messages_dict))
    messages_dict = {
        "role": "user",
        "content": [{"type": "text", "text": "hello world"}],
    }

    message = TextGenerationRequestMessage(**messages_dict)
    assert messages_dict == message.model_dump()

    # Test with images
    messages_dict_with_images = {
        "role": "user",
        "content": [
            {"type": "text", "text": "describe this"},
            {"type": "image"},
            {"type": "image"},
        ],
    }

    message_with_images = TextGenerationRequestMessage(
        **messages_dict_with_images
    )
    assert messages_dict_with_images == message_with_images.model_dump()


def test_text_generation_request_message_flatten_content() -> None:
    # Test with string content
    message_str = TextGenerationRequestMessage(
        role="user",
        content="hello world",
    )
    flattened = message_str.flatten_content()
    assert flattened == {
        "role": "user",
        "content": "hello world",
    }

    # Test with single text content part
    message_single_text = TextGenerationRequestMessage(
        role="assistant",
        content=[{"type": "text", "text": "response text"}],
    )
    flattened = message_single_text.flatten_content()
    assert flattened == {
        "role": "assistant",
        "content": "response text",
    }

    # Test with multiple text content parts (should be joined with newlines)
    message_multi_text = TextGenerationRequestMessage(
        role="user",
        content=[
            {"type": "text", "text": "first line"},
            {"type": "text", "text": "second line"},
            {"type": "text", "text": "third line"},
        ],
    )
    flattened = message_multi_text.flatten_content()
    assert flattened == {
        "role": "user",
        "content": "first line\nsecond line\nthird line",
    }

    # Test that image content raises ValueError
    message_with_image = TextGenerationRequestMessage(
        role="user",
        content=[
            {"type": "text", "text": "describe this"},
            {"type": "image"},
        ],
    )
    with pytest.raises(ValueError, match="only text content can be flattened"):
        message_with_image.flatten_content()

    # Test with only image content
    message_only_image = TextGenerationRequestMessage(
        role="user",
        content=[{"type": "image"}],
    )
    with pytest.raises(ValueError, match="only text content can be flattened"):
        message_only_image.flatten_content()
