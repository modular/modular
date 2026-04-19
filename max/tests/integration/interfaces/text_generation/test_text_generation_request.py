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
    ImageContentPart,
    RequestID,
    TextContentPart,
    TextGenerationRequest,
    TextGenerationRequestMessage,
    VideoContentPart,
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
                    content=[TextContentPart(text="hello world")],
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
                content=[TextContentPart(text="hello world")],
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
                        TextContentPart(text="hello world"),
                        ImageContentPart(),
                        ImageContentPart(),
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
                    content=[TextContentPart(text="hello world")],
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
                    TextContentPart(text="hello world"),
                    ImageContentPart(),
                    ImageContentPart(),
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
                TextGenerationRequestMessage.model_validate(
                    {
                        "role": "not_user",
                        "content": [{"type": "text", "text": "hello world"}],
                    }
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
                TextGenerationRequestMessage.model_validate(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "hello world"},
                            {
                                "type": "image_url",
                                "image_url": "https://example.com/image.jpg",
                            },
                        ],
                    }
                )
            ],
        )


def test_text_generation_request_message_dict_roundtrip() -> None:
    # Test that messages_dict == dict(TextGenerationRequestMessage(**messages_dict))
    messages_dict = {
        "role": "user",
        "content": [{"type": "text", "text": "hello world"}],
    }

    message = TextGenerationRequestMessage.model_validate(messages_dict)
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

    message_with_images = TextGenerationRequestMessage.model_validate(
        messages_dict_with_images
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
        content=[TextContentPart(text="response text")],
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
            TextContentPart(text="first line"),
            TextContentPart(text="second line"),
            TextContentPart(text="third line"),
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
            TextContentPart(text="describe this"),
            ImageContentPart(),
        ],
    )
    with pytest.raises(ValueError, match="only text content can be flattened"):
        message_with_image.flatten_content()

    # Test with only image content
    message_only_image = TextGenerationRequestMessage(
        role="user",
        content=[ImageContentPart()],
    )
    with pytest.raises(ValueError, match="only text content can be flattened"):
        message_only_image.flatten_content()


def test_video_content_part_creation() -> None:
    """VideoContentPart can be created and used in messages."""
    msg = TextGenerationRequestMessage(
        role="user",
        content=[TextContentPart(text="describe this"), VideoContentPart()],
    )
    assert msg.number_of_videos == 1
    assert msg.number_of_images == 0


def test_video_content_part_dict_validation() -> None:
    """VideoContentPart can be created from a dict with type='video'."""
    msg = TextGenerationRequestMessage.model_validate(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "watch this"},
                {"type": "video"},
            ],
        }
    )
    assert msg.number_of_videos == 1
    dumped = msg.model_dump()
    assert dumped["content"][1] == {"type": "video"}


def test_video_url_content_type_rejected() -> None:
    """video_url content type is not supported in internal format."""
    with pytest.raises(ValueError, match="video_url"):
        TextGenerationRequestMessage.model_validate(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "watch this"},
                    {
                        "type": "video_url",
                        "video_url": "https://example.com/v.mp4",
                    },
                ],
            }
        )


def test_video_request_validation() -> None:
    """TextGenerationRequest validates video count against messages."""
    # Valid: 1 video content part, 1 video bytes
    _ = TextGenerationRequest(
        request_id=RequestID(),
        model_name="test",
        messages=[
            TextGenerationRequestMessage(
                role="user",
                content=[TextContentPart(text="watch"), VideoContentPart()],
            )
        ],
        videos=[b"video_data"],
    )

    # Invalid: mismatch between video content parts and video bytes
    with pytest.raises(ValueError, match="videos"):
        TextGenerationRequest(
            request_id=RequestID(),
            model_name="test",
            messages=[
                TextGenerationRequestMessage(
                    role="user",
                    content=[TextContentPart(text="watch")],
                )
            ],
            videos=[b"video_data"],
        )

    # Invalid: string prompt with videos
    with pytest.raises(ValueError, match="string prompts"):
        TextGenerationRequest(
            request_id=RequestID(),
            model_name="test",
            prompt="hello",
            videos=[b"video_data"],
        )


def test_mixed_image_and_video() -> None:
    """Request with both images and videos validates correctly."""
    _ = TextGenerationRequest(
        request_id=RequestID(),
        model_name="test",
        messages=[
            TextGenerationRequestMessage(
                role="user",
                content=[
                    TextContentPart(text="compare"),
                    ImageContentPart(),
                    VideoContentPart(),
                ],
            )
        ],
        images=[b"img"],
        videos=[b"vid"],
    )
