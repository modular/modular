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


from max.serve.config import Settings
from max.serve.router.openai_routes import openai_parse_chat_completion_request

"""
It is unclear why the type ignore for CreateChatCompletionRequest is necessary.
bazel+mypy complain about this import not being available even though it is part of the serving package.
Explicitly importing //max/python/max/serve/schemas in the test's BUILD file hasn't worked either.
"""

import pytest
from max.serve.schemas.openai import CreateChatCompletionRequest
from pydantic import AnyUrl


@pytest.mark.skip
async def test_openai_extract_image_from_requests() -> None:
    request_images = {
        "smily_b64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAApgAAAKYB3X3/OAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAANCSURBVEiJtZZPbBtFFMZ/M7ubXdtdb1xSFyeilBapySVU8h8OoFaooFSqiihIVIpQBKci6KEg9Q6H9kovIHoCIVQJJCKE1ENFjnAgcaSGC6rEnxBwA04Tx43t2FnvDAfjkNibxgHxnWb2e/u992bee7tCa00YFsffekFY+nUzFtjW0LrvjRXrCDIAaPLlW0nHL0SsZtVoaF98mLrx3pdhOqLtYPHChahZcYYO7KvPFxvRl5XPp1sN3adWiD1ZAqD6XYK1b/dvE5IWryTt2udLFedwc1+9kLp+vbbpoDh+6TklxBeAi9TL0taeWpdmZzQDry0AcO+jQ12RyohqqoYoo8RDwJrU+qXkjWtfi8Xxt58BdQuwQs9qC/afLwCw8tnQbqYAPsgxE1S6F3EAIXux2oQFKm0ihMsOF71dHYx+f3NND68ghCu1YIoePPQN1pGRABkJ6Bus96CutRZMydTl+TvuiRW1m3n0eDl0vRPcEysqdXn+jsQPsrHMquGeXEaY4Yk4wxWcY5V/9scqOMOVUFthatyTy8QyqwZ+kDURKoMWxNKr2EeqVKcTNOajqKoBgOE28U4tdQl5p5bwCw7BWquaZSzAPlwjlithJtp3pTImSqQRrb2Z8PHGigD4RZuNX6JYj6wj7O4TFLbCO/Mn/m8R+h6rYSUb3ekokRY6f/YukArN979jcW+V/S8g0eT/N3VN3kTqWbQ428m9/8k0P/1aIhF36PccEl6EhOcAUCrXKZXXWS3XKd2vc/TRBG9O5ELC17MmWubD2nKhUKZa26Ba2+D3P+4/MNCFwg59oWVeYhkzgN/JDR8deKBoD7Y+ljEjGZ0sosXVTvbc6RHirr2reNy1OXd6pJsQ+gqjk8VWFYmHrwBzW/n+uMPFiRwHB2I7ih8ciHFxIkd/3Omk5tCDV1t+2nNu5sxxpDFNx+huNhVT3/zMDz8usXC3ddaHBj1GHj/As08fwTS7Kt1HBTmyN29vdwAw+/wbwLVOJ3uAD1wi/dUH7Qei66PfyuRj4Ik9is+hglfbkbfR3cnZm7chlUWLdwmprtCohX4HUtlOcQjLYCu+fzGJH2QRKvP3UNz8bWk1qMxjGTOMThZ3kvgLI5AzFfo379UAAAAASUVORK5CYII=",
        "boardwark_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
        "mountain_url": "https://picsum.photos/seed/picsum/200/300",
    }

    system_message = {
        "role": "system",
        "content": "You are an opinionated chat-bot.",
    }
    user_message_no_images = {
        "role": "user",
        "content": [
            {"type": "text", "text": "What'''s in this image?"},
        ],
    }
    request = CreateChatCompletionRequest.model_validate(
        {"model": "test", "messages": [system_message, user_message_no_images]}
    )

    settings = Settings()
    messages, images, _videos = await openai_parse_chat_completion_request(
        request, False, settings
    )
    assert len(messages) == 2
    assert len(images) == 0
    assert isinstance(messages[0].content, str)
    assert isinstance(messages[1].content, list)
    assert hasattr(messages[1].content[0], "text")

    messages, images, _videos = await openai_parse_chat_completion_request(
        request, True, settings
    )
    assert len(messages) == 2
    assert len(images) == 0
    assert isinstance(messages[0].content, list)
    assert isinstance(messages[1].content, list)
    assert hasattr(messages[1].content[0], "text")

    user_message_image_with_url = {
        "role": "user",
        "content": [
            {"type": "text", "text": "What'''s in this image?"},
            {
                "type": "image_url",
                "image_url": {"url": request_images["boardwark_url"]},
            },
        ],
    }
    request = CreateChatCompletionRequest.model_validate(
        {"model": "test", "messages": [user_message_image_with_url]}
    )
    messages, images, _videos = await openai_parse_chat_completion_request(
        request,
        False,
        settings,
    )
    assert len(messages) == 1
    assert len(images) == 1
    assert isinstance(messages[0].content, list)
    # When wrap_content=False, content items are dicts
    assert isinstance(messages[0].content[1], dict)
    assert "image_url" in messages[0].content[1]
    assert images[0] == AnyUrl(request_images["boardwark_url"])

    messages, images = await openai_parse_chat_completion_request(
        request,
        True,
        settings,
    )
    assert len(messages) == 1
    assert len(images) == 1
    assert isinstance(messages[0].content, list)
    assert messages[0].content[1].type == "image"
    assert images[0] == AnyUrl(request_images["boardwark_url"])

    user_message_image_two_urls = {
        "role": "user",
        "content": [
            {"type": "text", "text": "What'''s in these images?"},
            {
                "type": "image_url",
                "image_url": {"url": request_images["boardwark_url"]},
            },
            {
                "type": "image_url",
                "image_url": {"url": request_images["mountain_url"]},
            },
        ],
    }
    messages, images = await openai_parse_chat_completion_request(
        CreateChatCompletionRequest(
            model="test", messages=[system_message, user_message_image_two_urls]
        ),
        False,
        settings,
    )
    assert len(messages) == 2
    assert len(images) == 2
    assert images[0] == AnyUrl(request_images["boardwark_url"])
    assert images[1] == AnyUrl(request_images["mountain_url"])

    user_message_mixed_url_b64 = {
        "role": "user",
        "content": [
            {"type": "text", "text": "What'''s in these images?"},
            {
                "type": "image_url",
                "image_url": {"url": request_images["smily_b64"]},
            },
            {
                "type": "image_url",
                "image_url": {"url": request_images["mountain_url"]},
            },
        ],
    }
    request = CreateChatCompletionRequest(
        model="test", messages=[user_message_mixed_url_b64]
    )
    messages, images = await openai_parse_chat_completion_request(
        request, False, settings
    )
    assert len(messages) == 1
    assert len(images) == 2
    assert images[0] == AnyUrl(request_images["smily_b64"])
    assert images[1] == AnyUrl(request_images["mountain_url"])

    messages, images = await openai_parse_chat_completion_request(
        request, True, settings
    )
    assert len(messages) == 1
    assert len(images) == 2
    assert isinstance(messages[0].content, list)
    assert messages[0].content[1].type == "image"
    assert messages[0].content[2].type == "image"
    assert images[0] == AnyUrl(request_images["smily_b64"])
    assert images[1] == AnyUrl(request_images["mountain_url"])


async def test_openai_user_message_with_null_content() -> None:
    """Test that user messages with null content are accepted and handled."""
    # Test with explicit null content
    user_message_null_content = {
        "role": "user",
        "content": None,
    }
    request = CreateChatCompletionRequest.model_validate(
        {"model": "test", "messages": [user_message_null_content]}
    )
    settings = Settings()
    messages, _images, _videos = await openai_parse_chat_completion_request(
        request, False, settings
    )
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].content == ""

    # Test with missing content field (should default to None)
    user_message_no_content = {
        "role": "user",
    }
    request = CreateChatCompletionRequest.model_validate(
        {"model": "test", "messages": [user_message_no_content]}
    )
    messages, _images, _videos = await openai_parse_chat_completion_request(
        request, False, settings
    )
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].content == ""


def test_openai_chat_completion_accepts_prompt_tokens() -> None:
    """Schema must accept ``prompt_tokens`` (orchestrator pre-tokenized input).

    The Mammoth orchestrator tokenizes incoming requests once and forwards
    the integer token IDs to MAX Serve under the ``prompt_tokens`` field,
    bypassing re-tokenization. Regression coverage for SERVSYS-1239: the
    schema rewrite in #84789 dropped this MAX-only field, causing
    ``CreateChatCompletionRequest.model_validate_json`` to fail with
    ``ValidationError: prompt_tokens - Extra inputs are not permitted``.
    """
    body = (
        '{"model":"test","ignore_eos":true,"max_tokens":4,'
        '"messages":[{"role":"user","content":"hi"}],'
        '"prompt_tokens":[101,202,303]}'
    )
    request = CreateChatCompletionRequest.model_validate_json(body)
    assert request.prompt_tokens == [101, 202, 303]


async def test_openai_parse_forwards_tool_call_metadata() -> None:
    """Multi-turn tool-use messages must keep ``tool_calls`` and ``tool_call_id``.

    The router previously dropped these fields when building the internal
    ``TextGenerationRequestMessage`` list, so the chat-templated prompt
    rendered with an empty ``<think>`` block and a bare ``## Return of``
    header instead of the originating function name (for example Kimi-K2).
    """
    request_data = {
        "model": "test",
        "messages": [
            {"role": "user", "content": "search for cats"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "I'll call the search tool.",
                "tool_calls": [
                    {
                        "id": "call_9e53d2d2_0",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"q":"cats"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_9e53d2d2_0",
                "content": "1. fluffy cat\n2. orange cat",
            },
        ],
    }
    request = CreateChatCompletionRequest.model_validate(request_data)
    settings = Settings()

    messages, _images, _videos = await openai_parse_chat_completion_request(
        request, wrap_content=False, settings=settings
    )

    assert len(messages) == 3
    assert messages[0].role == "user"
    assert messages[0].tool_calls is None
    assert messages[0].tool_call_id is None

    assert messages[1].role == "assistant"
    assert messages[1].tool_calls is not None
    assert len(messages[1].tool_calls) == 1
    assert messages[1].tool_calls[0]["id"] == "call_9e53d2d2_0"
    assert messages[1].tool_calls[0]["function"]["name"] == "search"
    assert messages[1].reasoning_content == "I'll call the search tool."

    assert messages[2].role == "tool"
    assert messages[2].tool_call_id == "call_9e53d2d2_0"
    assert messages[2].content == "1. fluffy cat\n2. orange cat"


def test_openai_user_message_content_nullable_schema() -> None:
    """Test that the CreateChatCompletionRequest schema accepts null user content."""
    # Test with explicit null content in user message
    request_data = {
        "model": "test",
        "messages": [{"role": "user", "content": None}],
    }
    request = CreateChatCompletionRequest.model_validate(request_data)
    assert len(request.messages) == 1
    assert request.messages[0]["role"] == "user"
    assert request.messages[0].get("content") is None

    # Test with omitted content field
    request_data_no_content = {
        "model": "test",
        "messages": [{"role": "user"}],
    }
    request = CreateChatCompletionRequest.model_validate(
        request_data_no_content
    )
    assert len(request.messages) == 1
    assert request.messages[0]["role"] == "user"
    assert request.messages[0].get("content") is None

    # Test mixed messages with null user content
    request_data_mixed = {
        "model": "test",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": None},
            {"role": "assistant", "content": "How can I help you?"},
            {"role": "user", "content": "Hello!"},
        ],
    }
    request = CreateChatCompletionRequest.model_validate(request_data_mixed)
    assert len(request.messages) == 4
    assert request.messages[0]["content"] == "You are a helpful assistant."
    assert request.messages[1].get("content") is None
    assert request.messages[2]["content"] == "How can I help you?"
    assert request.messages[3]["content"] == "Hello!"
