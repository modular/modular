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
from pydantic import AnyUrl, ValidationError


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


async def test_openai_parse_normalizes_developer_role_to_system() -> None:
    """``role: "developer"`` must be accepted and normalized to ``"system"``.

    OpenAI's model chat-completion spec uses ``developer`` as the
    system-equivalent role. The internal ``TextGenerationRequestMessage``
    ``_MessageRole`` literal only enumerates the five spec-supported roles
    (``system``, ``user``, ``assistant``, ``tool``, ``function``), so the
    request was previously rejected with a 422. Normalize at the
    OpenAI-compat seam so requests from OpenAI model spec compliant clients are accepted.
    """
    request_data = {
        "model": "test",
        "messages": [
            {"role": "developer", "content": "You are a coding assistant."},
            {"role": "user", "content": "hi"},
        ],
    }
    request = CreateChatCompletionRequest.model_validate(request_data)
    settings = Settings()

    messages, _images, _videos = await openai_parse_chat_completion_request(
        request, wrap_content=False, settings=settings
    )

    assert len(messages) == 2
    assert messages[0].role == "system"
    assert messages[0].content == "You are a coding assistant."
    assert messages[1].role == "user"


def test_openai_parse_rejects_unknown_role() -> None:
    """Roles outside the spec-supported set still surface as a validation error.

    The ``developer`` normalization must not become a permissive sink:
    arbitrary role strings remain rejected by the schema so genuine malformed
    requests still produce a 4xx. The role union is now validated by pydantic
    at ``model_validate`` time (SERVSYS-1257), so the rejection happens here
    rather than later inside ``openai_parse_chat_completion_request``.
    """
    request_data = {
        "model": "test",
        "messages": [{"role": "wizard", "content": "abracadabra"}],
    }
    with pytest.raises(ValidationError):
        CreateChatCompletionRequest.model_validate(request_data)


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
    # ``function.arguments`` is decoded from the OpenAI JSON-string wire
    # format into a mapping so tool-use chat templates can iterate it.
    assert messages[1].tool_calls[0]["function"]["arguments"] == {"q": "cats"}
    assert messages[1].reasoning_content == "I'll call the search tool."

    assert messages[2].role == "tool"
    assert messages[2].tool_call_id == "call_9e53d2d2_0"
    assert messages[2].content == "1. fluffy cat\n2. orange cat"


async def test_openai_parse_drops_empty_tool_calls() -> None:
    """Empty assistant ``tool_calls`` lists must be dropped (vLLM parity).

    Some clients echo back ``"tool_calls": []`` on assistant turns even
    when the assistant did not call any tools. Letting an empty list reach
    the chat template causes tool-use branches to fire with no entries,
    which renders broken multi-turn prompts.
    """
    request_data = {
        "model": "test",
        "messages": [
            {
                "role": "assistant",
                "content": "ok",
                "tool_calls": [],
            },
        ],
    }
    request = CreateChatCompletionRequest.model_validate(request_data)
    settings = Settings()

    messages, _images, _videos = await openai_parse_chat_completion_request(
        request, wrap_content=False, settings=settings
    )

    assert len(messages) == 1
    assert messages[0].role == "assistant"
    assert messages[0].tool_calls is None
    assert messages[0].content == "ok"


def test_openai_request_rejects_tool_call_missing_name() -> None:
    """Assistant ``tool_calls`` missing ``function.name`` is rejected.

    Regression for the llm-fuzz ``tool_calling/bad_tool_call_missing_name``
    case. With the strongly-typed ``messages: list[ChatCompletionMessageParam]``
    field, pydantic validates ``function.name`` (declared ``Required[str]`` on
    the OpenAI SDK ``TypedDict``) at ``model_validate`` time and surfaces a
    422 - no hand-coded check needed in the route.
    """
    request_data = {
        "model": "test",
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_test",
                        "type": "function",
                        "function": {"arguments": "{}"},
                    }
                ],
            },
        ],
    }
    with pytest.raises(ValidationError) as excinfo:
        CreateChatCompletionRequest.model_validate(request_data)
    assert "function.name" in str(excinfo.value)


def test_openai_request_rejects_tool_call_empty_function() -> None:
    """Assistant ``tool_calls`` with an empty ``function`` object is rejected.

    Regression for the llm-fuzz ``tool_calling/bad_tool_call_empty_function``
    case. Same mechanism as ``test_openai_request_rejects_tool_call_missing_name``:
    ``function.name`` is ``Required[str]`` on the OpenAI SDK ``TypedDict`` so
    pydantic refuses ``function = {}`` at request validation.
    """
    request_data = {
        "model": "test",
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_test", "type": "function", "function": {}},
                ],
            },
        ],
    }
    with pytest.raises(ValidationError) as excinfo:
        CreateChatCompletionRequest.model_validate(request_data)
    assert "function.name" in str(excinfo.value)


async def test_openai_parse_coerces_empty_tool_call_arguments() -> None:
    """Empty ``function.arguments`` are coerced to ``{}``.

    Mirrors vLLM's ``_postprocess_messages``: clients that send no-arg
    tool calls (for example a ``get_time()`` invocation) emit
    ``"arguments": ""`` over the wire. Chat templates that iterate the
    mapping must see an empty dict, not the empty string.
    """
    request_data = {
        "model": "test",
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_time", "arguments": ""},
                    },
                ],
            },
        ],
    }
    request = CreateChatCompletionRequest.model_validate(request_data)
    settings = Settings()

    messages, _images, _videos = await openai_parse_chat_completion_request(
        request, wrap_content=False, settings=settings
    )

    assert len(messages) == 1
    assert messages[0].tool_calls is not None
    assert len(messages[0].tool_calls) == 1
    assert messages[0].tool_calls[0]["function"]["arguments"] == {}


def test_openai_request_rejects_tool_call_missing_arguments() -> None:
    """Assistant ``tool_calls`` missing ``function.arguments`` is rejected.

    OpenAI's spec marks ``Function.arguments`` as ``Required[str]``.
    Requests that omit it entirely must fail schema validation rather
    than being silently coerced.
    """
    request_data = {
        "model": "test",
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_date"},
                    },
                ],
            },
        ],
    }
    with pytest.raises(ValidationError):
        CreateChatCompletionRequest.model_validate(request_data)


def test_openai_chat_completion_accepts_explicit_null_tool_choice() -> None:
    """Schema must accept ``"tool_choice": null`` from OpenAI-compatible clients.

    OpenAI's ``ChatCompletionToolChoiceOptionParam`` is a non-Optional union
    of ``Literal["none","auto","required"]`` and tool-choice objects;
    omission is expressed via ``NotRequired`` on the TypedDict. Some clients
    (LangChain, certain JS SDKs, anything that serializes a dataclass with
    a ``None`` field) explicitly emit ``"tool_choice": null`` instead of
    omitting the key, which must be treated as equivalent to omission.
    """
    body = {
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "tool_choice": None,
    }
    request = CreateChatCompletionRequest.model_validate(body)
    assert request.tool_choice is None


def test_openai_chat_message_validates_structure() -> None:
    """Regression for SERVSYS-1257: ``messages`` should not be typed as
    ``list[dict[str, Any]]`` (the ``Any`` clobbered OpenAI SDK validation).

    Confirms that ``CreateChatCompletionRequest`` now validates each
    message against the OpenAI ``ChatCompletionMessageParam`` union -
    invalid roles, missing required fields, and malformed content parts
    surface as a 422 instead of being silently accepted.
    """
    base = {"model": "test"}

    # Invalid role is rejected (was accepted previously).
    with pytest.raises(ValidationError):
        CreateChatCompletionRequest.model_validate(
            {**base, "messages": [{"role": "wizard", "content": "magic"}]}
        )

    # Missing required ``role`` is rejected.
    with pytest.raises(ValidationError):
        CreateChatCompletionRequest.model_validate(
            {**base, "messages": [{"content": "no role"}]}
        )


def test_openai_chat_message_multipart_content_preserves_list() -> None:
    """Regression for SERVSYS-1257: validated multi-part ``content`` is a
    concrete ``list`` of plain dicts that can be re-iterated.

    Before this fix, ``messages`` was typed as ``list[dict[str, Any]]`` to
    sidestep pydantic stashing the inner ``Iterable[ContentPart]`` as a
    one-shot ``ValidatorIterator``. With the recursive ``Iterable -> list``
    normalization in place, the content array survives validation as a
    real list and the route can index/iterate it freely.
    """
    request = CreateChatCompletionRequest.model_validate(
        {
            "model": "test",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/foo.png"},
                        },
                    ],
                }
            ],
        }
    )
    content = request.messages[0].get("content")
    assert isinstance(content, list)
    assert len(content) == 2
    # Re-iteration must yield the same dicts (used to break on a
    # ``ValidatorIterator`` that consumed itself after the first pass).
    first_pass = [c.get("type") for c in content]
    second_pass = [c.get("type") for c in content]
    assert first_pass == second_pass == ["text", "image_url"]
    # Items are plain dicts (TypedDict -> dict at the JSON layer).
    assert content[0]["text"] == "what's in this image?"
    assert content[1]["image_url"]["url"] == "https://example.com/foo.png"


def test_openai_chat_message_preserves_vendor_extensions() -> None:
    """``reasoning_content`` (vLLM-style extension on assistant turns) and
    other vendor-specific message fields must survive validation.

    The normalized chat-message ``TypedDict`` mirrors are configured with
    ``extra='allow'`` so passthrough keys aren't silently dropped.
    """
    request = CreateChatCompletionRequest.model_validate(
        {
            "model": "test",
            "messages": [
                {
                    "role": "assistant",
                    "content": "ok",
                    "reasoning_content": "preserved",
                }
            ],
        }
    )
    assert request.messages[0].get("reasoning_content") == "preserved"


def test_openai_chat_message_validates_vendor_extension_type() -> None:
    """``reasoning_content`` is a first-class field; pydantic enforces its type.

    Previously ``reasoning_content`` rode through ``extra='allow'`` on
    the message TypedDict, so a non-string value passed validation and
    later tripped a route-level ``assert`` (-> unhandled ``AssertionError``
    -> 500). Declaring it on :class:`ChatCompletionMessageParam` makes
    pydantic reject the bad type at ``model_validate`` time so the
    request surfaces as a clean 4xx via the chat route's existing
    ``ValidationError`` handler.
    """
    with pytest.raises(ValidationError):
        CreateChatCompletionRequest.model_validate(
            {
                "model": "test",
                "messages": [
                    {
                        "role": "assistant",
                        "content": "ok",
                        "reasoning_content": 123,
                    }
                ],
            }
        )


def test_openai_chat_message_rejects_non_string_tool_call_id() -> None:
    """``tool_call_id`` must be a string; non-string values are rejected."""
    with pytest.raises(ValidationError):
        CreateChatCompletionRequest.model_validate(
            {
                "model": "test",
                "messages": [
                    {
                        "role": "tool",
                        "content": "result",
                        "tool_call_id": 123,
                    }
                ],
            }
        )


def test_openai_chat_message_rejects_invalid_content_type() -> None:
    """``content`` must be a string, list, or null; integers are rejected."""
    with pytest.raises(ValidationError):
        CreateChatCompletionRequest.model_validate(
            {
                "model": "test",
                "messages": [
                    {
                        "role": "user",
                        "content": 42,
                    }
                ],
            }
        )


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
