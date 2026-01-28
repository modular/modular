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
"""Tests for OpenResponses Pydantic schema."""

import json

import pytest
from max.interfaces.request.open_responses import (
    AssistantMessage,
    CreateResponseBody,
    FunctionCall,
    FunctionToolParam,
    InputTextContent,
    OutputTextContent,
    ResponseResource,
    SystemMessage,
    ToolChoiceValueEnum,
    UserMessage,
)
from pydantic import ValidationError


def test_import_all_types() -> None:
    """Test that all OpenResponses types can be imported."""

    # If we get here, all imports succeeded
    assert True


def test_create_response_body_minimal() -> None:
    """Test creating a minimal CreateResponseBody with required fields only."""
    request = CreateResponseBody(
        model="gpt-4",
        input="Hello, world!",
    )

    assert request.model == "gpt-4"
    assert request.input == "Hello, world!"
    assert request.temperature is None
    assert request.max_output_tokens is None


def test_create_response_body_with_messages() -> None:
    """Test CreateResponseBody with structured message input."""
    request = CreateResponseBody(
        model="gpt-4",
        input=[
            UserMessage(
                role="user",
                content="What's the weather in San Francisco?",
            )
        ],
        temperature=0.7,
        max_output_tokens=1000,
    )

    assert request.model == "gpt-4"
    assert len(request.input) == 1
    assert isinstance(request.input[0], UserMessage)
    assert request.temperature == 0.7
    assert request.max_output_tokens == 1000


def test_create_response_body_with_tools() -> None:
    """Test CreateResponseBody with tool definitions."""
    request = CreateResponseBody(
        model="gpt-4",
        input="What's the weather?",
        tools=[
            FunctionToolParam(
                type="function",
                name="get_weather",
                description="Get weather for a location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                    },
                    "required": ["location"],
                },
            )
        ],
        tool_choice=ToolChoiceValueEnum.auto,
    )

    assert request.tools is not None
    assert len(request.tools) == 1
    assert request.tools[0].name == "get_weather"
    assert request.tool_choice == ToolChoiceValueEnum.auto


def test_user_message_with_string_content() -> None:
    """Test UserMessage with simple string content."""
    msg = UserMessage(
        role="user",
        content="Hello!",
    )

    assert msg.role == "user"
    assert msg.content == "Hello!"
    assert msg.name is None


def test_user_message_with_structured_content() -> None:
    """Test UserMessage with structured content."""
    msg = UserMessage(
        role="user",
        content=[
            InputTextContent(type="input_text", text="Look at this image:"),
        ],
    )

    assert msg.role == "user"
    assert len(msg.content) == 1
    assert isinstance(msg.content[0], InputTextContent)
    assert msg.content[0].text == "Look at this image:"


def test_assistant_message_with_tool_calls() -> None:
    """Test AssistantMessage with tool calls."""
    msg = AssistantMessage(
        role="assistant",
        content="I'll check the weather for you.",
        tool_calls=[
            FunctionCall(
                id="call_123",
                type="function",
                name="get_weather",
                arguments='{"location": "San Francisco"}',
            )
        ],
    )

    assert msg.role == "assistant"
    assert msg.content == "I'll check the weather for you."
    assert msg.tool_calls is not None
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].name == "get_weather"


def test_system_message() -> None:
    """Test SystemMessage creation."""
    msg = SystemMessage(
        role="system",
        content="You are a helpful assistant.",
    )

    assert msg.role == "system"
    assert msg.content == "You are a helpful assistant."


def test_response_resource_minimal() -> None:
    """Test creating a minimal ResponseResource."""
    response = ResponseResource(
        id="resp_123",
        object="response",
        created_at=1234567890,
        status="completed",
        model="gpt-4",
    )

    assert response.id == "resp_123"
    assert response.object == "response"
    assert response.status == "completed"
    assert response.model == "gpt-4"
    assert response.output is None


def test_models_are_frozen() -> None:
    """Test that all models are frozen (immutable)."""
    request = CreateResponseBody(
        model="gpt-4",
        input="Hello!",
    )

    # Attempting to modify should raise an error
    # Pydantic frozen models raise ValidationError on attribute assignment
    with pytest.raises(ValidationError):
        request.model = "gpt-3.5"  # type: ignore[misc]


def test_temperature_validation() -> None:
    """Test temperature field validation constraints."""
    # Valid temperature
    request = CreateResponseBody(
        model="gpt-4",
        input="Hello!",
        temperature=0.7,
    )
    assert request.temperature == 0.7

    # Temperature too high should fail
    with pytest.raises(ValidationError):
        CreateResponseBody(
            model="gpt-4",
            input="Hello!",
            temperature=2.5,
        )

    # Temperature too low should fail
    with pytest.raises(ValidationError):
        CreateResponseBody(
            model="gpt-4",
            input="Hello!",
            temperature=-0.5,
        )


def test_top_p_validation() -> None:
    """Test top_p field validation constraints."""
    # Valid top_p
    request = CreateResponseBody(
        model="gpt-4",
        input="Hello!",
        top_p=0.9,
    )
    assert request.top_p == 0.9

    # top_p too high should fail
    with pytest.raises(ValidationError):
        CreateResponseBody(
            model="gpt-4",
            input="Hello!",
            top_p=1.5,
        )


def test_json_serialization() -> None:
    """Test that models can be serialized to JSON."""
    request = CreateResponseBody(
        model="gpt-4",
        input=[
            UserMessage(
                role="user",
                content="Hello!",
            )
        ],
        temperature=0.7,
    )

    # Serialize to JSON
    json_str = request.model_dump_json()
    json_data = json.loads(json_str)

    assert json_data["model"] == "gpt-4"
    assert json_data["temperature"] == 0.7
    assert len(json_data["input"]) == 1
    assert json_data["input"][0]["role"] == "user"


def test_json_deserialization() -> None:
    """Test that models can be deserialized from JSON."""
    json_data = {
        "model": "gpt-4",
        "input": "Hello!",
        "temperature": 0.8,
        "max_output_tokens": 500,
    }

    request = CreateResponseBody(**json_data)

    assert request.model == "gpt-4"
    assert request.input == "Hello!"
    assert request.temperature == 0.8
    assert request.max_output_tokens == 500


def test_output_text_content_with_annotations() -> None:
    """Test OutputTextContent with annotations."""
    content = OutputTextContent(
        type="output_text",
        text="The weather in San Francisco is 65°F.",
        annotations=[0, 1],
    )

    assert content.type == "output_text"
    assert content.text == "The weather in San Francisco is 65°F."
    assert content.annotations == [0, 1]
