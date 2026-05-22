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

"""Tests for the Kimi K2.5 tokenizer, specifically tool handling."""

from unittest.mock import MagicMock, patch

import pytest
from max.pipelines.architectures.kimik2_5.tokenizer import KimiK2_5VLTokenizer
from max.pipelines.modeling.types import (
    TextGenerationRequestMessage,
    TextGenerationRequestTool,
)


class TestApplyChatTemplateWithTools:
    """Tests for apply_chat_template with tool definitions."""

    @pytest.fixture
    def mock_delegate(self) -> MagicMock:
        """Create a mock HuggingFace tokenizer delegate."""
        delegate = MagicMock()
        delegate.apply_chat_template = MagicMock(
            return_value="templated_output"
        )
        return delegate

    @pytest.fixture
    def tokenizer_with_mock(
        self, mock_delegate: MagicMock
    ) -> KimiK2_5VLTokenizer:
        """Create a KimiK2_5VLTokenizer with a mocked delegate."""
        with patch.object(
            KimiK2_5VLTokenizer, "__init__", lambda self, *args, **kwargs: None
        ):
            tokenizer = KimiK2_5VLTokenizer.__new__(KimiK2_5VLTokenizer)
            tokenizer.delegate = mock_delegate
            return tokenizer

    def test_apply_chat_template_passes_tools_to_delegate(
        self, tokenizer_with_mock: KimiK2_5VLTokenizer, mock_delegate: MagicMock
    ) -> None:
        """Test that tools are passed to the delegate's apply_chat_template."""
        messages = [
            TextGenerationRequestMessage(
                role="user", content="Search for Python"
            )
        ]
        tools = [
            TextGenerationRequestTool(
                type="function",
                function={
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            )
        ]

        result = tokenizer_with_mock.apply_chat_template(messages, tools)

        # Verify the delegate was called with tools
        mock_delegate.apply_chat_template.assert_called_once()
        call_kwargs = mock_delegate.apply_chat_template.call_args.kwargs
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] == tools
        assert result == "templated_output"

    def test_apply_chat_template_without_tools(
        self, tokenizer_with_mock: KimiK2_5VLTokenizer, mock_delegate: MagicMock
    ) -> None:
        """Test apply_chat_template works when tools is None."""
        messages = [TextGenerationRequestMessage(role="user", content="Hello")]

        result = tokenizer_with_mock.apply_chat_template(messages, tools=None)

        mock_delegate.apply_chat_template.assert_called_once()
        call_kwargs = mock_delegate.apply_chat_template.call_args.kwargs
        assert call_kwargs["tools"] is None
        assert result == "templated_output"

    def test_apply_chat_template_with_empty_tools(
        self, tokenizer_with_mock: KimiK2_5VLTokenizer, mock_delegate: MagicMock
    ) -> None:
        """Test apply_chat_template with empty tools list."""
        messages = [TextGenerationRequestMessage(role="user", content="Hello")]
        tools: list[TextGenerationRequestTool] = []

        result = tokenizer_with_mock.apply_chat_template(messages, tools)

        mock_delegate.apply_chat_template.assert_called_once()
        call_kwargs = mock_delegate.apply_chat_template.call_args.kwargs
        assert call_kwargs["tools"] == []
        assert result == "templated_output"

    def test_apply_chat_template_with_multiple_tools(
        self, tokenizer_with_mock: KimiK2_5VLTokenizer, mock_delegate: MagicMock
    ) -> None:
        """Test apply_chat_template with multiple tool definitions."""
        messages = [
            TextGenerationRequestMessage(
                role="user", content="Get weather and search"
            )
        ]
        tools = [
            TextGenerationRequestTool(
                type="function",
                function={
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            ),
            TextGenerationRequestTool(
                type="function",
                function={
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            ),
        ]

        result = tokenizer_with_mock.apply_chat_template(messages, tools)

        mock_delegate.apply_chat_template.assert_called_once()
        call_kwargs = mock_delegate.apply_chat_template.call_args.kwargs
        assert call_kwargs["tools"] == tools
        assert len(call_kwargs["tools"]) == 2
        assert result == "templated_output"

    def test_apply_chat_template_sets_add_generation_prompt(
        self, tokenizer_with_mock: KimiK2_5VLTokenizer, mock_delegate: MagicMock
    ) -> None:
        """Test that add_generation_prompt=True is always set."""
        messages = [TextGenerationRequestMessage(role="user", content="Hello")]

        tokenizer_with_mock.apply_chat_template(messages)

        call_kwargs = mock_delegate.apply_chat_template.call_args.kwargs
        assert call_kwargs["add_generation_prompt"] is True

    def test_apply_chat_template_sets_tokenize_false(
        self, tokenizer_with_mock: KimiK2_5VLTokenizer, mock_delegate: MagicMock
    ) -> None:
        """Test that tokenize=False is set (returns string, not token IDs)."""
        messages = [TextGenerationRequestMessage(role="user", content="Hello")]

        tokenizer_with_mock.apply_chat_template(messages)

        call_kwargs = mock_delegate.apply_chat_template.call_args.kwargs
        assert call_kwargs["tokenize"] is False

    def test_apply_chat_template_forwards_chat_template_options(
        self, tokenizer_with_mock: KimiK2_5VLTokenizer, mock_delegate: MagicMock
    ) -> None:
        """Test that chat_template_options are forwarded to the delegate.

        This is a regression test for MXSERV-79: chat_template_kwargs from
        the request must be forwarded to the Jinja template. Without this
        fix, options like ``{"thinking": false}`` were silently dropped.
        """
        messages = [TextGenerationRequestMessage(role="user", content="Hello")]

        tokenizer_with_mock.apply_chat_template(
            messages, tools=None, thinking=False
        )

        call_kwargs = mock_delegate.apply_chat_template.call_args.kwargs
        # The "thinking" option should be forwarded to the delegate
        assert "thinking" in call_kwargs
        assert call_kwargs["thinking"] is False
        # add_generation_prompt should still be set
        assert call_kwargs["add_generation_prompt"] is True

    def test_apply_chat_template_options_do_not_override_add_generation_prompt(
        self, tokenizer_with_mock: KimiK2_5VLTokenizer, mock_delegate: MagicMock
    ) -> None:
        """Test that caller options can override add_generation_prompt if needed."""
        messages = [TextGenerationRequestMessage(role="user", content="Hello")]

        # Caller explicitly sets add_generation_prompt=False via kwargs
        tokenizer_with_mock.apply_chat_template(
            messages, tools=None, add_generation_prompt=False, thinking=True
        )

        call_kwargs = mock_delegate.apply_chat_template.call_args.kwargs
        # Caller's setting should override the default
        assert call_kwargs["add_generation_prompt"] is False
        assert call_kwargs["thinking"] is True

    def test_apply_chat_template_with_no_extra_options(
        self, tokenizer_with_mock: KimiK2_5VLTokenizer, mock_delegate: MagicMock
    ) -> None:
        """Test apply_chat_template with no extra chat_template_options."""
        messages = [TextGenerationRequestMessage(role="user", content="Hello")]

        tokenizer_with_mock.apply_chat_template(messages, tools=None)

        call_kwargs = mock_delegate.apply_chat_template.call_args.kwargs
        # Only default options should be set
        assert call_kwargs["add_generation_prompt"] is True
        assert call_kwargs["tokenize"] is False
        # "thinking" should not be in kwargs if not provided
        assert "thinking" not in call_kwargs
