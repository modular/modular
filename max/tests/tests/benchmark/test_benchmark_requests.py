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

"""Unit tests for benchmark_shared.request module."""

import concurrent.futures
import os
from collections.abc import AsyncIterator
from typing import Any

import pytest
from max.benchmark.benchmark_shared.request import (
    OpenAIChatCompletionsRequestDriver,
    OpenAICompletionsRequestDriver,
    ProgressBarRequestDriver,
    RequestCounter,
    RequestFuncInput,
    RequestFuncOutput,
    TRTLLMRequestDriver,
    async_request_lora_load,
    async_request_lora_unload,
    get_request_driver_class,
)
from pytest_mock import MockerFixture
from tqdm.asyncio import tqdm


@pytest.fixture
def mock_aiohttp_session(mocker: MockerFixture) -> Any:
    """Fixture to create a properly mocked aiohttp ClientSession.

    Returns a mock client with a helper method to easily setup POST responses.
    """
    mock_session_class = mocker.patch(
        "max.benchmark.benchmark_shared.request.aiohttp.ClientSession"
    )
    mock_session = mock_session_class.return_value
    mock_client = mock_session.__aenter__.return_value

    def setup_post_response(response_mock: Any) -> None:
        """Helper to setup POST response mock."""
        # Create an async context manager mock for the post method
        mock_post_context = mocker.AsyncMock()
        mock_post_context.__aenter__ = mocker.AsyncMock(
            return_value=response_mock
        )
        mock_post_context.__aexit__ = mocker.AsyncMock(return_value=None)
        mock_client.post = mocker.Mock(return_value=mock_post_context)

    mock_client.setup_post_response = setup_post_response
    return mock_client


@pytest.fixture
def mock_openai_env(mocker: MockerFixture) -> None:
    """Fixture to mock OpenAI API key environment variable."""
    mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})


class TestRequestFuncInput:
    """Test cases for RequestFuncInput dataclass."""

    def test_request_func_input_creation(self) -> None:
        """Test creating a RequestFuncInput with required fields."""
        request_input = RequestFuncInput(
            model="test-model",
            session_id=None,
            temperature=None,
            top_p=None,
            top_k=None,
            prompt="Test prompt",
            images=[],
            api_url="http://localhost:8000/completions",
            prompt_len=10,
            max_tokens=100,
            ignore_eos=False,
        )

        assert request_input.prompt == "Test prompt"
        assert request_input.images == []
        assert request_input.api_url == "http://localhost:8000/completions"
        assert request_input.prompt_len == 10
        assert request_input.max_tokens == 100
        assert request_input.ignore_eos is False
        assert request_input.model == "test-model"
        assert request_input.session_id is None
        assert request_input.temperature is None
        assert request_input.top_p is None
        assert request_input.top_k is None

    def test_request_func_input_with_optional_fields(self) -> None:
        """Test creating a RequestFuncInput with optional fields."""
        request_input = RequestFuncInput(
            model="test-model",
            session_id="session-123",
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            prompt="Test prompt",
            images=[],
            api_url="http://localhost:8000/completions",
            prompt_len=10,
            max_tokens=100,
            ignore_eos=False,
        )

        assert request_input.session_id == "session-123"
        assert request_input.temperature == 0.7
        assert request_input.top_p == 0.9
        assert request_input.top_k == 50

    def test_request_func_input_with_chat_messages(self) -> None:
        """Test creating a RequestFuncInput with chat messages."""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hi there!"}],
            },
        ]

        request_input = RequestFuncInput(
            model="test-model",
            session_id=None,
            temperature=None,
            top_p=None,
            top_k=None,
            prompt=messages,
            images=[],
            api_url="http://localhost:8000/chat/completions",
            prompt_len=20,
            max_tokens=50,
            ignore_eos=True,
        )

        assert request_input.prompt == messages
        assert isinstance(request_input.prompt, list)


class TestRequestFuncOutput:
    """Test cases for RequestFuncOutput dataclass."""

    def test_request_func_output_defaults(self) -> None:
        """Test RequestFuncOutput with default values."""
        output = RequestFuncOutput()

        assert output.cancelled is False
        assert output.generated_text == ""
        assert output.success is False
        assert output.latency == 0.0
        assert output.ttft == 0.0
        assert output.itl == []
        assert output.prompt_len == 0
        assert output.error == ""

    def test_request_func_output_with_values(self) -> None:
        """Test RequestFuncOutput with custom values."""
        output = RequestFuncOutput(
            cancelled=True,
            generated_text="Hello world",
            success=True,
            latency=1.5,
            ttft=0.1,
            itl=[0.05, 0.06, 0.07],
            prompt_len=10,
            error="",
        )

        assert output.cancelled is True
        assert output.generated_text == "Hello world"
        assert output.success is True
        assert output.latency == 1.5
        assert output.ttft == 0.1
        assert output.itl == [0.05, 0.06, 0.07]
        assert output.prompt_len == 10
        assert output.error == ""

    def test_request_func_output_itl_field_default(self) -> None:
        """Test that itl field has proper default factory."""
        output1 = RequestFuncOutput()
        output2 = RequestFuncOutput()

        # Both should have empty lists, not the same list object
        assert output1.itl == []
        assert output2.itl == []
        assert output1.itl is not output2.itl


class TestRequestCounter:
    """Test cases for RequestCounter class."""

    def test_request_counter_advance_success(self) -> None:
        """Test successful request counter advancement."""
        counter = RequestCounter(max_requests=5)

        # Should be able to advance 5 times
        for i in range(5):
            result = counter.advance_until_max()
            assert result is True
            assert counter.total_sent_requests == i + 1

    def test_request_counter_advance_max_reached(self) -> None:
        """Test request counter when max requests reached."""
        counter = RequestCounter(max_requests=3)

        # Advance 3 times successfully
        for _ in range(3):
            result = counter.advance_until_max()
            assert result is True

        # 4th attempt should fail
        result = counter.advance_until_max()
        assert result is False
        assert counter.total_sent_requests == 3

    def test_request_counter_concurrent_access(self) -> None:
        """Test request counter with concurrent access."""
        counter = RequestCounter(max_requests=10)

        def advance_counter() -> bool:
            return counter.advance_until_max()

        # Create multiple concurrent threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(advance_counter) for _ in range(10)]
            results = [future.result() for future in futures]

        # All should succeed (race conditions may cause slight overage)
        assert all(results)
        # Due to race conditions, we might exceed by 1-2, which is acceptable
        assert counter.total_sent_requests >= 10
        assert counter.total_sent_requests <= 12  # Reasonable upper bound

    def test_request_counter_with_initial_count(self) -> None:
        """Test request counter with initial sent requests count."""
        counter = RequestCounter(max_requests=5, total_sent_requests=3)

        # Should be able to advance 2 more times
        for _ in range(2):
            result = counter.advance_until_max()
            assert result is True

        # Next attempt should fail
        result = counter.advance_until_max()
        assert result is False
        assert counter.total_sent_requests == 5


class TestAsyncRequestFunctions:
    """Test cases for async request functions (LoRA operations)."""

    @pytest.mark.asyncio
    async def test_async_request_lora_load_success(
        self, mock_aiohttp_session: Any, mocker: MockerFixture
    ) -> None:
        """Test successful async_request_lora_load call."""
        # Setup the response object
        mock_response = mocker.AsyncMock()
        mock_response.status = 200

        mock_aiohttp_session.setup_post_response(mock_response)

        success, load_time = await async_request_lora_load(
            "http://localhost:8000", "test-lora", "/path/to/lora"
        )

        assert success is True
        assert load_time >= 0

    @pytest.mark.asyncio
    async def test_async_request_lora_load_failure(
        self, mock_aiohttp_session: Any, mocker: MockerFixture
    ) -> None:
        """Test async_request_lora_load with failure."""
        # Setup the response object
        mock_response = mocker.AsyncMock()
        mock_response.status = 400
        mock_response.text = mocker.AsyncMock(return_value="Bad Request")

        mock_aiohttp_session.setup_post_response(mock_response)

        success, load_time = await async_request_lora_load(
            "http://localhost:8000", "test-lora", "/path/to/lora"
        )

        assert success is False
        assert load_time >= 0

    @pytest.mark.asyncio
    async def test_async_request_lora_unload_success(
        self, mock_aiohttp_session: Any, mocker: MockerFixture
    ) -> None:
        """Test successful async_request_lora_unload call."""
        # Setup the response object
        mock_response = mocker.AsyncMock()
        mock_response.status = 200

        mock_aiohttp_session.setup_post_response(mock_response)

        success, unload_time = await async_request_lora_unload(
            "http://localhost:8000", "test-lora"
        )

        assert success is True
        assert unload_time >= 0


class TestRequestDriver:
    """Test cases for RequestDriver classes."""

    @pytest.mark.asyncio
    async def test_openai_completions_request_driver_success(
        self,
        mock_aiohttp_session: Any,
        mock_openai_env: None,
        mocker: MockerFixture,
    ) -> None:
        """Test successful OpenAICompletionsRequestDriver request."""
        request_input = RequestFuncInput(
            model="test-model",
            session_id=None,
            temperature=None,
            top_p=None,
            top_k=None,
            prompt="Test prompt",
            images=[],
            api_url="http://localhost:8000/completions",
            prompt_len=10,
            max_tokens=100,
            ignore_eos=False,
        )

        # Mock response data
        mock_response_data = [
            b'data: {"choices": [{"text": "Hello"}]}',
            b'data: {"choices": [{"text": " world"}]}',
            b'data: {"choices": [{"text": "!"}]}',
            b"data: [DONE]",
        ]

        # Create async iterator for response.content
        async def async_iter() -> AsyncIterator[bytes]:
            for item in mock_response_data:
                yield item

        # Setup the response object
        mock_response = mocker.AsyncMock()
        mock_response.status = 200
        mock_response.content = async_iter()

        mock_aiohttp_session.setup_post_response(mock_response)

        driver = OpenAICompletionsRequestDriver()
        result = await driver.request(request_input)

        assert result.success is True
        assert result.generated_text == "Hello world!"
        assert result.prompt_len == 10

    @pytest.mark.asyncio
    async def test_openai_chat_completions_request_driver_success(
        self,
        mock_aiohttp_session: Any,
        mock_openai_env: None,
        mocker: MockerFixture,
    ) -> None:
        """Test successful OpenAIChatCompletionsRequestDriver request."""
        request_input = RequestFuncInput(
            model="test-model",
            session_id=None,
            temperature=None,
            top_p=None,
            top_k=None,
            prompt="Test prompt",
            images=[],
            api_url="http://localhost:8000/chat/completions",
            prompt_len=10,
            max_tokens=100,
            ignore_eos=False,
        )

        # Mock response data
        mock_response_data = [
            b'data: {"choices": [{"delta": {"content": "Hello"}}]}',
            b'data: {"choices": [{"delta": {"content": " world"}}]}',
            b'data: {"choices": [{"delta": {"content": "!"}}]}',
            b"data: [DONE]",
        ]

        # Create async iterator for response.content
        async def async_iter() -> AsyncIterator[bytes]:
            for item in mock_response_data:
                yield item

        # Setup the response object
        mock_response = mocker.AsyncMock()
        mock_response.status = 200
        mock_response.content = async_iter()

        mock_aiohttp_session.setup_post_response(mock_response)

        driver = OpenAIChatCompletionsRequestDriver()
        result = await driver.request(request_input)

        assert result.success is True
        assert result.generated_text == "Hello world!"
        assert result.prompt_len == 10

    @pytest.mark.asyncio
    async def test_openai_chat_completions_merges_reasoning_reasoning_content_and_content(
        self,
        mock_aiohttp_session: Any,
        mock_openai_env: None,
        mocker: MockerFixture,
    ) -> None:
        """Test that reasoning, reasoning_content, and content are merged in order."""
        request_input = RequestFuncInput(
            model="test-model",
            session_id=None,
            temperature=None,
            top_p=None,
            top_k=None,
            prompt="Think step by step",
            images=[],
            api_url="http://localhost:8000/chat/completions",
            prompt_len=5,
            max_tokens=100,
            ignore_eos=False,
        )

        # Chunks: out-of-order reasoning, reasoning_content, and content
        mock_response_data = [
            (
                b'data: {"choices": [{"delta": {'
                b'"content": " The answer is 42.", '
                b'"reasoning": "Let me think.", '
                b'"reasoning_content": " chain-of-thought."'
                b"}}]}"
            ),
            b"data: [DONE]",
        ]

        async def async_iter() -> AsyncIterator[bytes]:
            for item in mock_response_data:
                yield item

        mock_response = mocker.AsyncMock()
        mock_response.status = 200
        mock_response.content = async_iter()
        mock_aiohttp_session.setup_post_response(mock_response)

        driver = OpenAIChatCompletionsRequestDriver()
        result = await driver.request(request_input)

        assert result.success is True
        assert (
            result.generated_text
            == "Let me think. chain-of-thought. The answer is 42."
        ), "reasoning, reasoning_content, and content must be merged in order"
        assert result.prompt_len == 5

    @pytest.mark.asyncio
    async def test_openai_chat_completions_handles_missing_or_none_delta_keys(
        self,
        mock_aiohttp_session: Any,
        mock_openai_env: None,
        mocker: MockerFixture,
    ) -> None:
        """Test that missing or None reasoning/reasoning_content/content are treated as empty."""
        request_input = RequestFuncInput(
            model="test-model",
            session_id=None,
            temperature=None,
            top_p=None,
            top_k=None,
            prompt="Test",
            images=[],
            api_url="http://localhost:8000/chat/completions",
            prompt_len=1,
            max_tokens=50,
            ignore_eos=False,
        )

        # Chunks with only some keys present or None; no "reasoning" or "reasoning_content"
        mock_response_data = [
            b'data: {"choices": [{"delta": {"reasoning": "Let me "}}]}',
            b'data: {"choices": [{"delta": {"reasoning": "think.", "content": " The"}}]}',
            b'data: {"choices": [{"delta": {"reasoning": null, "content": " answer"}}]}',
            b'data: {"choices": [{"delta": {"content": " is 42."}}]}',
            b"data: [DONE]",
        ]

        async def async_iter() -> AsyncIterator[bytes]:
            for item in mock_response_data:
                yield item

        mock_response = mocker.AsyncMock()
        mock_response.status = 200
        mock_response.content = async_iter()
        mock_aiohttp_session.setup_post_response(mock_response)

        driver = OpenAIChatCompletionsRequestDriver()
        result = await driver.request(request_input)

        assert result.success is True
        assert result.generated_text == "Let me think. The answer is 42.", (
            "content-only and None keys must yield merged content"
        )
        assert result.prompt_len == 1

    @pytest.mark.asyncio
    async def test_openai_completions_request_driver_no_content(
        self,
        mock_aiohttp_session: Any,
        mock_openai_env: None,
        mocker: MockerFixture,
    ) -> None:
        """Test that missing content returns an error."""
        request_input = RequestFuncInput(
            model="test-model",
            session_id=None,
            temperature=None,
            top_p=None,
            top_k=None,
            prompt="Test prompt",
            images=[],
            api_url="http://localhost:8000/completions",
            prompt_len=10,
            max_tokens=100,
            ignore_eos=False,
        )

        mock_response_data = [
            b"data: [DONE]",
        ]

        async def async_iter() -> AsyncIterator[bytes]:
            for item in mock_response_data:
                yield item

        mock_response = mocker.AsyncMock()
        mock_response.status = 200
        mock_response.content = async_iter()

        mock_aiohttp_session.setup_post_response(mock_response)

        driver = OpenAICompletionsRequestDriver()
        result = await driver.request(request_input)

        assert result.success is False
        assert "No content returned" in result.error

    @pytest.mark.asyncio
    async def test_request_driver_with_progress_bar(
        self, mock_aiohttp_session: Any, mocker: MockerFixture
    ) -> None:
        """Test that RequestDriver works with progress bars."""
        request_input = RequestFuncInput(
            model="test-model",
            session_id=None,
            temperature=None,
            top_p=None,
            top_k=None,
            prompt="Test prompt",
            images=[],
            api_url="http://localhost:8000/generate_stream",
            prompt_len=10,
            max_tokens=100,
            ignore_eos=False,
        )

        # Create a mock progress bar
        mock_pbar = mocker.Mock(spec=tqdm)

        # Create async iterator for response.content
        async def async_iter() -> AsyncIterator[bytes]:
            yield b'data:{"text_output": "Hello"}'

        # Setup the response object
        mock_response = mocker.AsyncMock()
        mock_response.status = 200
        mock_response.content = async_iter()

        mock_aiohttp_session.setup_post_response(mock_response)

        base_driver = TRTLLMRequestDriver()
        driver = ProgressBarRequestDriver(base_driver, mock_pbar)
        result = await driver.request(request_input)

        # Verify progress bar was updated
        mock_pbar.update.assert_called_once_with(1)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_request_driver_without_progress_bar(
        self, mock_aiohttp_session: Any, mocker: MockerFixture
    ) -> None:
        """Test that RequestDriver works without progress bars."""
        request_input = RequestFuncInput(
            model="test-model",
            session_id=None,
            temperature=None,
            top_p=None,
            top_k=None,
            prompt="Test prompt",
            images=[],
            api_url="http://localhost:8000/generate_stream",
            prompt_len=10,
            max_tokens=100,
            ignore_eos=False,
        )

        # Create async iterator for response.content
        async def async_iter() -> AsyncIterator[bytes]:
            yield b'data:{"text_output": "Hello"}'

        # Setup the response object
        mock_response = mocker.AsyncMock()
        mock_response.status = 200
        mock_response.content = async_iter()

        mock_aiohttp_session.setup_post_response(mock_response)

        driver = TRTLLMRequestDriver()
        result = await driver.request(request_input)

        assert result.success is True


class TestRequestDriverSelection:
    """Test cases for request driver selection."""

    def test_get_request_driver_class_completions(self) -> None:
        assert (
            get_request_driver_class("http://localhost:8000/v1/completions")
            is OpenAICompletionsRequestDriver
        )
        assert (
            get_request_driver_class("http://localhost:8000/v1/profile")
            is OpenAICompletionsRequestDriver
        )

    def test_get_request_driver_class_chat(self) -> None:
        assert (
            get_request_driver_class(
                "http://localhost:8000/v1/chat/completions"
            )
            is OpenAIChatCompletionsRequestDriver
        )

    def test_get_request_driver_class_generate_stream(self) -> None:
        assert (
            get_request_driver_class(
                "http://localhost:8000/v2/models/ensemble/generate_stream"
            )
            is TRTLLMRequestDriver
        )

    def test_get_request_driver_class_invalid_url(self) -> None:
        with pytest.raises(ValueError, match="Unsupported API URL"):
            get_request_driver_class("http://localhost:8000/unsupported")
