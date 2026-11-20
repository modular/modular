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

import asyncio
import concurrent.futures
import json
import logging
import os
import time
from array import array
from collections.abc import AsyncIterator, Sequence
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from max.benchmark.benchmark_serving import validate_task_and_endpoint
from max.benchmark.benchmark_shared.config import (
    BenchmarkTask,
    Endpoint,
    SamplingConfig,
    ServingBenchmarkConfig,
)
from max.benchmark.benchmark_shared.datasets.types import (
    ChatMessage,
    ImageContentBlock,
    ImageURLDetail,
    OpenAIImage,
    PixelGenerationImageOptions,
    ResponseFormat,
    TextContentBlock,
)
from max.benchmark.benchmark_shared.request import (
    OpenAIChatCompletionsRequestDriver,
    OpenAICompletionsRequestDriver,
    OpenAIStreamRecorder,
    OpenResponsesRequestDriver,
    PixelGenerationRequestFuncInput,
    ProgressBarRequestDriver,
    RequestCounter,
    RequestDriver,
    RequestFuncInput,
    RequestFuncOutput,
    SglangPixelGenerationRequestDriver,
    SglangVideoPayload,
    SglangVideoRequestDriver,
    TRTLLMRequestDriver,
    VllmOmniPixelGenerationRequestDriver,
    VllmOmniVideoRequestDriver,
    _attach_images_to_first_user_message,
    _build_final_payload,
    _build_sglang_pixel_generation_payload,
    _build_sglang_video_payload,
    _build_vllm_omni_pixel_generation_payload,
    _build_vllm_omni_video_payload,
    _count_generated_media,
    async_request_lora_load,
    async_request_lora_unload,
    compute_chunk_tpots,
    get_request_driver_class,
    mark_cancelled_if_past_deadline,
    tag_lora_route,
    tag_response_format_outcome,
)
from pytest_mock import MockerFixture
from tests.benchmark.benchmark_fakes import StreamBody, make_request_input
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
            sampling=SamplingConfig(),
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
        assert request_input.sampling.temperature is None
        assert request_input.sampling.top_p is None
        assert request_input.sampling.top_k is None

    def test_request_func_input_with_optional_fields(self) -> None:
        """Test creating a RequestFuncInput with optional fields."""
        request_input = RequestFuncInput(
            model="test-model",
            session_id="session-123",
            sampling=SamplingConfig(temperature=0.7, top_p=0.9, top_k=50),
            prompt="Test prompt",
            images=[],
            api_url="http://localhost:8000/completions",
            prompt_len=10,
            max_tokens=100,
            ignore_eos=False,
        )

        assert request_input.session_id == "session-123"
        assert request_input.sampling.temperature == 0.7
        assert request_input.sampling.top_p == 0.9
        assert request_input.sampling.top_k == 50

    def test_request_func_input_with_chat_messages(self) -> None:
        """Test creating a RequestFuncInput with chat messages."""
        messages = [
            ChatMessage(
                role="user",
                content=[TextContentBlock(type="text", text="Hello")],
            ),
            ChatMessage(
                role="assistant",
                content=[TextContentBlock(type="text", text="Hi there!")],
            ),
        ]

        request_input = RequestFuncInput(
            model="test-model",
            session_id=None,
            sampling=SamplingConfig(),
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


class _WordCountingTokenizer:
    """Counts whitespace-separated words and records every call it receives."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(
        self, texts: str | Sequence[str], add_special_tokens: bool = True
    ) -> SimpleNamespace:
        batch = [texts] if isinstance(texts, str) else list(texts)
        self.calls.append(batch)
        return SimpleNamespace(
            input_ids=[[0] * len(text.split()) for text in batch]
        )


def _text_event(content: str) -> str:
    """Returns a streamed text chunk carrying ``content``."""
    return json.dumps({"choices": [{"delta": {"content": content}}]})


def _tpot_output(
    text: str, *, itl: Sequence[float], ends: Sequence[int]
) -> RequestFuncOutput:
    """Returns an output carrying recorded chunk offsets, as a stream does."""
    return RequestFuncOutput(
        success=True,
        generated_text=text,
        itl=list(itl),
        pending_tpot_ends=array("I", ends),
    )


class TestOpenAIStreamRecorder:
    def test_records_ttft_itl_and_generated_text_for_chat_chunks(self) -> None:
        """The first content chunk sets ttft; later chunks add itl entries."""
        recorder = OpenAIStreamRecorder(prompt_len=10, chat=True)

        assert (
            recorder.on_event(
                json.dumps({"choices": [{"delta": {"content": "Hello"}}]})
            )
            is None
        )
        assert recorder.output.ttft > 0.0
        assert recorder.output.itl == []

        assert (
            recorder.on_event(
                json.dumps({"choices": [{"delta": {"content": " world"}}]})
            )
            is None
        )
        assert len(recorder.output.itl) == 1

        assert recorder.on_event("[DONE]") is None

        output = recorder.finish()
        assert output is recorder.output
        assert output.success is True
        assert output.generated_text == "Hello world"
        assert output.latency > 0.0
        assert output.prompt_len == 10

    def test_records_text_for_completion_chunks(self) -> None:
        """``chat=False`` reads the completions schema's ``text`` field."""
        recorder = OpenAIStreamRecorder(prompt_len=5, chat=False)

        recorder.on_event(json.dumps({"choices": [{"text": "Hi"}]}))
        output = recorder.finish()

        assert output.success is True
        assert output.generated_text == "Hi"

    def test_on_event_returns_output_and_stops_on_error_chunk(self) -> None:
        """An error chunk ends the stream immediately with the failed output."""
        recorder = OpenAIStreamRecorder(prompt_len=1, chat=True)

        result = recorder.on_event(
            json.dumps({"choices": [], "error": {"message": "boom"}})
        )

        assert result is recorder.output
        assert result.success is False
        assert result.error == "boom"

    def test_an_empty_error_object_still_fails_the_request(self) -> None:
        """``{"error": {}}`` fails the request, as it did before."""
        recorder = OpenAIStreamRecorder(prompt_len=1, chat=True)

        result = recorder.on_event(json.dumps({"error": {}}))

        assert result is not None
        assert result.success is False
        assert '{"error": {}}' in result.error

    @pytest.mark.parametrize(
        "data",
        [
            pytest.param("null", id="null"),
            pytest.param("5", id="number"),
            pytest.param("[1]", id="array"),
        ],
    )
    def test_a_chunk_that_is_not_an_object_fails_with_its_payload(
        self, data: str
    ) -> None:
        """A non-object chunk fails at once, quoting the bytes that broke it."""
        recorder = OpenAIStreamRecorder(prompt_len=1, chat=True)

        with pytest.raises(TypeError, match="expected a JSON object") as raised:
            recorder.on_event(data)

        assert repr(data) in str(raised.value)

    @pytest.mark.parametrize(
        ("data", "why"),
        [
            pytest.param(
                '{"choices": [{"content": "hi"}]}',
                "carry a delta object",
                id="choice_without_delta",
            ),
            pytest.param(
                '{"choices": [{"delta": {"content": {"a": 1}}}]}',
                "delta.content",
                id="object_content",
            ),
            pytest.param(
                '{"choices": [{"delta": {"content": 7}}]}',
                "delta.content",
                id="number_content",
            ),
            pytest.param(
                '{"choices": {}}',
                "choices to be a JSON array",
                id="choices_object",
            ),
            pytest.param(
                '{"choices": 5}',
                "choices to be a JSON array",
                id="choices_number",
            ),
            pytest.param(
                '{"choices": "hi"}',
                "choices to be a JSON array",
                id="choices_string",
            ),
        ],
    )
    def test_a_bad_choice_fails_at_that_chunk(
        self, data: str, why: str
    ) -> None:
        """A shape the client cannot read fails there, naming the field."""
        recorder = OpenAIStreamRecorder(prompt_len=1, chat=True)

        with pytest.raises(TypeError, match=why) as raised:
            recorder.on_event(data)

        assert repr(data) in str(raised.value)

    def test_a_non_string_completion_text_fails_at_that_chunk(self) -> None:
        """``chat=False`` rejects a non-string ``text`` the same way."""
        recorder = OpenAIStreamRecorder(prompt_len=1, chat=False)
        data = '{"choices": [{"text": {"a": 1}}]}'

        with pytest.raises(TypeError, match=r"choice\.text") as raised:
            recorder.on_event(data)

        assert repr(data) in str(raised.value)

    @pytest.mark.parametrize(
        "data",
        [
            pytest.param('{"choices": [{"text": null}]}', id="null"),
            pytest.param(
                '{"choices": [{"finish_reason": "stop"}]}', id="missing"
            ),
        ],
    )
    def test_a_completion_chunk_without_text_fails_at_that_chunk(
        self, data: str
    ) -> None:
        """A completion's ``text`` is the response text, so null is not empty.

        A chat delta may carry a null field; the completions schema has no such
        option, and reading null as ``""`` hid a chunk schema the client does
        not understand.
        """
        recorder = OpenAIStreamRecorder(prompt_len=1, chat=False)

        with pytest.raises(TypeError, match=r"choice\.text") as raised:
            recorder.on_event(data)

        assert repr(data) in str(raised.value)

    def test_a_bool_usage_count_is_rejected(self) -> None:
        """``true`` in a count field is a schema change, not a count of one."""
        recorder = OpenAIStreamRecorder(prompt_len=1, chat=True)

        with pytest.raises(TypeError, match="not an integer"):
            recorder.on_event(
                json.dumps(
                    {
                        "choices": [{"delta": {"content": "x"}}],
                        "usage": {"completion_tokens": True},
                    }
                )
            )

    @pytest.mark.parametrize(
        ("usage", "expected_cached_tokens", "details_reported"),
        [
            pytest.param(
                {
                    "prompt_tokens": 3,
                    "completion_tokens": 1,
                    "total_tokens": 4,
                    "prompt_tokens_details": {"cached_tokens": 2},
                },
                2,
                True,
                id="with_cached_tokens",
            ),
            pytest.param(
                {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
                0,
                False,
                id="without_prompt_tokens_details",
            ),
            pytest.param(
                {
                    "prompt_tokens": "3",
                    "completion_tokens": 1.0,
                    "total_tokens": "4",
                    "prompt_tokens_details": {"cached_tokens": "2"},
                },
                2,
                True,
                id="lax_numeric_forms",
            ),
            pytest.param(
                {
                    "prompt_tokens": "3.0",
                    "completion_tokens": "1.0",
                    "total_tokens": 4,
                    "prompt_tokens_details": {"cached_tokens": 2},
                },
                2,
                True,
                id="integral_float_strings",
            ),
            # A server that reports the field as null still reported the
            # details, so the run is not read as one with no cache lookups.
            pytest.param(
                {
                    "prompt_tokens": 3,
                    "completion_tokens": 1,
                    "total_tokens": 4,
                    "prompt_tokens_details": {"cached_tokens": None},
                },
                0,
                True,
                id="null_cached_tokens",
            ),
        ],
    )
    def test_usage_chunk_is_recorded_as_server_token_stats(
        self,
        usage: dict[str, Any],
        expected_cached_tokens: int,
        details_reported: bool,
    ) -> None:
        """A usage-bearing chunk populates ``server_token_stats``."""
        recorder = OpenAIStreamRecorder(prompt_len=1, chat=True)

        recorder.on_event(
            json.dumps(
                {"choices": [{"delta": {"content": "x"}}], "usage": usage}
            )
        )

        stats = recorder.output.server_token_stats
        assert stats.prompt_tokens == 3
        assert stats.completion_tokens == 1
        assert stats.total_tokens == 4
        assert stats.cached_tokens == expected_cached_tokens
        assert stats.cached_tokens_reported is details_reported
        # The counts feed integer arithmetic after the run, so a bool or a
        # numeric string must not reach the metrics as itself.
        assert isinstance(stats.completion_tokens, int)
        assert not isinstance(stats.completion_tokens, bool)

    def test_usage_without_prompt_details_reports_no_cached_tokens(
        self,
    ) -> None:
        """A server that omits ``prompt_tokens_details`` reports nothing."""
        recorder = OpenAIStreamRecorder(prompt_len=1, chat=True)

        recorder.on_event(
            json.dumps(
                {
                    "choices": [{"delta": {"content": "x"}}],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 1,
                        "total_tokens": 4,
                    },
                }
            )
        )

        stats = recorder.output.server_token_stats
        assert stats.cached_tokens == 0
        assert not stats.cached_tokens_reported

    def test_finish_without_content_fails(self) -> None:
        """Choices present but no modeled text field must fail."""
        recorder = OpenAIStreamRecorder(prompt_len=1, chat=True)

        recorder.on_event(
            json.dumps({"choices": [{"delta": {"role": "assistant"}}]})
        )
        output = recorder.finish()

        assert output.success is False
        assert "No text content" in output.error

    def test_fail_marks_output_unsuccessful(self) -> None:
        """``fail`` sets the error and marks the request unsuccessful."""
        recorder = OpenAIStreamRecorder(prompt_len=1, chat=True)

        output = recorder.fail("connection reset")

        assert output is recorder.output
        assert output.success is False
        assert output.error == "connection reset"

    def test_arrival_times_override_the_clock(self) -> None:
        """Given arrival times, TTFT, ITL and latency come from them."""
        recorder = OpenAIStreamRecorder(prompt_len=1, chat=True)
        start = recorder._st

        recorder.on_event(
            json.dumps({"choices": [{"delta": {"content": "a"}}]}),
            arrived=start + 0.5,
        )
        recorder.on_event(
            json.dumps({"choices": [{"delta": {"content": "b"}}]}),
            arrived=start + 0.75,
        )
        recorder.on_event("[DONE]", arrived=start + 1.0)
        output = recorder.finish()

        assert output.ttft == pytest.approx(0.5)
        assert output.itl == [pytest.approx(0.25)]
        assert output.latency == pytest.approx(1.0)

    def test_chunk_tpot_is_tokenized_once_after_the_stream(self) -> None:
        """Streaming records chunk offsets; one pass at the end computes TPOT."""
        tokenizer: Any = _WordCountingTokenizer()
        recorder = OpenAIStreamRecorder(
            prompt_len=1, chat=True, tokenizer=tokenizer
        )
        start = recorder._st

        for content, arrived in [
            ("ab", 0.5),
            ("c d", 0.75),
            (" ", 0.875),
            ("e", 1.0),
        ]:
            recorder.on_event(
                json.dumps({"choices": [{"delta": {"content": content}}]}),
                arrived=start + arrived,
            )
        output = recorder.finish()

        assert tokenizer.calls == []
        assert output.tpot == []
        assert list(output.pending_tpot_ends) == [2, 5, 6, 7]

        compute_chunk_tpots([output], tokenizer)

        # One call, for the chunks that can carry TPOT, in stream order. The
        # first chunk's text belongs to TTFT, and the whitespace-only chunk
        # counts no tokens, so neither contributes.
        assert tokenizer.calls == [["c d", " ", "e"]]
        assert output.tpot == [
            pytest.approx(0.25 / 2),
            pytest.approx(0.125),
        ]
        assert len(output.pending_tpot_ends) == 0

    def test_chunk_tpot_with_no_tokenizer_drops_the_recorded_offsets(
        self,
    ) -> None:
        """Without a tokenizer the recorded offsets are drained, not used."""
        tokenizer: Any = _WordCountingTokenizer()
        recorder = OpenAIStreamRecorder(
            prompt_len=1, chat=True, tokenizer=tokenizer
        )
        start = recorder._st
        for content, arrived in [("ab", 0.5), ("c d", 0.75)]:
            recorder.on_event(_text_event(content), arrived=start + arrived)
        output = recorder.finish()

        assert list(output.pending_tpot_ends) == [2, 5]

        compute_chunk_tpots([output], None)

        assert tokenizer.calls == []
        assert output.tpot == []
        assert len(output.pending_tpot_ends) == 0

    def test_chunk_tpot_is_tokenized_in_bounded_slices(
        self, mocker: MockerFixture
    ) -> None:
        """A run wider than one slice is tokenized slice by slice."""
        mocker.patch(
            "max.benchmark.benchmark_shared.request._TPOT_TOKENIZE_SLICE", 2
        )
        tokenizer: Any = _WordCountingTokenizer()
        first = _tpot_output(
            "a bb ccc dddd", itl=[0.2, 0.3, 0.4], ends=[1, 4, 8, 13]
        )
        second = _tpot_output("e f", itl=[0.5], ends=[1, 3])

        compute_chunk_tpots([first, second], tokenizer)

        # Two chunks per call: the first output's three chunks fill one split
        # slice, then a mixed slice.
        assert tokenizer.calls == [[" bb", " ccc"], [" dddd", " f"]]
        assert first.tpot == [
            pytest.approx(0.2),
            pytest.approx(0.3),
            pytest.approx(0.4),
        ]
        assert second.tpot == [pytest.approx(0.5)]
        assert len(first.pending_tpot_ends) == 0
        assert len(second.pending_tpot_ends) == 0

    def test_a_desynchronized_response_costs_only_its_own_tpot(self) -> None:
        """Offsets out of step with ITLs skip that response, not the run."""
        tokenizer: Any = _WordCountingTokenizer()
        synced = _tpot_output("c d e", itl=[0.4, 0.4], ends=[1, 3, 5])
        # One offset pair per ITL means three pairs for one ITL here.
        unsynced = _tpot_output("c d e", itl=[0.4], ends=[1, 3, 5])

        compute_chunk_tpots([synced, unsynced], tokenizer)

        assert unsynced.tpot == []
        assert synced.tpot == [pytest.approx(0.4), pytest.approx(0.4)]

    def test_a_refused_slice_leaves_no_partial_tpot(
        self, mocker: MockerFixture
    ) -> None:
        """A slice the tokenizer rejects drops that response's TPOT whole."""
        mocker.patch(
            "max.benchmark.benchmark_shared.request._TPOT_TOKENIZE_SLICE", 2
        )

        class _FussyTokenizer(_WordCountingTokenizer):
            """Counts words, but rejects any batch carrying ``bad``."""

            def __call__(
                self,
                texts: str | Sequence[str],
                add_special_tokens: bool = True,
            ) -> SimpleNamespace:
                batch = [texts] if isinstance(texts, str) else list(texts)
                if any("bad" in text for text in batch):
                    raise ValueError("unsupported text")
                return super().__call__(texts, add_special_tokens)

        tokenizer: Any = _FussyTokenizer()
        kept = _tpot_output("c d", itl=[0.2], ends=[1, 3])
        # Its first slice tokenizes and its second is refused, so the response
        # must not keep the values the first slice already appended.
        dropped = _tpot_output(
            "aa bb cc bad ee",
            itl=[0.1, 0.2, 0.3, 0.4],
            ends=[2, 5, 8, 12, 15],
        )

        compute_chunk_tpots([kept, dropped], tokenizer)

        # Only the batch that reached the tokenizer is recorded; the refused
        # one raised before counting.
        assert tokenizer.calls == [[" d", " bb"]]
        assert kept.tpot == [pytest.approx(0.2)]
        assert dropped.tpot == []

    @pytest.mark.parametrize(
        ("chunk", "expected_text"),
        [
            pytest.param(
                {
                    "id": "c1",
                    "vendor_extra": {"trace": "x"},
                    "choices": [
                        {
                            "index": 0,
                            "logprobs": None,
                            "delta": {"unmodeled": "ignored", "content": "a"},
                        }
                    ],
                },
                "a",
                id="unknown_fields_ignored",
            ),
            pytest.param(
                {"choices": [{"delta": {"content": None, "reasoning": None}}]},
                "",
                id="null_text_fields",
            ),
            pytest.param(
                {"choices": [{"delta": {"tool_calls": [{"index": 0}]}}]},
                "",
                id="tool_call_without_function",
            ),
            pytest.param(
                {"choices": [{"delta": {"content": "a"}, "finish_reason": 5}]},
                "a",
                id="non_string_finish_reason",
            ),
            pytest.param(
                {
                    "choices": [
                        {"delta": {"tool_calls": [{"function": "nope"}]}}
                    ]
                },
                "",
                id="tool_call_with_non_object_function",
            ),
            pytest.param({"choices": None}, "", id="null_choices"),
            pytest.param(
                {"choices": [], "usage": {"prompt_tokens": 3}},
                "",
                id="no_choices",
            ),
        ],
    )
    def test_chunk_tolerance(
        self, chunk: dict[str, Any], expected_text: str
    ) -> None:
        """A chunk of any tolerated shape contributes only its own text.

        The tolerated shapes are the ones that cannot change what the run
        reports: an absent choices list, a field the client does not model, a
        null optional delta field, a tool-call fragment with no function text,
        and a ``finish_reason`` that is not a string (only ``"length"`` is
        ever compared). Everything that would corrupt the measured text or the
        token counts fails at the chunk instead.
        """
        recorder = OpenAIStreamRecorder(prompt_len=1, chat=True)

        assert recorder.on_event(json.dumps(chunk)) is None
        # A trailing real chunk makes the case's own text observable, since a
        # response with no text at all is a failure rather than a record.
        recorder.on_event(
            json.dumps({"choices": [{"delta": {"content": "."}}]})
        )

        output = recorder.finish()
        assert output.success is True
        assert output.generated_text == expected_text + "."

    def test_chunk_tpot_survives_a_tokenizer_that_raises(self) -> None:
        """A failing tokenizer costs per-chunk TPOT, not the result build."""

        class _RefusingTokenizer(_WordCountingTokenizer):
            def __call__(
                self, texts: Any, add_special_tokens: bool = True
            ) -> Any:
                raise ValueError("unsupported text")

        tokenizer: Any = _RefusingTokenizer()
        recorder = OpenAIStreamRecorder(
            prompt_len=1, chat=True, tokenizer=tokenizer
        )
        start = recorder._st
        for event, arrived in [
            (_text_event("ab"), 0.5),
            (_text_event("c d"), 1.0),
        ]:
            recorder.on_event(event, arrived=start + arrived)
        output = recorder.finish()

        compute_chunk_tpots([output], tokenizer)

        assert output.tpot == []

    def test_failed_slices_cost_one_warning_naming_the_first(
        self, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Two refused slices report once, with the failure that caused it."""
        mocker.patch(
            "max.benchmark.benchmark_shared.request._TPOT_TOKENIZE_SLICE", 2
        )

        class _RefusingTokenizer(_WordCountingTokenizer):
            def __call__(
                self, texts: Any, add_special_tokens: bool = True
            ) -> Any:
                raise ValueError("unsupported text")

        tokenizer: Any = _RefusingTokenizer()
        # Three chunks each, so each response fills one slice on its own.
        first = _tpot_output(
            "a bb ccc dddd", itl=[0.2, 0.3, 0.4], ends=[1, 4, 8, 13]
        )
        second = _tpot_output(
            "e ff ggg hhhh", itl=[0.5, 0.6, 0.7], ends=[1, 4, 8, 13]
        )

        with caplog.at_level(
            logging.WARNING, logger="max.benchmark.benchmark_shared.request"
        ):
            compute_chunk_tpots([first, second], tokenizer)

        warnings = [
            rec for rec in caplog.records if rec.levelno == logging.WARNING
        ]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "2 of 2 responses" in message
        assert "2 tokenizer calls failed" in message
        assert "First failure: ValueError: unsupported text." in message

        # One refused slice reads as one call, not "1 calls".
        caplog.clear()
        single = _tpot_output("a bb ccc", itl=[0.5, 0.6], ends=[1, 4, 8])
        with caplog.at_level(
            logging.WARNING, logger="max.benchmark.benchmark_shared.request"
        ):
            compute_chunk_tpots([single], tokenizer)

        (warning,) = [
            rec for rec in caplog.records if rec.levelno == logging.WARNING
        ]
        assert "1 tokenizer call failed" in warning.getMessage()


def _streaming_response(
    mocker: MockerFixture, chunks: Sequence[bytes]
) -> AsyncMock:
    """Returns a 200 response mock whose body streams ``chunks``."""
    response = mocker.AsyncMock()
    response.status = 200
    response.content = StreamBody(chunks)
    return response


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
        request_input = make_request_input("http://localhost:8000/completions")

        # Mock response data
        mock_response_data = [
            b'data: {"choices": [{"text": "Hello"}]}\n\n',
            b'data: {"choices": [{"text": " world"}]}\n\n',
            b'data: {"choices": [{"text": "!"}]}\n\n',
            b"data: [DONE]\n\n",
        ]

        mock_aiohttp_session.setup_post_response(
            _streaming_response(mocker, mock_response_data)
        )

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
        request_input = make_request_input(
            "http://localhost:8000/chat/completions"
        )

        # Mock response data
        mock_response_data = [
            b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n',
            b'data: {"choices": [{"delta": {"content": " world"}}]}\n\n',
            b'data: {"choices": [{"delta": {"content": "!"}}]}\n\n',
            b"data: [DONE]\n\n",
        ]

        mock_aiohttp_session.setup_post_response(
            _streaming_response(mocker, mock_response_data)
        )

        driver = OpenAIChatCompletionsRequestDriver()
        result = await driver.request(request_input)

        assert result.success is True
        assert result.generated_text == "Hello world!"
        assert result.prompt_len == 10

    @pytest.mark.asyncio
    async def test_openai_chat_completions_choices_but_no_text_is_failure(
        self,
        mock_aiohttp_session: Any,
        mock_openai_env: None,
        mocker: MockerFixture,
    ) -> None:
        """Choices present but no modeled text field must fail, not succeed with ttft=0.

        Regression test for PERF-2615: gpt-oss streamed chunks that had
        ``choices`` but carried text in a delta field this client does not
        model, so ``reasoning``/``reasoning_content``/``content`` were all
        empty. Those runs were marked success with ttft=0 and no tokens, which
        zeroed out TTFT/TPOT for the whole run and produced a misleading
        "0 valid requests" steady-state warning. Such a response must now be a
        failure.
        """
        request_input = make_request_input(
            "http://localhost:8000/chat/completions"
        )

        # Chunks have choices but the text is in an unmodeled delta field;
        # reasoning/reasoning_content/content are all absent, so no text is
        # captured.
        mock_response_data = [
            b'data: {"choices": [{"delta": {"role": "assistant"}}]}\n\n',
            b'data: {"choices": [{"delta": {"unmodeled_channel": "hi"}}]}\n\n',
            b'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ]

        mock_aiohttp_session.setup_post_response(
            _streaming_response(mocker, mock_response_data)
        )

        driver = OpenAIChatCompletionsRequestDriver()
        result = await driver.request(request_input)

        assert result.success is False
        assert result.ttft == 0.0
        assert result.generated_text == ""
        assert result.error is not None
        assert "No text content" in result.error

    @pytest.mark.asyncio
    async def test_openai_chat_completions_tool_call_only_response_is_success(
        self,
        mock_aiohttp_session: Any,
        mock_openai_env: None,
        mocker: MockerFixture,
    ) -> None:
        """A pure tool-call turn is a successful, content-bearing response.

        Regression test for the nemotron-opencode "No text content captured"
        failures: an agentic response can consist entirely of ``tool_calls``
        deltas (no ``content``/``reasoning`` at all). The client must model
        the ``tool_calls`` field and treat those chunks as content-bearing
        rather than reporting the request as failed.
        """
        request_input = make_request_input(
            "http://localhost:8000/chat/completions", prompt="Write a file"
        )

        # Mirrors the server's streaming shape for a tool-call-only turn:
        # an opener chunk with id + name, then argument fragments.
        mock_response_data = [
            (
                b'data: {"choices": [{"delta": {"role": "assistant", '
                b'"tool_calls": [{"index": 0, "id": "write:fa3c5398", '
                b'"type": "function", '
                b'"function": {"name": "write", "arguments": "{\\""}}]}}]}\n\n'
            ),
            (
                b'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, '
                b'"function": {"arguments": "content\\": \\"hi\\"}"}}]}}]}\n\n'
            ),
            (
                b'data: {"choices": [{"delta": {}, '
                b'"finish_reason": "length"}]}\n\n'
            ),
            b"data: [DONE]\n\n",
        ]

        mock_aiohttp_session.setup_post_response(
            _streaming_response(mocker, mock_response_data)
        )

        driver = OpenAIChatCompletionsRequestDriver()
        result = await driver.request(request_input)

        assert result.success is True
        assert result.error == ""
        assert result.ttft > 0.0
        # The tool name and argument bytes are the generated text.
        assert result.generated_text == 'write{"content": "hi"}'

    @pytest.mark.asyncio
    async def test_openai_chat_completions_content_before_tool_call_is_not_double_counted(
        self,
        mock_aiohttp_session: Any,
        mock_openai_env: None,
        mocker: MockerFixture,
    ) -> None:
        """Content and tool_calls deltas are additive, never overlapping.

        Regression guard for double-counting: a turn that streams prose
        before deciding to call a tool must count that prose exactly once,
        even though ``_delta_text`` sums ``content`` and
        ``tool_calls`` text from the same delta object. The server only ever
        populates one of the two per ``ParsedToolCallDelta``, so summing them
        is additive by construction -- this test pins that with an exact
        length/text check rather than relying on that invariant by
        inspection alone.
        """
        request_input = make_request_input(
            "http://localhost:8000/chat/completions",
            prompt="What's the weather, then check the time",
        )

        mock_response_data = [
            (
                b'data: {"choices": [{"delta": {"role": "assistant", '
                b'"content": "Let me check that for you."}}]}\n\n'
            ),
            (
                b'data: {"choices": [{"delta": {'
                b'"tool_calls": [{"index": 0, "id": "get_weather:1", '
                b'"type": "function", '
                b'"function": {"name": "get_weather", "arguments": "{}"}}]}}]}\n\n'
            ),
            (
                b'data: {"choices": [{"delta": {}, '
                b'"finish_reason": "tool_calls"}]}\n\n'
            ),
            b"data: [DONE]\n\n",
        ]

        mock_aiohttp_session.setup_post_response(
            _streaming_response(mocker, mock_response_data)
        )

        driver = OpenAIChatCompletionsRequestDriver()
        result = await driver.request(request_input)

        assert result.success is True
        # The prose and the tool call are disjoint deltas; the total is
        # their concatenation, not double the prose or double the call.
        assert (
            result.generated_text == "Let me check that for you.get_weather{}"
        )
        assert result.generated_text.count("Let me check that for you.") == 1

    @pytest.mark.asyncio
    async def test_openai_chat_completions_merges_reasoning_reasoning_content_and_content(
        self,
        mock_aiohttp_session: Any,
        mock_openai_env: None,
        mocker: MockerFixture,
    ) -> None:
        """Test that reasoning, reasoning_content, and content are merged in order."""
        request_input = make_request_input(
            "http://localhost:8000/chat/completions",
            prompt="Think step by step",
            prompt_len=5,
        )

        # Chunks: out-of-order reasoning, reasoning_content, and content
        mock_response_data = [
            (
                b'data: {"choices": [{"delta": {'
                b'"content": " The answer is 42.", '
                b'"reasoning": "Let me think.", '
                b'"reasoning_content": " chain-of-thought."'
                b"}}]}\n\n"
            ),
            b"data: [DONE]\n\n",
        ]

        mock_aiohttp_session.setup_post_response(
            _streaming_response(mocker, mock_response_data)
        )

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
        request_input = make_request_input(
            "http://localhost:8000/chat/completions",
            prompt="Test",
            prompt_len=1,
            max_tokens=50,
        )

        # Chunks with only some keys present or None; no "reasoning" or "reasoning_content"
        mock_response_data = [
            b'data: {"choices": [{"delta": {"reasoning": "Let me "}}]}\n\n',
            b'data: {"choices": [{"delta": {"reasoning": "think.", "content": " The"}}]}\n\n',
            b'data: {"choices": [{"delta": {"reasoning": null, "content": " answer"}}]}\n\n',
            b'data: {"choices": [{"delta": {"content": " is 42."}}]}\n\n',
            b"data: [DONE]\n\n",
        ]

        mock_aiohttp_session.setup_post_response(
            _streaming_response(mocker, mock_response_data)
        )

        driver = OpenAIChatCompletionsRequestDriver()
        result = await driver.request(request_input)

        assert result.success is True
        assert result.generated_text == "Let me think. The answer is 42.", (
            "content-only and None keys must yield merged content"
        )
        assert result.prompt_len == 1

    @pytest.mark.asyncio
    async def test_openai_chat_completions_error_chunk_reports_server_message(
        self,
        mock_aiohttp_session: Any,
        mock_openai_env: None,
        mocker: MockerFixture,
    ) -> None:
        """Regression test for MXTOOLS-203: server error message is surfaced in output.error."""
        request_input = make_request_input(
            "http://localhost:8000/v1/chat/completions",
            prompt="A very long prompt",
            prompt_len=264408,
        )

        error_msg = "Input string is larger than tokenizer's max length (264823 > 262144)."
        error_json = json.dumps(
            {
                "error": {
                    "code": "500",
                    "message": error_msg,
                    "param": "",
                    "type": "",
                }
            }
        )

        mock_aiohttp_session.setup_post_response(
            _streaming_response(
                mocker,
                [
                    f"data: {error_json}\n\n".encode(),
                ],
            )
        )

        driver = OpenAIChatCompletionsRequestDriver()
        result = await driver.request(request_input)

        assert result.success is False
        assert result.error == error_msg

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("prompt", "images", "expected"),
        [
            pytest.param("hi", [], False, id="text-only"),
            pytest.param(
                "hi",
                [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,AAA"},
                    }
                ],
                True,
                id="attached",
            ),
            pytest.param(
                [
                    ChatMessage(
                        role="user",
                        content=[
                            TextContentBlock(text="what is this?"),
                            ImageContentBlock(
                                image_url=ImageURLDetail(
                                    url="data:image/jpeg;base64,AAA"
                                )
                            ),
                        ],
                    ),
                    ChatMessage(role="assistant", content="a cat"),
                    ChatMessage(
                        role="user", content=[TextContentBlock(text="and?")]
                    ),
                ],
                [],
                True,
                id="resent-with-history",
            ),
        ],
    )
    async def test_openai_chat_completions_records_whether_the_payload_carries_an_image(
        self,
        mock_aiohttp_session: Any,
        mock_openai_env: None,
        mocker: MockerFixture,
        prompt: str | list[ChatMessage],
        images: list[OpenAIImage],
        expected: bool,
    ) -> None:
        """A turn that only resends an earlier image still carries one."""
        request_input = make_request_input(
            "http://localhost:8000/v1/chat/completions",
            prompt=prompt,
            prompt_len=1,
            max_tokens=8,
            images=images,
        )

        mock_aiohttp_session.setup_post_response(
            _streaming_response(
                mocker,
                [
                    b'data: {"choices": [{"delta": {"content": "x"}}]}\n\n',
                    b"data: [DONE]\n\n",
                ],
            )
        )

        result = await OpenAIChatCompletionsRequestDriver().request(
            request_input
        )

        assert result.success is True
        assert result.carries_image is expected

    @pytest.mark.asyncio
    async def test_openai_completions_request_driver_no_content(
        self,
        mock_aiohttp_session: Any,
        mock_openai_env: None,
        mocker: MockerFixture,
    ) -> None:
        """Test that missing content returns an error."""
        request_input = make_request_input("http://localhost:8000/completions")

        mock_response_data = [
            b"data: [DONE]\n\n",
        ]

        mock_aiohttp_session.setup_post_response(
            _streaming_response(mocker, mock_response_data)
        )

        driver = OpenAICompletionsRequestDriver()
        result = await driver.request(request_input)

        assert result.success is False
        assert "No text content" in result.error

    @pytest.mark.asyncio
    async def test_openai_chat_completions_ignores_events_after_done(
        self,
        mock_aiohttp_session: Any,
        mock_openai_env: None,
        mocker: MockerFixture,
    ) -> None:
        """Nothing after ``[DONE]`` is parsed, and the body is drained."""
        request_input = RequestFuncInput(
            model="test-model",
            session_id=None,
            sampling=SamplingConfig(),
            prompt="Test prompt",
            images=[],
            api_url="http://localhost:8000/v1/chat/completions",
            prompt_len=10,
            max_tokens=100,
            ignore_eos=False,
        )

        mock_response_data = [
            b'data: {"choices": [{"delta": {"content": "Hi"}}]}\n\n',
            (
                b'data: {"choices": [], "usage": {"prompt_tokens": 3, '
                b'"completion_tokens": 1, "total_tokens": 4, '
                b'"prompt_tokens_details": {"cached_tokens": 2}}}\n\n'
            ),
            b"data: [DONE]\n\n",
            # A chunk this client cannot parse: drained, never parsed, so it
            # cannot fail the request.
            b"data: {not json\n\n",
        ]

        body = StreamBody(mock_response_data)
        response = mocker.AsyncMock()
        response.status = 200
        response.content = body
        mock_aiohttp_session.setup_post_response(response)

        result = await OpenAIChatCompletionsRequestDriver().request(
            request_input
        )

        assert result.success is True
        assert result.generated_text == "Hi"
        assert result.server_token_stats.cached_tokens == 2
        # Read to EOF: the response closes cleanly instead of resetting.
        assert body.chunks == []

    @pytest.mark.asyncio
    async def test_done_early_exit_leaves_latency_alone(
        self,
        mock_aiohttp_session: Any,
        mock_openai_env: None,
        mocker: MockerFixture,
    ) -> None:
        """The drain after ``[DONE]`` does not extend the measured latency."""
        request_input = make_request_input(
            "http://localhost:8000/v1/chat/completions"
        )

        class _HeldTailBody(StreamBody):
            """Holds the response open after ``[DONE]`` before it ends."""

            async def iter_chunked(self, nbytes: int) -> AsyncIterator[bytes]:
                await asyncio.sleep(0.2)
                async for chunk in super().iter_chunked(nbytes):
                    yield chunk

        body = _HeldTailBody(
            [
                b'data: {"choices": [{"delta": {"content": "Hi"}}]}\n\n',
                b"data: [DONE]\n\n",
                b"data: {not json\n\n",
            ]
        )
        response = mocker.AsyncMock()
        response.status = 200
        response.content = body
        mock_aiohttp_session.setup_post_response(response)

        started = time.perf_counter()
        result = await OpenAIChatCompletionsRequestDriver().request(
            request_input
        )
        elapsed = time.perf_counter() - started

        assert result.success is True
        # The clock stops at ``[DONE]``; only the drain waits for the tail.
        assert elapsed - result.latency >= 0.1

    @pytest.mark.asyncio
    async def test_a_post_done_trailer_is_relayed_but_not_measured(
        self,
        mock_aiohttp_session: Any,
        mock_openai_env: None,
        mocker: MockerFixture,
    ) -> None:
        """Work after ``[DONE]`` is read out and never measured.

        A trailer that carries text would corrupt TTFT, ITL and the generated
        text if the client kept recording, and leaving it unread resets the
        socket. This pins all three: nothing after ``[DONE]`` is parsed, the
        tail is relayed to EOF, and ``finish()`` reports the ``[DONE]``
        measurement.
        """
        request_input = make_request_input(
            "http://localhost:8000/v1/chat/completions"
        )
        drained: list[bytes] = []

        class _RelayRecordingBody(StreamBody):
            """Records what the drain actually reads."""

            async def iter_chunked(self, nbytes: int) -> AsyncIterator[bytes]:
                async for chunk in super().iter_chunked(nbytes):
                    drained.append(chunk)
                    yield chunk

        body = _RelayRecordingBody(
            [
                b'data: {"choices": [{"delta": {"content": "Hi"}}]}\n\n',
                b"data: [DONE]\n\n",
                b'data: {"choices": [{"delta": {"content": "TRAILER"}}]}\n\n',
            ]
        )
        response = mocker.AsyncMock()
        response.status = 200
        response.content = body
        mock_aiohttp_session.setup_post_response(response)

        started = time.perf_counter()
        result = await OpenAIChatCompletionsRequestDriver().request(
            request_input
        )
        elapsed = time.perf_counter() - started

        assert result.success is True
        assert result.generated_text == "Hi"
        assert result.itl == []
        # finish() reported the [DONE] measurement, not the end of the drain.
        assert 0.0 < result.latency <= elapsed
        # The trailer was relayed, and the body is empty afterwards.
        assert b"TRAILER" in b"".join(drained)
        assert body.chunks == []

    @pytest.mark.asyncio
    async def test_openai_chat_completions_malformed_chunk_fails_the_request(
        self,
        mock_aiohttp_session: Any,
        mock_openai_env: None,
        mocker: MockerFixture,
    ) -> None:
        """A chunk that is not JSON fails the request, quoting the payload."""
        request_input = RequestFuncInput(
            model="test-model",
            session_id=None,
            sampling=SamplingConfig(),
            prompt="Test prompt",
            images=[],
            api_url="http://localhost:8000/v1/chat/completions",
            prompt_len=10,
            max_tokens=100,
            ignore_eos=False,
        )

        mock_response_data = [
            b'data: {"choices": [{"delta": {"content": "Hi"}}]}\n\n',
            b"data: {not json\n\n",
            b"data: [DONE]\n\n",
        ]

        mock_aiohttp_session.setup_post_response(
            _streaming_response(mocker, mock_response_data)
        )

        result = await OpenAIChatCompletionsRequestDriver().request(
            request_input
        )

        assert result.success is False
        assert "Malformed stream chunk" in result.error
        assert repr("{not json") in result.error

    @pytest.mark.asyncio
    async def test_a_bool_usage_count_fails_the_request(
        self,
        mock_aiohttp_session: Any,
        mock_openai_env: None,
        mocker: MockerFixture,
    ) -> None:
        """``true`` in a count field fails the request where it arrives."""
        request_input = make_request_input(
            "http://localhost:8000/v1/chat/completions"
        )

        mock_aiohttp_session.setup_post_response(
            _streaming_response(
                mocker,
                [
                    b'data: {"choices": [{"delta": {"content": "Hi"}}]}\n\n',
                    b'data: {"choices": [], "usage": {"prompt_tokens": true}}\n\n',
                    b"data: [DONE]\n\n",
                ],
            )
        )

        result = await OpenAIChatCompletionsRequestDriver().request(
            request_input
        )

        assert result.success is False
        assert "usage count is not an integer" in result.error

    @pytest.mark.asyncio
    async def test_request_driver_with_progress_bar(
        self, mock_aiohttp_session: Any, mocker: MockerFixture
    ) -> None:
        """Test that RequestDriver works with progress bars."""
        request_input = make_request_input(
            "http://localhost:8000/generate_stream"
        )

        # Create a mock progress bar
        mock_pbar = mocker.Mock(spec=tqdm)

        mock_aiohttp_session.setup_post_response(
            _streaming_response(
                mocker,
                [
                    b'data: {"text_output": "Hello"}\n\n',
                ],
            )
        )

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
            sampling=SamplingConfig(),
            prompt="Test prompt",
            images=[],
            api_url="http://localhost:8000/generate_stream",
            prompt_len=10,
            max_tokens=100,
            ignore_eos=False,
        )

        mock_aiohttp_session.setup_post_response(
            _streaming_response(
                mocker,
                [
                    b'data: {"text_output": "Hello"}\n\n',
                ],
            )
        )

        driver = TRTLLMRequestDriver()
        result = await driver.request(request_input)

        assert result.success is True

    @pytest.mark.parametrize(
        "chunk",
        [
            pytest.param(b'data: {"text_output": null}\n\n', id="null"),
            pytest.param(b'data: {"text_output": 7}\n\n', id="number"),
            pytest.param(b"data: {}\n\n", id="missing"),
            pytest.param(b"data: 5\n\n", id="not_an_object"),
            pytest.param(b"data: {not json\n\n", id="not_json"),
        ],
    )
    @pytest.mark.asyncio
    async def test_trtllm_non_string_text_output_fails_only_that_request(
        self, mock_aiohttp_session: Any, mocker: MockerFixture, chunk: bytes
    ) -> None:
        """A chunk the client cannot read fails the request, not the run."""
        request_input = RequestFuncInput(
            model="test-model",
            session_id=None,
            sampling=SamplingConfig(),
            prompt="Test prompt",
            images=[],
            api_url="http://localhost:8000/generate_stream",
            prompt_len=10,
            max_tokens=100,
            ignore_eos=False,
        )

        mock_aiohttp_session.setup_post_response(
            _streaming_response(
                mocker, [b'data: {"text_output": "ok"}\n\n', chunk]
            )
        )

        result = await TRTLLMRequestDriver().request(request_input)

        assert result.success is False
        assert result.generated_text == "ok"
        # The message quotes the payload, so the log says what broke it.
        assert "Malformed stream chunk" in result.error
        payload = chunk.removeprefix(b"data: ").removesuffix(b"\n\n")
        assert repr(payload.decode())[:60] in result.error

    @pytest.mark.asyncio
    async def test_trtllm_records_chunk_offsets_for_the_post_run_pass(
        self, mock_aiohttp_session: Any, mocker: MockerFixture
    ) -> None:
        """TRT-LLM's chunks get TPOT from the post-run pass, not usage."""
        request_input = RequestFuncInput(
            model="test-model",
            session_id=None,
            sampling=SamplingConfig(),
            prompt="Test prompt",
            images=[],
            api_url="http://localhost:8000/generate_stream",
            prompt_len=10,
            max_tokens=100,
            ignore_eos=False,
        )

        mock_aiohttp_session.setup_post_response(
            _streaming_response(
                mocker,
                [
                    b'data: {"text_output": "ab"}\n\n',
                    b'data: {"text_output": "c d"}\n\n',
                    b'data: {"text_output": "e"}\n\n',
                ],
            )
        )

        tokenizer: Any = _WordCountingTokenizer()
        result = await TRTLLMRequestDriver(tokenizer=tokenizer).request(
            request_input
        )

        assert list(result.pending_tpot_ends) == [2, 5, 6]

        compute_chunk_tpots([result], tokenizer)

        assert tokenizer.calls == [["c d", "e"]]
        assert result.tpot == [
            pytest.approx(result.itl[0] / 2),
            pytest.approx(result.itl[1]),
        ]
        assert len(result.pending_tpot_ends) == 0

    @pytest.mark.asyncio
    async def test_trtllm_ttft_is_the_first_chunk_not_the_bookkeeping(
        self, mock_aiohttp_session: Any, mocker: MockerFixture
    ) -> None:
        """TTFT is the read taken before the chunk is appended, not after."""
        request_input = RequestFuncInput(
            model="test-model",
            session_id=None,
            sampling=SamplingConfig(),
            prompt="Test prompt",
            images=[],
            api_url="http://localhost:8000/generate_stream",
            prompt_len=10,
            max_tokens=100,
            ignore_eos=False,
        )

        mock_aiohttp_session.setup_post_response(
            _streaming_response(
                mocker,
                [
                    b'data: {"text_output": "ab"}\n\n',
                    b'data: {"text_output": "c d"}\n\n',
                ],
            )
        )

        class _SteppingClock:
            """Advances 1.0 s per read, so a read's value is its position."""

            def __init__(self) -> None:
                self.now = 0.0

            def perf_counter(self) -> float:
                self.now += 1.0
                return self.now

            def __getattr__(self, name: str) -> Any:
                return getattr(time, name)

        mocker.patch(
            "max.benchmark.benchmark_shared.request.time", _SteppingClock()
        )

        result = await TRTLLMRequestDriver().request(request_input)

        assert result.success is True
        # Submit at 1.0, first chunk at 2.0, second at 3.0. A separate read
        # after the appends would report 3.0 - 1.0 for the first chunk.
        assert result.ttft == pytest.approx(1.0)
        assert result.latency == pytest.approx(2.0)


class TestBuildFinalPayload:
    """Unit tests for the ``_build_final_payload`` payload helper."""

    def test_json_override_assignment_is_resolved_before_payload(self) -> None:
        config = ServingBenchmarkConfig(model="m")
        # Workload overrides are assigned after the benchmark config is built.
        overrides = {"extra_body": '{"repetition_penalty":1.0}'}
        for key, value in overrides.items():
            setattr(config, key, value)

        payload = _build_final_payload({"model": "m"}, config.extra_body)

        assert payload == {"model": "m", "repetition_penalty": 1.0}

    def test_none_and_empty_return_copy_unchanged(self) -> None:
        payload = {"model": "m"}
        assert _build_final_payload(payload, None) == {"model": "m"}
        assert _build_final_payload(payload, {}) == {"model": "m"}
        # The helper returns a fresh dict; the caller's payload is not mutated.
        assert _build_final_payload(payload, {"x": 1}) is not payload
        assert "x" not in payload

    def test_adds_new_keys_with_nested_verbatim(self) -> None:
        payload: dict[str, Any] = {"model": "m"}
        extra = {
            "vendor_ext": {"flags": ["a", "b"]},
            "stop": ["}"],
        }
        body = _build_final_payload(payload, extra)
        assert body["vendor_ext"] == {"flags": ["a", "b"]}
        assert body["stop"] == ["}"]

    def test_collision_overwrites_and_logs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        payload = {"max_tokens": 100, "stream": True}
        with caplog.at_level(
            logging.WARNING, logger="max.benchmark.benchmark_shared.request"
        ):
            body = _build_final_payload(payload, {"max_tokens": 15})
        # last-writer-wins
        assert body["max_tokens"] == 15
        assert body["stream"] is True
        # collision is surfaced
        assert any(
            "overwrites managed request field" in rec.message
            and "max_tokens" in rec.message
            for rec in caplog.records
        )


class TestDriverExtraBody:
    """End-to-end: extra_body reaches the POST payload of text-gen drivers."""

    _EXTRA = {
        "stop": ["}"],
        "chat_template_kwargs": {"reasoning_effort": "low"},
        # Synthetic vendor extension: a nested object holding an array, to
        # confirm arbitrary nested structures pass through verbatim.
        "vendor_ext": {"flags": ["a", "b"]},
    }

    @pytest.mark.asyncio
    async def test_chat_driver_merges_extra_body(
        self,
        mock_aiohttp_session: Any,
        mock_openai_env: None,
        mocker: MockerFixture,
    ) -> None:
        mock_aiohttp_session.setup_post_response(
            _streaming_response(
                mocker,
                [
                    b'data: {"choices": [{"delta": {"content": "ok"}}]}\n\n',
                    b"data: [DONE]\n\n",
                ],
            )
        )

        driver = OpenAIChatCompletionsRequestDriver(extra_body=self._EXTRA)
        result = await driver.request(
            make_request_input(
                "http://localhost:8000/v1/chat/completions",
                prompt="hi",
                prompt_len=1,
            )
        )
        assert result.success is True
        sent = mock_aiohttp_session.post.call_args[1]["json"]
        assert sent["stop"] == ["}"]
        assert sent["chat_template_kwargs"] == {"reasoning_effort": "low"}
        assert sent["vendor_ext"] == {"flags": ["a", "b"]}

    @pytest.mark.asyncio
    async def test_completions_driver_merges_extra_body(
        self,
        mock_aiohttp_session: Any,
        mock_openai_env: None,
        mocker: MockerFixture,
    ) -> None:
        mock_aiohttp_session.setup_post_response(
            _streaming_response(
                mocker,
                [
                    b'data: {"choices": [{"text": "ok"}]}\n\n',
                    b"data: [DONE]\n\n",
                ],
            )
        )

        driver = OpenAICompletionsRequestDriver(extra_body=self._EXTRA)
        result = await driver.request(
            make_request_input(
                "http://localhost:8000/v1/completions",
                prompt="hi",
                prompt_len=1,
            )
        )
        assert result.success is True
        sent = mock_aiohttp_session.post.call_args[1]["json"]
        assert sent["stop"] == ["}"]
        assert sent["vendor_ext"] == {"flags": ["a", "b"]}

    @pytest.mark.asyncio
    async def test_trtllm_driver_merges_extra_body(
        self,
        mock_aiohttp_session: Any,
        mocker: MockerFixture,
    ) -> None:
        mock_aiohttp_session.setup_post_response(
            _streaming_response(
                mocker,
                [
                    # The TRT-LLM driver validates every event as a chunk and
                    # has no ``[DONE]`` sentinel, so emit only one data line.
                    b'data: {"text_output": "ok"}\n\n',
                ],
            )
        )

        driver = TRTLLMRequestDriver(extra_body=self._EXTRA)
        result = await driver.request(
            make_request_input(
                "http://localhost:8000/v2/models/ensemble/generate_stream",
                prompt="hi",
                prompt_len=1,
            )
        )
        assert result.success is True
        sent = mock_aiohttp_session.post.call_args[1]["json"]
        assert sent["stop"] == ["}"]
        assert sent["vendor_ext"] == {"flags": ["a", "b"]}


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

    @pytest.mark.parametrize(
        ("path", "task", "driver"),
        [
            # task=None exercises the default (text-generation) argument.
            pytest.param(
                "/v1/chat/completions",
                None,
                OpenAIChatCompletionsRequestDriver,
                id="chat",
            ),
            pytest.param(
                "/v2/models/ensemble/generate_stream",
                None,
                TRTLLMRequestDriver,
                id="generate_stream",
            ),
            pytest.param(
                "/v1/responses",
                "text-to-image",
                OpenResponsesRequestDriver,
                id="pixel_gen_responses",
            ),
            pytest.param(
                "/v1/images/generations",
                "text-to-image",
                SglangPixelGenerationRequestDriver,
                id="pixel_gen_sglang",
            ),
            pytest.param(
                "/v1/chat/completions",
                "text-to-image",
                VllmOmniPixelGenerationRequestDriver,
                id="pixel_gen_vllm_omni",
            ),
            pytest.param(
                "/v1/images/generations",
                "image-to-image",
                SglangPixelGenerationRequestDriver,
                id="pixel_gen_image_to_image",
            ),
            pytest.param(
                "/v1/videos/sync",
                "text-to-video",
                VllmOmniVideoRequestDriver,
                id="pixel_gen_vllm_omni_video",
            ),
            pytest.param(
                "/v1/videos",
                "text-to-video",
                SglangVideoRequestDriver,
                id="pixel_gen_sglang_video",
            ),
            # Chat completions must stay on the chat driver, not the vLLM-Omni
            # pixel driver, when the task is text generation.
            pytest.param(
                "/v1/chat/completions",
                "text-generation",
                OpenAIChatCompletionsRequestDriver,
                id="text_gen_chat_not_pixel",
            ),
        ],
    )
    def test_get_request_driver_class(
        self,
        path: str,
        task: BenchmarkTask | None,
        driver: type[RequestDriver],
    ) -> None:
        url = f"http://localhost:8000{path}"
        if task is None:
            assert get_request_driver_class(url) is driver
        else:
            assert get_request_driver_class(url, task=task) is driver

    def test_get_request_driver_class_invalid_url(self) -> None:
        with pytest.raises(ValueError, match="Unsupported API URL"):
            get_request_driver_class("http://localhost:8000/unsupported")

    def test_get_request_driver_class_pixel_gen_invalid_url(self) -> None:
        with pytest.raises(ValueError, match="pixel-generation"):
            get_request_driver_class(
                "http://localhost:8000/v1/completions",
                task="text-to-image",
            )

    @pytest.mark.parametrize("path", ["/v1/videos/sync", "/v1/videos"])
    def test_get_request_driver_class_text_gen_videos_rejected(
        self, path: str
    ) -> None:
        """text-generation + a video endpoint should raise, not return the
        video driver."""
        with pytest.raises(ValueError, match="Unsupported API URL"):
            get_request_driver_class(
                f"http://localhost:8000{path}",
                task="text-generation",
            )


class TestPixelGenerationPayloadBuilders:
    """Tests for sglang and vllm omni pixel generation payload builders."""

    def _make_input(
        self,
        image_options: PixelGenerationImageOptions | None = None,
    ) -> PixelGenerationRequestFuncInput:
        return PixelGenerationRequestFuncInput(
            model="black-forest-labs/FLUX.2-dev",
            session_id=None,
            prompt="A beautiful sunset",
            input_image_paths=None,
            api_url="http://localhost:8000/v1/images/generations",
            image_options=image_options,
        )

    def test_sglang_payload_all_options(self) -> None:
        opts = PixelGenerationImageOptions(
            width=1024,
            height=768,
            steps=28,
            guidance_scale=3.5,
            seed=42,
            negative_prompt="blurry",
        )
        payload = _build_sglang_pixel_generation_payload(self._make_input(opts))
        assert payload["model"] == "black-forest-labs/FLUX.2-dev"
        assert payload["prompt"] == "A beautiful sunset"
        assert payload["size"] == "1024x768"
        assert payload["num_inference_steps"] == 28
        assert payload["guidance_scale"] == 3.5
        assert payload["seed"] == 42
        assert payload["n"] == 1
        assert payload["response_format"] == "b64_json"
        # negative_prompt is intentionally not supported
        assert "negative_prompt" not in payload

    def test_sglang_payload_no_options(self) -> None:
        payload = _build_sglang_pixel_generation_payload(self._make_input(None))
        assert payload["prompt"] == "A beautiful sunset"
        assert "size" not in payload
        assert "num_inference_steps" not in payload

    def test_sglang_payload_partial_options(self) -> None:
        opts = PixelGenerationImageOptions(steps=20)
        payload = _build_sglang_pixel_generation_payload(self._make_input(opts))
        assert payload["num_inference_steps"] == 20
        # size requires both width and height
        assert "size" not in payload
        assert "guidance_scale" not in payload

    def test_vllm_omni_payload_all_options(self) -> None:
        opts = PixelGenerationImageOptions(
            width=1024,
            height=768,
            steps=28,
            guidance_scale=3.5,
            seed=42,
            negative_prompt="blurry",
        )
        payload = _build_vllm_omni_pixel_generation_payload(
            self._make_input(opts)
        )
        assert payload["model"] == "black-forest-labs/FLUX.2-dev"
        assert payload["messages"] == [
            {"role": "user", "content": "A beautiful sunset"}
        ]
        extra = payload["extra_body"]
        assert extra["height"] == 768
        assert extra["width"] == 1024
        assert extra["num_inference_steps"] == 28
        assert extra["guidance_scale"] == 3.5
        assert extra["seed"] == 42
        # negative_prompt is intentionally not supported
        assert "negative_prompt" not in extra

    def test_vllm_omni_payload_no_options(self) -> None:
        payload = _build_vllm_omni_pixel_generation_payload(
            self._make_input(None)
        )
        assert "extra_body" not in payload
        assert payload["messages"][0]["content"] == "A beautiful sunset"

    def test_vllm_omni_payload_partial_options(self) -> None:
        opts = PixelGenerationImageOptions(height=512, width=512)
        payload = _build_vllm_omni_pixel_generation_payload(
            self._make_input(opts)
        )
        extra = payload["extra_body"]
        assert extra["height"] == 512
        assert extra["width"] == 512
        assert "num_inference_steps" not in extra

    def test_vllm_omni_video_payload_all_options(self) -> None:
        opts = PixelGenerationImageOptions(
            width=832,
            height=480,
            steps=50,
            guidance_scale=4.0,
            seed=42,
            negative_prompt="blurry",
            num_frames=81,
        )
        payload = _build_vllm_omni_video_payload(self._make_input(opts))
        assert payload["prompt"] == "A beautiful sunset"
        assert payload["model"] == "black-forest-labs/FLUX.2-dev"
        assert payload["width"] == "832"
        assert payload["height"] == "480"
        assert payload["num_inference_steps"] == "50"
        assert payload["guidance_scale"] == "4.0"
        assert payload["seed"] == "42"
        assert payload["negative_prompt"] == "blurry"
        assert payload["num_frames"] == "81"

    def test_vllm_omni_video_payload_no_options(self) -> None:
        payload = _build_vllm_omni_video_payload(self._make_input(None))
        assert payload["prompt"] == "A beautiful sunset"
        assert payload["model"] == "black-forest-labs/FLUX.2-dev"
        assert len(payload) == 2

    def test_vllm_omni_video_payload_partial_options(self) -> None:
        opts = PixelGenerationImageOptions(width=480, height=480, num_frames=33)
        payload = _build_vllm_omni_video_payload(self._make_input(opts))
        assert payload["width"] == "480"
        assert payload["height"] == "480"
        assert payload["num_frames"] == "33"
        assert "num_inference_steps" not in payload
        assert "guidance_scale" not in payload

    def test_sglang_video_payload_all_options(self) -> None:
        opts = PixelGenerationImageOptions(
            width=832,
            height=480,
            steps=50,
            guidance_scale=4.0,
            seed=42,
            negative_prompt="blurry",
            num_frames=81,
        )
        # The builder also accepts registered video schemas; narrow it
        # here so mypy checks the sglang keys against SglangVideoPayload.
        payload = cast(
            SglangVideoPayload,
            _build_sglang_video_payload(self._make_input(opts)),
        )
        assert payload["prompt"] == "A beautiful sunset"
        assert payload["model"] == "black-forest-labs/FLUX.2-dev"
        assert payload["width"] == 832
        assert payload["height"] == 480
        assert payload["num_inference_steps"] == 50
        assert payload["guidance_scale"] == 4.0
        assert payload["seed"] == 42
        assert payload["negative_prompt"] == "blurry"
        assert payload["num_frames"] == 81
        assert "n" not in payload
        assert "response_format" not in payload

    def test_sglang_video_payload_no_options(self) -> None:
        payload = cast(
            SglangVideoPayload,
            _build_sglang_video_payload(self._make_input(None)),
        )
        assert payload["prompt"] == "A beautiful sunset"
        assert payload["model"] == "black-forest-labs/FLUX.2-dev"
        assert len(payload) == 2

    def test_sglang_video_payload_partial_options(self) -> None:
        opts = PixelGenerationImageOptions(width=832, height=480, num_frames=81)
        payload = cast(
            SglangVideoPayload,
            _build_sglang_video_payload(self._make_input(opts)),
        )
        assert payload["width"] == 832
        assert payload["height"] == 480
        assert payload["num_frames"] == 81
        assert "num_inference_steps" not in payload
        assert "guidance_scale" not in payload


class TestCountGeneratedMedia:
    """Tests for _count_generated_media (OpenResponses response parser)."""

    @staticmethod
    def _wrap(content: list[dict[str, Any]]) -> dict[str, Any]:
        """Wrap content items in the OpenResponses message envelope."""
        return {
            "output": [
                {
                    "id": "msg_test_0",
                    "role": "assistant",
                    "content": content,
                    "status": "completed",
                }
            ]
        }

    def test_counts_output_image(self) -> None:
        body = self._wrap(
            [{"type": "output_image", "image_url": "http://x/img.png"}]
        )
        assert _count_generated_media(body) == 1

    def test_counts_output_video(self) -> None:
        # Regression: video responses were previously miscounted as 0, causing
        # every text-to-video request to fail with "No output_image content
        # found in OpenResponses response body." (PERF-2518).
        body = self._wrap(
            [
                {
                    "type": "output_video",
                    "video_url": "http://x/vid.mp4",
                    "format": "mp4",
                    "frames_per_second": 16,
                    "num_frames": 81,
                }
            ]
        )
        assert _count_generated_media(body) == 1

    def test_counts_mixed_image_and_video(self) -> None:
        body = self._wrap(
            [
                {"type": "output_image", "image_url": "http://x/a.png"},
                {"type": "output_video", "video_url": "http://x/b.mp4"},
                {"type": "output_image", "image_url": "http://x/c.png"},
            ]
        )
        assert _count_generated_media(body) == 3

    def test_ignores_non_media_content_types(self) -> None:
        body = self._wrap(
            [
                {"type": "output_text", "text": "hello"},
                {"type": "refusal", "refusal": "no"},
                {"type": "reasoning_summary", "summary": "thinking"},
            ]
        )
        assert _count_generated_media(body) == 0

    def test_empty_output(self) -> None:
        assert _count_generated_media({"output": []}) == 0

    def test_missing_output_key(self) -> None:
        assert _count_generated_media({}) == 0

    def test_malformed_output_not_a_list(self) -> None:
        assert _count_generated_media({"output": "not a list"}) == 0


class TestValidateTaskAndEndpoint:
    """Tests for validate_task_and_endpoint."""

    @pytest.mark.parametrize(
        ("task", "endpoint"),
        [
            pytest.param(
                "text-generation", "/v1/chat/completions", id="text_gen_chat"
            ),
            pytest.param(
                "text-generation", "/v1/completions", id="text_gen_completions"
            ),
            pytest.param(
                "text-to-image", "/v1/responses", id="pixel_responses"
            ),
            pytest.param(
                "text-to-image", "/v1/images/generations", id="pixel_images"
            ),
            pytest.param(
                "text-to-image", "/v1/chat/completions", id="pixel_chat"
            ),
            pytest.param(
                "text-to-video", "/v1/videos/sync", id="pixel_videos_sync"
            ),
            # i2v is a video task and may use the video endpoints.
            pytest.param(
                "image-to-video", "/v1/videos/sync", id="i2v_videos_sync"
            ),
            pytest.param("text-to-video", "/v1/videos", id="pixel_videos"),
            pytest.param("image-to-video", "/v1/videos", id="i2v_videos"),
            pytest.param("image-to-image", "/v1/responses", id="i2i_responses"),
        ],
    )
    def test_accepts(self, task: BenchmarkTask, endpoint: Endpoint) -> None:
        validate_task_and_endpoint(task, endpoint)

    @pytest.mark.parametrize(
        ("task", "endpoint", "match"),
        [
            pytest.param(
                "text-generation",
                "/v1/responses",
                "does not support",
                id="text_gen_responses",
            ),
            pytest.param(
                "text-generation",
                "/v1/images/generations",
                "does not support",
                id="text_gen_images",
            ),
            pytest.param(
                "text-to-image",
                "/v1/completions",
                "requires --endpoint",
                id="pixel_completions",
            ),
            pytest.param(
                "text-generation",
                "/v1/videos/sync",
                "does not support",
                id="text_gen_videos_sync",
            ),
            pytest.param(
                "text-to-image",
                "/v1/videos/sync",
                "only valid for",
                id="t2i_videos_sync",
            ),
            pytest.param(
                "image-to-image",
                "/v1/videos/sync",
                "only valid for",
                id="i2i_videos_sync",
            ),
            pytest.param(
                "text-generation",
                "/v1/videos",
                "does not support",
                id="text_gen_videos",
            ),
            pytest.param(
                "text-to-image",
                "/v1/videos",
                "only valid for",
                id="t2i_videos",
            ),
            pytest.param(
                "image-to-image",
                "/v1/videos",
                "only valid for",
                id="i2i_videos",
            ),
        ],
    )
    def test_rejects(
        self, task: BenchmarkTask, endpoint: Endpoint, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            validate_task_and_endpoint(task, endpoint)


class TestMarkCancelledIfPastDeadline:
    """Tests for ``mark_cancelled_if_past_deadline``.

    A request cut off by benchmark end (its non-success result surfaces after
    the deadline) should be reclassified as cancelled rather than failed.
    """

    def test_failed_past_deadline_becomes_cancelled(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        past = time.perf_counter_ns() - int(1e9)
        out = RequestFuncOutput(success=False, error="swallowed timeout")
        with caplog.at_level(
            logging.INFO, logger="max.benchmark.benchmark_shared.request"
        ):
            result = mark_cancelled_if_past_deadline(out, past)
        assert result is out
        assert out.cancelled is True
        # The reclassification is logged so the user is aware the request was
        # cut off by benchmark end rather than silently dropped.
        assert any(
            "cut off by benchmark end" in rec.message for rec in caplog.records
        )

    def test_failed_before_deadline_stays_failed(self) -> None:
        future = time.perf_counter_ns() + int(60 * 1e9)
        out = RequestFuncOutput(success=False, error="real failure")
        mark_cancelled_if_past_deadline(out, future)
        assert out.cancelled is False

    def test_failed_unbounded_deadline_stays_failed(self) -> None:
        out = RequestFuncOutput(success=False, error="real failure")
        mark_cancelled_if_past_deadline(out, None)
        assert out.cancelled is False

    def test_success_past_deadline_untouched(self) -> None:
        past = time.perf_counter_ns() - int(1e9)
        out = RequestFuncOutput(success=True)
        mark_cancelled_if_past_deadline(out, past)
        assert out.cancelled is False


def test_attach_images_targets_first_user_message() -> None:
    """A leading system message must not receive the images."""
    img: OpenAIImage = {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,AAA"},
    }
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": "sys"}]},
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
    ]
    _attach_images_to_first_user_message(messages, [img])
    assert messages[0]["content"] == [{"type": "text", "text": "sys"}]
    assert messages[1]["content"][-1] == img


def test_attach_images_normalizes_string_content() -> None:
    """agentic-code passes plain-string content through, which used to raise
    AttributeError on append."""
    img: OpenAIImage = {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,AAA"},
    }
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "plain string"}
    ]
    _attach_images_to_first_user_message(messages, [img])
    assert messages[0]["content"] == [
        {"type": "text", "text": "plain string"},
        img,
    ]


def test_attach_images_without_user_message_is_a_noop() -> None:
    img: OpenAIImage = {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,AAA"},
    }
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": "sys"}]}
    ]
    _attach_images_to_first_user_message(messages, [img])
    assert messages[0]["content"] == [{"type": "text", "text": "sys"}]


def _chat_input(response_format: ResponseFormat | None) -> RequestFuncInput:
    return make_request_input(
        "http://localhost:8000/v1/chat/completions",
        model="m",
        prompt="hi",
        prompt_len=1,
        max_tokens=16,
        response_format=response_format,
    )


_PERSON_SCHEMA: ResponseFormat = {
    "type": "json_schema",
    "json_schema": {
        "name": "person",
        "schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        },
    },
}


def test_tag_lora_route_copies_the_adapter() -> None:
    routed = _chat_input(None)
    routed.lora_id = "adapter-a"
    output = RequestFuncOutput()
    tag_lora_route(output, routed)
    assert output.lora_id == "adapter-a"

    tag_lora_route(output, _chat_input(None))
    assert output.lora_id is None


@pytest.mark.parametrize(
    ("text", "conformed"),
    [
        ('{"name": "ada"}', True),
        # Well-formed JSON that the schema rejects.
        ('{"name": 7}', False),
        # A backend that ignored response_format and answered in prose. This
        # is the case the metric exists to catch.
        ("Sure! The name is Ada.", False),
        ("", False),
    ],
)
def test_tag_response_format_outcome_judges_the_response(
    text: str, conformed: bool
) -> None:
    output = RequestFuncOutput(success=True, generated_text=text)
    tag_response_format_outcome(output, _chat_input(_PERSON_SCHEMA))
    assert output.response_format_constrained
    assert output.response_format_conformed is conformed


def test_tag_response_format_outcome_leaves_unconstrained_unjudged() -> None:
    output = RequestFuncOutput(success=True, generated_text="anything at all")
    tag_response_format_outcome(output, _chat_input(None))
    assert not output.response_format_constrained
    assert output.response_format_conformed is None


@pytest.mark.parametrize(
    ("text", "conformed"), [('{"any": 1}', True), ("[1, 2]", False)]
)
def test_tag_response_format_outcome_holds_json_object_to_an_object(
    text: str, conformed: bool
) -> None:
    """json_object means any JSON *object*, which is what the server's
    permissive schema encodes -- a bare array does not satisfy it."""
    output = RequestFuncOutput(success=True, generated_text=text)
    tag_response_format_outcome(output, _chat_input({"type": "json_object"}))
    assert output.response_format_conformed is conformed


def test_tag_response_format_outcome_leaves_an_unusable_schema_unjudged() -> (
    None
):
    """The conformance checker fails open, so an uncompilable schema scores
    ``valid``. Counting that as conforming would report a perfect rate for a
    schema that was never enforced, so it has to stay unjudged."""
    broken: ResponseFormat = {
        "type": "json_schema",
        "json_schema": {"name": "broken", "schema": {"type": "not-a-type"}},
    }
    output = RequestFuncOutput(success=True, generated_text="prose, not json")
    tag_response_format_outcome(output, _chat_input(broken))
    assert output.response_format_constrained
    assert output.response_format_conformed is None


def test_tag_response_format_outcome_ignores_a_text_format() -> None:
    """``text`` constrains nothing, so there is no schema to judge against."""
    output = RequestFuncOutput(success=True, generated_text="prose")
    tag_response_format_outcome(output, _chat_input({"type": "text"}))
    assert output.response_format_constrained
    assert output.response_format_conformed is None


def test_stream_recorder_cancel_discards_partial_measurement() -> None:
    recorder = OpenAIStreamRecorder(prompt_len=12, chat=True)
    recorder.on_event('{"choices": [{"delta": {"content": "Hello"}}]}')
    recorder.on_event('{"choices": [{"delta": {"content": " world"}}]}')

    output = recorder.cancel()

    assert recorder.done
    assert output is recorder.output
    assert output.cancelled
    assert not output.success
    assert output.prompt_len == 0
    assert output.ttft == 0.0
    assert output.itl == []
    assert output.request_submit_time is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("chat", [False, True])
@pytest.mark.parametrize(
    "case",
    [
        "special_only",
        "measured",
        "missing_usage",
        "zero_usage",
        "missing_done",
        "http_error",
        "stream_error",
    ],
)
async def test_cache_warmup_empty_stream_requires_completed_generation(
    chat: bool,
    case: str,
    mock_aiohttp_session: MagicMock,
    mock_openai_env: None,
    mocker: MockerFixture,
) -> None:
    request_input = RequestFuncInput(
        model="test-model",
        session_id=None,
        sampling=SamplingConfig(),
        prompt="shared prefix",
        images=[],
        api_url="http://localhost:8000/"
        + ("chat/completions" if chat else "completions"),
        prompt_len=2,
        max_tokens=1,
        ignore_eos=True,
        is_cache_warmup=case != "measured",
    )
    choice: dict[str, object] = {"delta": {}} if chat else {"text": ""}
    events = [json.dumps({"choices": [choice]})]
    if case != "missing_usage":
        events.append(
            json.dumps(
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 2,
                        "completion_tokens": 0 if case == "zero_usage" else 1,
                    },
                }
            )
        )
    if case == "stream_error":
        events.append(json.dumps({"error": {"message": "generation failed"}}))
    if case != "missing_done":
        events.append("[DONE]")

    response = _streaming_response(
        mocker, [f"data: {event}\n\n".encode() for event in events]
    )
    response.status = 400 if case == "http_error" else 200
    response.reason = "Bad Request"
    mock_aiohttp_session.setup_post_response(response)
    driver = (
        OpenAIChatCompletionsRequestDriver()
        if chat
        else OpenAICompletionsRequestDriver()
    )
    result = await driver.request(request_input)

    assert result.success == (case == "special_only")
    assert result.generated_text == ""
    if result.success:
        assert result.server_token_stats.completion_tokens == 1
        assert result.latency > 0
        assert result.error == ""
    else:
        assert result.error
    payload = mock_aiohttp_session.post.call_args.kwargs["json"]
    assert "is_cache_warmup" not in payload
    assert payload["max_tokens"] == 1
    # Without it the server sends no usage chunk, and the warmup cannot tell a
    # special-token-only generation from an empty one.
    assert payload["stream_options"] == {"include_usage": True}


# A GLM-5.3-Flash stream captured at ``max_tokens=1``: the one token was
# ``</think>``, which the glm45 reasoning parser consumes, so no text reaches
# the client.
_DELIMITER_ONLY_STREAM = [
    json.dumps(
        {
            "id": "556c1d809b74424a829ca40d270f666b",
            "choices": [
                {
                    "delta": {"content": ""},
                    "finish_reason": "length",
                    "index": 0,
                    "token_ids": [154842],
                }
            ],
            "model": "zai-org/GLM-5.3-Flash",
            "object": "chat.completion.chunk",
        }
    ),
    json.dumps(
        {
            "id": "556c1d809b74424a829ca40d270f666b",
            "choices": [],
            "model": "zai-org/GLM-5.3-Flash",
            "object": "chat.completion.chunk",
            "usage": {
                "completion_tokens": 1,
                "prompt_tokens": 127092,
                "total_tokens": 127093,
                "completion_tokens_details": {"reasoning_tokens": 1},
                "prompt_tokens_details": {"cached_tokens": 126976},
            },
        }
    ),
    "[DONE]",
]


def _with_completion_tokens(events: list[str], tokens: int) -> list[str]:
    usage_chunk = json.loads(events[1])
    usage_chunk["usage"]["completion_tokens"] = tokens
    return [events[0], json.dumps(usage_chunk), events[2]]


def _with_finish_reason(events: list[str], reason: str) -> list[str]:
    choice_chunk = json.loads(events[0])
    choice_chunk["choices"][0]["finish_reason"] = reason
    return [json.dumps(choice_chunk), *events[1:]]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("max_tokens", "events", "success"),
    [
        pytest.param(1, _DELIMITER_ONLY_STREAM, True, id="captured"),
        pytest.param(
            2,
            _with_completion_tokens(_DELIMITER_ONLY_STREAM, 2),
            True,
            id="two_token_budget",
        ),
        pytest.param(128, _DELIMITER_ONLY_STREAM, False, id="large_budget"),
        pytest.param(
            128,
            _with_completion_tokens(_DELIMITER_ONLY_STREAM, 128),
            False,
            id="large_budget_exhausted",
        ),
        pytest.param(
            2, _DELIMITER_ONLY_STREAM, False, id="budget_not_exhausted"
        ),
        pytest.param(
            1,
            _with_finish_reason(_DELIMITER_ONLY_STREAM, "stop"),
            False,
            id="stopped",
        ),
        pytest.param(
            1, _DELIMITER_ONLY_STREAM[:1], False, id="missing_usage_and_done"
        ),
        pytest.param(None, _DELIMITER_ONLY_STREAM, False, id="no_budget"),
    ],
)
async def test_measured_empty_stream_succeeds_only_when_tiny_budget_is_spent(
    max_tokens: int | None,
    events: list[str],
    success: bool,
    mock_aiohttp_session: MagicMock,
    mock_openai_env: None,
    mocker: MockerFixture,
) -> None:
    request_input = RequestFuncInput(
        model="zai-org/GLM-5.3-Flash",
        session_id=None,
        sampling=SamplingConfig(),
        prompt="hi",
        images=[],
        api_url="http://localhost:8000/v1/chat/completions",
        prompt_len=127092,
        max_tokens=max_tokens,
        ignore_eos=True,
    )

    response = _streaming_response(
        mocker, [f"data: {event}\n\n".encode() for event in events]
    )
    mock_aiohttp_session.setup_post_response(response)
    result = await OpenAIChatCompletionsRequestDriver().request(request_input)

    assert result.success == success
    assert result.generated_text == ""
    if success:
        assert result.error == ""
        assert result.server_token_stats.completion_tokens == max_tokens
        assert 0.0 < result.ttft <= result.latency
    else:
        assert "No text content" in result.error
