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
"""Tests for benchmark_shared.multi_turn."""

from __future__ import annotations

import asyncio
import dataclasses
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from max.benchmark.benchmark_shared.config import SamplingConfig
from max.benchmark.benchmark_shared.datasets.types import (
    ChatSession,
    SessionMessage,
    TextContentBlock,
)
from max.benchmark.benchmark_shared.multi_turn import (
    ConcurrentTurnsRequestDriver,
    chat_session_driver,
    prerun_warmup_turns,
    run_chat_judge_benchmark,
)
from max.benchmark.benchmark_shared.request import (
    BaseRequestFuncInput,
    RequestCounter,
    RequestDriver,
    RequestFuncInput,
    RequestFuncOutput,
)


class _CapturingDriver(RequestDriver):
    """Request driver that records all requests and returns success."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[RequestFuncInput] = []

    async def request(
        self, request_func_input: BaseRequestFuncInput
    ) -> RequestFuncOutput:
        assert isinstance(request_func_input, RequestFuncInput)
        self.calls.append(request_func_input)
        return RequestFuncOutput(
            success=True,
            latency=0.1,
            ttft=0.05,
            prompt_len=request_func_input.prompt_len,
            generated_text="ok",
        )


def _make_4turn_session(
    prefix_turns: int = 0,
    delay_ms: float = 1000.0,
) -> ChatSession:
    """Create a 4-turn chat session for testing prefix_turns behavior."""
    return ChatSession(
        id=0,
        messages=[
            SessionMessage(source="user", content="Turn 1", num_tokens=5),
            SessionMessage(
                source="assistant",
                content="",
                num_tokens=5,
                delay_until_next_message=delay_ms,
            ),
            SessionMessage(source="user", content="Turn 2", num_tokens=5),
            SessionMessage(
                source="assistant",
                content="",
                num_tokens=5,
                delay_until_next_message=delay_ms,
            ),
            SessionMessage(source="user", content="Turn 3", num_tokens=5),
            SessionMessage(
                source="assistant",
                content="",
                num_tokens=5,
                delay_until_next_message=delay_ms,
            ),
            SessionMessage(source="user", content="Turn 4", num_tokens=5),
            SessionMessage(
                source="assistant",
                content="",
                num_tokens=5,
            ),
        ],
        prefix_turns=prefix_turns,
    )


def _make_session_with_id(session_id: int, prefix_turns: int) -> ChatSession:
    """Helper: 4-turn session with a specific id."""
    session = _make_4turn_session(prefix_turns=prefix_turns)
    return dataclasses.replace(session, id=session_id)


def test_chat_session_driver_forwards_sampling_params() -> None:
    """Test that chat_session_driver forwards temperature, top_p, top_k."""

    captured_inputs: list[RequestFuncInput] = []

    class CapturingDriver(RequestDriver):
        async def request(
            self, request_func_input: BaseRequestFuncInput
        ) -> RequestFuncOutput:
            assert isinstance(request_func_input, RequestFuncInput)
            captured_inputs.append(request_func_input)
            return RequestFuncOutput(
                success=True,
                latency=0.1,
                ttft=0.05,
                prompt_len=request_func_input.prompt_len,
                generated_text="Hello",
            )

    async def run_test() -> None:
        chat_session = ChatSession(
            id=0,
            messages=[
                SessionMessage(source="user", content="Hi", num_tokens=5),
                SessionMessage(
                    source="assistant", content="Hello", num_tokens=5
                ),
            ],
        )
        request_counter = RequestCounter(max_requests=10, total_sent_requests=0)

        await chat_session_driver(
            model_id="test-model",
            api_url="http://localhost:8000/v1/chat/completions",
            request_driver=CapturingDriver(),
            request_counter=request_counter,
            chat_session=chat_session,
            max_chat_len=4096,
            sampling=SamplingConfig(temperature=0.7, top_p=0.9, top_k=50),
        )

    asyncio.run(run_test())

    assert len(captured_inputs) == 1
    assert captured_inputs[0].sampling.temperature == 0.7
    assert captured_inputs[0].sampling.top_p == 0.9
    assert captured_inputs[0].sampling.top_k == 50


def test_chat_session_driver_run_prefix_prepends_first_turn() -> None:
    """First user message gets the run prefix when run_prefix is set."""

    captured: list[RequestFuncInput] = []

    class CapturingDriver(RequestDriver):
        async def request(
            self, request_func_input: BaseRequestFuncInput
        ) -> RequestFuncOutput:
            assert isinstance(request_func_input, RequestFuncInput)
            captured.append(request_func_input)
            return RequestFuncOutput(
                success=True,
                latency=0.1,
                ttft=0.05,
                prompt_len=request_func_input.prompt_len,
                generated_text="Hello",
            )

    async def run_test() -> None:
        chat_session = ChatSession(
            id=0,
            messages=[
                SessionMessage(source="user", content="Hi", num_tokens=5),
                SessionMessage(
                    source="assistant", content="Hello", num_tokens=5
                ),
                SessionMessage(source="user", content="Again", num_tokens=5),
                SessionMessage(source="assistant", content="Hi", num_tokens=5),
            ],
        )
        request_counter = RequestCounter(max_requests=10, total_sent_requests=0)

        await chat_session_driver(
            model_id="test-model",
            api_url="http://localhost:8000/v1/chat/completions",
            request_driver=CapturingDriver(),
            request_counter=request_counter,
            chat_session=chat_session,
            max_chat_len=4096,
            sampling=SamplingConfig(),
            run_prefix="RUN-UUID: ",
            run_prefix_len=4,
        )

    asyncio.run(run_test())

    assert len(captured) == 2
    assert isinstance(captured[0].prompt, list)
    first_user_content_block = captured[0].prompt[0].content[0]
    assert isinstance(first_user_content_block, TextContentBlock)
    first_user_text = first_user_content_block.text
    assert first_user_text.endswith("Hi")
    assert first_user_text != "Hi"
    assert isinstance(captured[1].prompt, list)
    second_user_content_block = captured[1].prompt[2].content[0]
    assert isinstance(second_user_content_block, TextContentBlock)
    second_user_text = second_user_content_block.text
    assert second_user_text == "Again"


def test_prefix_turns_excluded_from_results() -> None:
    """With prefix_turns=2, a 4-turn session should return only 2 results."""

    async def run_test() -> list[RequestFuncOutput]:
        session = _make_4turn_session(prefix_turns=2)
        counter = RequestCounter(max_requests=100)
        driver = _CapturingDriver()
        return await chat_session_driver(
            model_id="test",
            api_url="http://localhost:8000/v1/chat/completions",
            request_driver=driver,
            request_counter=counter,
            chat_session=session,
            max_chat_len=4096,
            sampling=SamplingConfig(),
        )

    outputs = asyncio.run(run_test())
    assert len(outputs) == 2


def test_prefix_turns_dont_count_against_max_requests() -> None:
    """Prefix turns should not consume max_requests budget."""

    async def run_test() -> tuple[list[RequestFuncOutput], int, int]:
        session = _make_4turn_session(prefix_turns=2)
        counter = RequestCounter(max_requests=2)
        driver = _CapturingDriver()
        outputs = await chat_session_driver(
            model_id="test",
            api_url="http://localhost:8000/v1/chat/completions",
            request_driver=driver,
            request_counter=counter,
            chat_session=session,
            max_chat_len=4096,
            sampling=SamplingConfig(),
        )
        return outputs, len(driver.calls), counter.total_sent_requests

    outputs, total_calls, counter_value = asyncio.run(run_test())
    # Prefix turns are built locally, so only measured turns hit the server.
    assert total_calls == 2
    assert counter_value == 2
    assert len(outputs) == 2


def test_prefix_turns_no_server_or_delays() -> None:
    """Prefix turns are built locally: no server calls, no delays."""
    sleep_calls: list[float] = []

    async def mock_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    async def run_test() -> int:
        session = _make_4turn_session(prefix_turns=2, delay_ms=5000.0)
        counter = RequestCounter(max_requests=100)
        driver = _CapturingDriver()
        with patch("asyncio.sleep", side_effect=mock_sleep):
            await chat_session_driver(
                model_id="test",
                api_url="http://localhost:8000/v1/chat/completions",
                request_driver=driver,
                request_counter=counter,
                chat_session=session,
                max_chat_len=4096,
                sampling=SamplingConfig(),
            )
        return len(driver.calls)

    total_calls = asyncio.run(run_test())
    # Prefix turns don't hit the server.
    assert total_calls == 2
    # Only turn 3 has a delay (turn 4 has no delay_until_next_message).
    assert len(sleep_calls) == 1
    assert sleep_calls[0] == pytest.approx(5.0)


def test_prefix_turns_zero_is_noop() -> None:
    """prefix_turns=0 should behave identically to the old code."""

    async def run_test() -> list[RequestFuncOutput]:
        session = _make_4turn_session(prefix_turns=0)
        counter = RequestCounter(max_requests=100)
        driver = _CapturingDriver()
        return await chat_session_driver(
            model_id="test",
            api_url="http://localhost:8000/v1/chat/completions",
            request_driver=driver,
            request_counter=counter,
            chat_session=session,
            max_chat_len=4096,
            sampling=SamplingConfig(),
        )

    outputs = asyncio.run(run_test())
    assert len(outputs) == 4


def test_prerun_warmup_turns_one_request_per_session_with_prefix() -> None:
    """Each session with prefix_turns>0 produces exactly one warmup request."""
    sessions = [
        _make_session_with_id(0, prefix_turns=0),
        _make_session_with_id(1, prefix_turns=2),
        _make_session_with_id(2, prefix_turns=0),
        _make_session_with_id(3, prefix_turns=3),
    ]
    driver = _CapturingDriver()

    async def run() -> None:
        await prerun_warmup_turns(
            sessions=sessions,
            request_driver=driver,
            model_id="test",
            api_url="http://localhost:8000/v1/chat/completions",
            max_chat_len=4096,
            sampling=SamplingConfig(),
        )

    asyncio.run(run())
    # Sessions 1 and 3 fire one request each; sessions with
    # prefix_turns=0 fire nothing.
    assert len(driver.calls) == 2
    assert {call.session_id for call in driver.calls} == {"1", "3"}
    assert all(call.ignore_eos is True for call in driver.calls)


def test_prerun_warmup_turns_request_prompt_is_last_turn_prefix() -> None:
    """For prefix_turns=N, the one request's prompt = dataset messages[0:2N-1]."""
    session = _make_session_with_id(0, prefix_turns=3)
    msgs = session.messages
    driver = _CapturingDriver()

    async def run() -> None:
        await prerun_warmup_turns(
            sessions=[session],
            request_driver=driver,
            model_id="test",
            api_url="http://localhost:8000/v1/chat/completions",
            max_chat_len=4096,
            sampling=SamplingConfig(),
        )

    asyncio.run(run())
    assert len(driver.calls) == 1
    call = driver.calls[0]
    # prefix_turns=3 means the prompt is messages[0:5]:
    # [user_1, asst_1, user_2, asst_2, user_3].
    assert len(call.prompt) == 5
    last = call.prompt[-1]
    assert not isinstance(last, str)
    block = last.content[0]
    assert isinstance(block, TextContentBlock)
    assert block.text == msgs[4].content  # user_3
    assert call.max_tokens == msgs[5].num_tokens  # dataset turn-3 output


def test_prerun_warmup_turns_cross_session_parallelism() -> None:
    """Requests for different sessions fire concurrently."""

    enter_count = 0
    max_concurrent_seen = 0
    in_flight = 0
    release = asyncio.Event()

    class _SlowDriver(RequestDriver):
        async def request(
            self, request_func_input: BaseRequestFuncInput
        ) -> RequestFuncOutput:
            nonlocal enter_count, max_concurrent_seen, in_flight
            enter_count += 1
            in_flight += 1
            max_concurrent_seen = max(max_concurrent_seen, in_flight)
            try:
                if enter_count >= 3:
                    release.set()
                await release.wait()
            finally:
                in_flight -= 1
            assert isinstance(request_func_input, RequestFuncInput)
            return RequestFuncOutput(
                success=True,
                latency=0.0,
                ttft=0.0,
                prompt_len=request_func_input.prompt_len,
                generated_text="ok",
            )

    # 3 sessions, one request each. All 3 should overlap.
    sessions = [_make_session_with_id(i, prefix_turns=2) for i in range(3)]

    async def run() -> None:
        await prerun_warmup_turns(
            sessions=sessions,
            request_driver=_SlowDriver(),
            model_id="test",
            api_url="http://localhost:8000/v1/chat/completions",
            max_chat_len=4096,
            sampling=SamplingConfig(),
        )

    asyncio.run(run())
    # If serialised, only one would be in flight before release fires.
    assert max_concurrent_seen == 3


def test_prerun_warmup_turns_noop_without_prefix_sessions() -> None:
    """When no session has prefix_turns > 0, no requests are issued."""
    sessions = [_make_session_with_id(idx, prefix_turns=0) for idx in range(3)]
    driver = _CapturingDriver()

    async def run() -> None:
        await prerun_warmup_turns(
            sessions=sessions,
            request_driver=driver,
            model_id="test",
            api_url="http://localhost:8000/v1/chat/completions",
            max_chat_len=4096,
            sampling=SamplingConfig(),
        )

    asyncio.run(run())
    assert driver.calls == []


def test_prerun_warmup_turns_respects_chat_length_budget() -> None:
    """Turns that would overflow max_chat_len are dropped."""
    # Each turn = 5 user + 5 asst tokens. With max_chat_len=15, only
    # turn 1 fits.
    session = _make_session_with_id(0, prefix_turns=3)
    driver = _CapturingDriver()

    async def run() -> None:
        await prerun_warmup_turns(
            sessions=[session],
            request_driver=driver,
            model_id="test",
            api_url="http://localhost:8000/v1/chat/completions",
            max_chat_len=15,
            sampling=SamplingConfig(),
        )

    asyncio.run(run())
    assert len(driver.calls) == 1


def _make_chat_judge_session(
    session_id: int,
    num_user_turns: int = 2,
) -> ChatSession:
    """Self-contained per-turn session with a system message at turn 0
    and `num_user_turns` user turns (no assistant turns)."""
    messages: list[SessionMessage] = [
        SessionMessage(
            source="system", content="You are a judge.", num_tokens=4
        )
    ]
    for i in range(num_user_turns):
        messages.append(
            SessionMessage(source="user", content=f"Item {i}", num_tokens=5)
        )
    return ChatSession(id=session_id, messages=messages)


@pytest.mark.asyncio
async def test_run_chat_judge_benchmark_smoke_with_system_role() -> None:
    """Orchestrator runs multiple sessions with system role end-to-end:
    outputs are collected per session and every call carries the
    system+user prompt without crashing through the TaskGroup wrapper."""
    driver = _CapturingDriver()
    sessions = [
        _make_chat_judge_session(0, num_user_turns=2),
        _make_chat_judge_session(1, num_user_turns=3),
    ]
    outputs_by_session = await run_chat_judge_benchmark(
        chat_sessions=sessions,
        max_output_tokens=16,
        max_requests=10,
        semaphore=asyncio.Semaphore(len(sessions)),
        benchmark_should_end_time=None,
        request_driver=driver,
        model_id="test-model",
        api_url="http://localhost:8000/v1/chat/completions",
        lora_manager=None,
        warmup_delay_ms=0.0,
        max_concurrency=None,
        sampling=SamplingConfig(),
    )

    assert set(outputs_by_session.keys()) == {"0", "1"}
    assert len(outputs_by_session["0"]) == 2
    assert len(outputs_by_session["1"]) == 3
    assert len(driver.calls) == 5
    for call in driver.calls:
        assert isinstance(call.prompt, list)
        assert len(call.prompt) == 2
        assert call.prompt[0].role == "system"
        assert call.prompt[1].role == "user"
        assert call.prompt_len == 4 + 5  # system + user tokens


def test_concurrent_turns_driver_expired_deadline_cancels_without_calling_base() -> (
    None
):
    """A turn that acquires the semaphore after the deadline is cancelled, not forwarded."""
    semaphore = asyncio.Semaphore(10)
    mock_driver = AsyncMock()
    mock_driver.tokenizer = None
    driver = ConcurrentTurnsRequestDriver(
        mock_driver,
        semaphore,
        benchmark_should_end_time=time.perf_counter_ns() - 1,
    )

    output_cls = MagicMock()
    output_cls.return_value = MagicMock()
    inp = MagicMock()
    inp.get_output_type.return_value = output_cls

    asyncio.run(driver.request(inp))

    assert output_cls.call_args.kwargs["cancelled"] is True
    mock_driver.request.assert_not_called()
