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

"""Multi-turn benchmark request generation and execution."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import time
from collections.abc import Sequence

try:
    from asyncio import TaskGroup  # type: ignore[attr-defined]  # added in 3.11
except ImportError:
    from taskgroup import TaskGroup  # Python < 3.11 backport

import numpy as np
from max.benchmark.benchmark_shared.config import SamplingConfig
from max.benchmark.benchmark_shared.datasets import ChatSession
from max.benchmark.benchmark_shared.datasets.types import TextContentBlock
from max.benchmark.benchmark_shared.lora_benchmark_manager import (
    LoRABenchmarkManager,
)
from max.benchmark.benchmark_shared.request import (
    BaseRequestFuncInput,
    BaseRequestFuncOutput,
    ChatMessage,
    RequestCounter,
    RequestDriver,
    RequestFuncInput,
    RequestFuncOutput,
)
from max.benchmark.benchmark_shared.warmup import (
    log_warmup_sampling_report,
    pick_warmup_population,
)
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


async def chat_session_driver(
    model_id: str,
    api_url: str,
    request_driver: RequestDriver,
    request_counter: RequestCounter,
    chat_session: ChatSession,
    max_chat_len: int,
    sampling: SamplingConfig,
    ignore_first_turn_stats: bool = False,
    benchmark_should_end_time: int | None = None,
    randomize_session_start: bool = False,
    run_prefix: str | None = None,
    run_prefix_len: int = 0,
) -> list[RequestFuncOutput]:
    request_func_input = RequestFuncInput(
        model=model_id,
        session_id=str(chat_session.id),
        sampling=sampling,
        prompt=[],
        images=[],
        api_url=api_url,
        prompt_len=0,
        max_tokens=0,
        ignore_eos=True,
    )
    content_idx = 0  # Assume user initiates the conversation

    session_outputs: list[RequestFuncOutput] = []
    message_history: list[ChatMessage] = []
    chat_len = 0

    messages = chat_session.messages
    prefix_end_idx = chat_session.prefix_turns * 2
    applied_initial_sleep = False

    # Build prefix turns locally (no server round-trips). The first
    # measured turn sends the full history for KV cache prefill.
    while content_idx < prefix_end_idx and content_idx + 1 < len(messages):
        chat_len += messages[content_idx].num_tokens
        output_len = messages[content_idx + 1].num_tokens
        if chat_len + output_len > max_chat_len:
            logger.warning(
                f"Session {chat_session.id}: prefix exceeded max chat"
                f" length {max_chat_len}, no measured turns possible"
            )
            break

        user_prompt = messages[content_idx].content
        message_history.append(
            ChatMessage(
                role="user",
                content=[TextContentBlock(text=user_prompt)],
            )
        )
        # Synthetic placeholder for the assistant response.
        assistant_content = messages[content_idx + 1].content
        if not assistant_content:
            assistant_content = " ".join(["token"] * max(output_len, 1))
        message_history.append(
            ChatMessage(
                role="assistant",
                content=[TextContentBlock(text=assistant_content)],
            )
        )
        chat_len += output_len
        content_idx += 2

    # If prefix exhausted the chat length budget, skip measured turns.
    if content_idx < prefix_end_idx:
        return session_outputs

    while content_idx + 1 < len(messages):
        chat_len += messages[content_idx].num_tokens
        if content_idx == 0 and run_prefix:
            chat_len += run_prefix_len
        output_len = messages[content_idx + 1].num_tokens
        if chat_len + output_len > max_chat_len:
            logger.warning(
                f"Ending conversation: hitting max chat length {max_chat_len}"
            )
            break

        advance_request = request_counter.advance_until_max()
        if not advance_request:  # reached max_requests
            break

        user_prompt = messages[content_idx].content
        if content_idx == 0 and run_prefix:
            user_prompt = run_prefix + user_prompt
        message_history.append(
            ChatMessage(
                role="user",
                content=[TextContentBlock(text=user_prompt)],
            )
        )
        request_func_input.prompt = message_history
        request_func_input.prompt_len = chat_len
        request_func_input.max_tokens = output_len

        if not applied_initial_sleep:
            applied_initial_sleep = True
            if randomize_session_start:
                delay_ms = messages[content_idx + 1].delay_until_next_message
                if delay_ms and delay_ms > 0:
                    await asyncio.sleep(random.uniform(0, delay_ms) / 1000)

        if (
            benchmark_should_end_time is not None
            and time.perf_counter_ns() >= benchmark_should_end_time
        ):
            response = RequestFuncOutput(
                cancelled=True, request_submit_time=time.perf_counter()
            )
        else:
            raw_response = await request_driver.request(request_func_input)
            if not isinstance(raw_response, RequestFuncOutput):
                raise TypeError(
                    "Expected RequestFuncOutput in text-generation benchmark flow."
                )
            response = raw_response

        if not (ignore_first_turn_stats and content_idx == prefix_end_idx):
            session_outputs.append(response)

        if not response.success:
            if not response.cancelled:
                logger.error(
                    f"Ending chat session {chat_session.id} due to server"
                    f" error response: {response.error}"
                )
            break

        message_history.append(
            ChatMessage(
                role="assistant",
                content=[TextContentBlock(text=response.generated_text)],
            )
        )
        chat_len += output_len

        if delay_ms := messages[content_idx + 1].delay_until_next_message:
            await asyncio.sleep(delay_ms / 1000)

        content_idx += 2

    return session_outputs


async def prime_prefix_turns(
    sessions: Sequence[ChatSession],
    request_driver: RequestDriver,
    model_id: str,
    api_url: str,
    max_chat_len: int,
    sampling: SamplingConfig,
    max_sessions: int | None = None,
) -> None:
    """Prime the server's KV cache for sessions with prefix turns.

    Sends one request per session with the full prefix context and
    max_tokens=1. Runs before the benchmark timer so priming doesn't
    affect measured throughput or duration.

    Sessions beyond ``max_sessions`` are skipped because the multiturn /
    kv-cache-stress runners reset ``prefix_turns=0`` for them anyway
    (they represent new conversations arriving mid-benchmark).
    """
    if max_sessions is not None:
        sessions = sessions[:max_sessions]
    sessions_with_prefix = [s for s in sessions if s.prefix_turns > 0]
    if not sessions_with_prefix:
        return

    logger.info(
        f"Priming prefix turns for {len(sessions_with_prefix)} sessions..."
    )

    async def _prime_session(session: ChatSession) -> None:
        messages = session.messages
        prefix_end_idx = session.prefix_turns * 2
        message_history: list[ChatMessage] = []
        chat_len = 0
        content_idx = 0
        while content_idx < prefix_end_idx and content_idx + 1 < len(messages):
            chat_len += messages[content_idx].num_tokens
            output_len = messages[content_idx + 1].num_tokens
            if chat_len + output_len > max_chat_len:
                break
            message_history.append(
                ChatMessage(
                    role="user",
                    content=[
                        TextContentBlock(text=messages[content_idx].content)
                    ],
                )
            )
            assistant_content = messages[content_idx + 1].content
            if not assistant_content:
                assistant_content = " ".join(["token"] * max(output_len, 1))
            message_history.append(
                ChatMessage(
                    role="assistant",
                    content=[TextContentBlock(text=assistant_content)],
                )
            )
            chat_len += output_len
            content_idx += 2
        if message_history:
            prime_input = RequestFuncInput(
                model=model_id,
                session_id=str(session.id),
                sampling=sampling,
                prompt=message_history,
                images=[],
                api_url=api_url,
                prompt_len=chat_len,
                max_tokens=1,
                ignore_eos=True,
            )
            await request_driver.request(prime_input)

    await asyncio.gather(*(_prime_session(s) for s in sessions_with_prefix))
    logger.info("Prefix turns priming complete.")


async def run_multiturn_benchmark(
    *,
    chat_sessions: Sequence[ChatSession],
    max_requests: int,
    semaphore: contextlib.AbstractAsyncContextManager[None],
    benchmark_should_end_time: int | None,
    request_driver: RequestDriver,
    model_id: str,
    api_url: str,
    tokenizer: PreTrainedTokenizerBase,
    ignore_first_turn_stats: bool,
    lora_manager: LoRABenchmarkManager | None,
    warmup_delay_ms: float,
    max_concurrency: int | None,
    sampling: SamplingConfig,
    randomize_session_start: bool = False,
    warmup_to_steady_state: bool = False,
    warmup_oversample_factor: int = 0,
    num_chat_sessions: int = 0,
    seed: int | None = None,
    run_prefix: str | None = None,
    run_prefix_len: int = 0,
) -> dict[str, list[RequestFuncOutput]]:
    """Run multi-turn chat benchmark scenario."""

    # Track total sent requests among chat sessions
    request_counter = RequestCounter(
        max_requests=max_requests,
        total_sent_requests=0,
    )

    # apply the semaphore at the session level
    # ex: with max_concurrency = 1,
    # the first session finishes before the second session starts
    async def limited_chat_session_driver(
        chat_session: ChatSession,
        session_idx: int,
    ) -> tuple[str, list[RequestFuncOutput]]:
        # Determine which LoRA to use for this chat session
        lora_id = None
        if lora_manager:
            lora_id = lora_manager.get_lora_for_request(session_idx)

        async with semaphore:
            outputs = await chat_session_driver(
                model_id=model_id if lora_id is None else lora_id,
                api_url=api_url,
                request_driver=request_driver,
                request_counter=request_counter,
                chat_session=chat_session,
                max_chat_len=tokenizer.model_max_length,
                sampling=sampling,
                ignore_first_turn_stats=ignore_first_turn_stats,
                benchmark_should_end_time=benchmark_should_end_time,
                randomize_session_start=randomize_session_start,
                run_prefix=run_prefix,
                run_prefix_len=run_prefix_len,
            )
        session_id = (
            str(chat_session.id)
            if chat_session.id is not None
            else f"anonymous-{session_idx}"
        )
        return session_id, outputs

    sessions = list(chat_sessions)
    if warmup_to_steady_state:
        warmup_count = max_concurrency or len(chat_sessions)
        sessions, report = pick_warmup_population(
            chat_sessions,
            warmup_count,
            warmup_to_steady_state=True,
            warmup_oversample_factor=warmup_oversample_factor,
            main_pool_target=num_chat_sessions or len(chat_sessions),
            rng=np.random.default_rng(seed),
        )
        if report is not None:
            log_warmup_sampling_report(report)

    tasks: list[asyncio.Task[tuple[str, list[RequestFuncOutput]]]] = []
    for idx, chat_session in enumerate(sessions):
        if warmup_delay_ms > 0 and max_concurrency and idx < max_concurrency:
            await asyncio.sleep(warmup_delay_ms / 1000)
        tasks.append(
            asyncio.create_task(limited_chat_session_driver(chat_session, idx))
        )

    outputs_by_session: dict[str, list[RequestFuncOutput]] = dict(
        await asyncio.gather(*tasks)
    )

    if (
        benchmark_should_end_time is not None
        and time.perf_counter_ns() < benchmark_should_end_time
    ):
        logger.warning(
            "All chat sessions completed before the time limit. "
            "Consider increasing --num-chat-sessions for more stable load."
        )

    return outputs_by_session


class ConcurrentTurnsRequestDriver(RequestDriver):
    """Wraps a RequestDriver to cap the number of concurrent in-flight turns.

    Acquires a semaphore slot before issuing each turn request and releases it
    as soon as the response returns. Inter-turn delays (e.g. delay_until_next_message)
    fall outside the slot's hold window, so idle user-think-time does not consume
    concurrency capacity.

    With many concurrent conversations, a turn request may wait in the semaphore
    backlog long enough for the deadline to expire. Cancel it when stale.
    """

    def __init__(
        self,
        request_driver: RequestDriver,
        semaphore: contextlib.AbstractAsyncContextManager[None],
        benchmark_should_end_time: int | None = None,
    ) -> None:
        super().__init__(tokenizer=request_driver.tokenizer)
        self._request_driver = request_driver
        self._semaphore = semaphore
        self._benchmark_should_end_time = benchmark_should_end_time

    async def request(
        self, request_func_input: BaseRequestFuncInput
    ) -> BaseRequestFuncOutput:
        async with self._semaphore:
            if (
                self._benchmark_should_end_time is not None
                and time.perf_counter_ns() >= self._benchmark_should_end_time
            ):
                return request_func_input.get_output_type()(
                    cancelled=True, request_submit_time=time.perf_counter()
                )
            return await self._request_driver.request(request_func_input)


async def run_kv_cache_stress_benchmark(
    *,
    chat_sessions: Sequence[ChatSession],
    max_requests: int,
    max_concurrent_conversations: int,
    semaphore: contextlib.AbstractAsyncContextManager[None],
    benchmark_should_end_time: int | None,
    request_driver: RequestDriver,
    model_id: str,
    api_url: str,
    tokenizer: PreTrainedTokenizerBase,
    ignore_first_turn_stats: bool,
    lora_manager: LoRABenchmarkManager | None,
    warmup_delay_ms: float,
    sampling: SamplingConfig,
    randomize_session_start: bool = False,
    warmup_to_steady_state: bool = False,
    warmup_oversample_factor: int = 0,
    num_chat_sessions: int = 0,
    seed: int | None = None,
    run_prefix: str | None = None,
    run_prefix_len: int = 0,
) -> dict[str, list[RequestFuncOutput]]:
    """Run a KV-cache stress benchmark with independent conversation and turn concurrency.

    Two independent concurrency controls:

    - `max_concurrent_conversations`: at most this many chat sessions are
      driven at once. Workers pick up the next session from the queue when one
      finishes, growing the server's KV-cache footprint.
    - `semaphore` (`max_concurrency` in the CLI): caps the number of turn
      requests in-flight globally across all concurrent sessions. Workers that
      cannot acquire a turn slot block without sending a request; the session's
      `session_id` and client-side conversation state are preserved in the
      backlog until a slot becomes available.

    NOTE: TTFT reflects pure server-side cost (KV re-computation or reloading)
          since the timer starts only after the semaphore is acquired. Backlog
          wait reduces each session's firing cadence beyond what
          `delay_between_chat_turns` specifies — sessions are less frequent
          than configured.
    """
    request_counter = RequestCounter(
        max_requests=max_requests,
        total_sent_requests=0,
    )

    request_driver = ConcurrentTurnsRequestDriver(
        request_driver, semaphore, benchmark_should_end_time
    )

    sessions = list(chat_sessions)
    if warmup_to_steady_state:
        sessions, report = pick_warmup_population(
            chat_sessions,
            max_concurrent_conversations,
            warmup_to_steady_state=True,
            warmup_oversample_factor=warmup_oversample_factor,
            main_pool_target=num_chat_sessions or len(chat_sessions),
            rng=np.random.default_rng(seed),
        )
        if report is not None:
            log_warmup_sampling_report(report)

    # Queue holds (original_index, session) pairs so LoRA assignment is stable.
    session_queue: asyncio.Queue[tuple[int, ChatSession]] = asyncio.Queue()
    for idx, session in enumerate(sessions):
        await session_queue.put((idx, session))

    num_workers = min(max_concurrent_conversations, len(sessions))
    worker_outputs: list[dict[str, list[RequestFuncOutput]]] = [
        {} for _ in range(num_workers)
    ]

    async def _conversation_worker(worker_idx: int) -> None:
        # Stagger workers to avoid thundering-herd at startup.
        if warmup_delay_ms > 0:
            await asyncio.sleep(worker_idx * warmup_delay_ms / 1000)

        local_count = 0
        while True:
            try:
                idx, chat_session = session_queue.get_nowait()
            except asyncio.QueueEmpty:
                return

            lora_id = (
                lora_manager.get_lora_for_request(idx) if lora_manager else None
            )
            outputs = await chat_session_driver(
                model_id=model_id if lora_id is None else lora_id,
                api_url=api_url,
                request_driver=request_driver,
                request_counter=request_counter,
                chat_session=chat_session,
                max_chat_len=tokenizer.model_max_length,
                sampling=sampling,
                ignore_first_turn_stats=ignore_first_turn_stats,
                benchmark_should_end_time=benchmark_should_end_time,
                randomize_session_start=randomize_session_start,
                run_prefix=run_prefix,
                run_prefix_len=run_prefix_len,
            )
            session_id = (
                str(chat_session.id)
                if chat_session.id is not None
                else f"anonymous-w{worker_idx}-{local_count}"
            )
            local_count += 1
            worker_outputs[worker_idx].setdefault(session_id, []).extend(
                outputs
            )

    async with TaskGroup() as tg:
        for i in range(num_workers):
            tg.create_task(_conversation_worker(i))

    outputs_by_session: dict[str, list[RequestFuncOutput]] = {}
    for worker_dict in worker_outputs:
        for sid, outs in worker_dict.items():
            outputs_by_session.setdefault(sid, []).extend(outs)
    return outputs_by_session
