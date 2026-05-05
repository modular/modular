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

"""Benchmark online serving throughput."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
import os
import random
import shlex
import subprocess
import sys
import time
from collections.abc import (
    AsyncGenerator,
    Generator,
    Iterator,
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated
from urllib.parse import urlparse
from uuid import uuid4

try:
    from asyncio import TaskGroup  # type: ignore[attr-defined]  # added in 3.11
except ImportError:
    from taskgroup import TaskGroup  # Python < 3.11 backport

import numpy as np
import yaml
from cyclopts import App, Parameter
from cyclopts.config import Env
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

if TYPE_CHECKING:
    from max.benchmark.benchmark_shared.server_metrics import ParsedMetrics
    from max.diagnostics.gpu import BackgroundRecorder as GPUBackgroundRecorder
    from max.diagnostics.gpu import GPUStats

from max.benchmark.benchmark_shared.config import (
    CACHE_RESET_ENDPOINT_MAP,
    PIXEL_GEN_DEFAULT_ENDPOINT,
    PIXEL_GENERATION_ENDPOINTS,
    PIXEL_GENERATION_TASKS,
    Backend,
    BenchmarkTask,
    Endpoint,
    ServingBenchmarkConfig,
)
from max.benchmark.benchmark_shared.datasets import (
    ChatSession,
    SampledRequest,
)
from max.benchmark.benchmark_shared.datasets.all import sample_requests
from max.benchmark.benchmark_shared.datasets.types import (
    ChatSamples,
    PixelGenerationSampledRequest,
    RequestSamples,
    Samples,
    TextContentBlock,
)
from max.benchmark.benchmark_shared.lora_benchmark_manager import (
    LoRABenchmarkManager,
)
from max.benchmark.benchmark_shared.metrics import (
    PixelGenerationBenchmarkResult,
    SpecDecodeMetrics,
    TextGenerationBenchmarkResult,
    calculate_spec_decode_stats,
)
from max.benchmark.benchmark_shared.request import (
    BaseRequestFuncInput,
    BaseRequestFuncOutput,
    ChatMessage,
    PixelGenerationRequestFuncInput,
    PixelGenerationRequestFuncOutput,
    ProgressBarRequestDriver,
    RequestCounter,
    RequestDriver,
    RequestFuncInput,
    RequestFuncOutput,
    get_request_driver_class,
)
from max.benchmark.benchmark_shared.server_metrics import (
    collect_benchmark_metrics,
    fetch_spec_decode_metrics,
)
from max.benchmark.benchmark_shared.serving_metrics import (
    build_pixel_generation_result,
    build_text_generation_result,
    compute_output_len,
)
from max.benchmark.benchmark_shared.serving_result_output import (
    print_benchmark_summary,
    print_input_prompts,
    print_workload_stats,
    save_output_lengths,
    save_result_json,
)
from max.benchmark.benchmark_shared.utils import (
    argmedian,
    get_tokenizer,
    int_or_none,
    is_castable_to_int,
    parse_comma_separated,
    print_section,
    set_ulimit,
    wait_for_server_ready,
)
from max.benchmark.benchmark_shared.warmup import (
    log_warmup_sampling_report,
    pick_warmup_population,
)
from max.diagnostics.cpu import (
    CPUMetrics,
    CPUMetricsCollector,
    collect_pids_for_port,
)
from max.diagnostics.gpu import GPUDiagContext
from openai.types.chat.completion_create_params import ResponseFormat
from pydantic import TypeAdapter, ValidationError

BENCHMARK_SERVING_ARGPARSER_DESCRIPTION = (
    "This command runs comprehensive benchmark tests on a model server to"
    " measure performance metrics including throughput, latency, and resource"
    " utilization. Make sure that the MAX server is running and hosting a model"
    " before running this command."
)

logger = logging.getLogger(__name__)


def _prepend_run_prefix_to_formatted_prompt(
    prompt: str | list[ChatMessage], run_prefix: str
) -> str | list[ChatMessage]:
    """Return a new prompt with `run_prefix` prepended to the first message."""
    if isinstance(prompt, str):
        return run_prefix + prompt

    # Chat format: prepend to the text content of the first message.
    # content may be a plain string or a list of typed content blocks.
    if not prompt:
        raise ValueError("run_prefix: empty prompt list")
    msg = prompt[0]
    content = msg.content
    if isinstance(content, str):
        new_msg = ChatMessage(role=msg.role, content=run_prefix + content)
    elif isinstance(content, list):
        text_block_idx = next(
            (
                idx
                for idx, block in enumerate(content)
                if isinstance(block, TextContentBlock)
            ),
            None,
        )
        if text_block_idx is None:
            raise ValueError(
                "run_prefix: no text block found in content list; cannot"
                " prepend run prefix"
            )
        text_block = content[text_block_idx]
        assert isinstance(text_block, TextContentBlock)
        new_block = TextContentBlock(text=run_prefix + text_block.text)
        new_content = [
            *content[:text_block_idx],
            new_block,
            *content[text_block_idx + 1 :],
        ]
        new_msg = ChatMessage(role=msg.role, content=new_content)
    else:
        raise ValueError(
            "run_prefix: unsupported prompt shape for first message"
        )
    return [new_msg, *prompt[1:]]


def parse_response_format(arg: str) -> ResponseFormat:
    """Parse response format from CLI arg (inline JSON or @filepath).

    Args:
        arg: Either a JSON string or '@path/to/schema.json' to load from file.

    Returns:
        Validated ResponseFormat.

    Raises:
        ValueError: If the JSON is invalid, the file cannot be read, or the
            value does not match a recognised OpenAI response format.
    """
    if arg.startswith("@"):
        # Load from file
        file_path = Path(arg[1:])
        try:
            raw = file_path.read_text()
        except FileNotFoundError as e:
            raise ValueError(
                f"Response format file not found: {file_path}"
            ) from e
        try:
            return TypeAdapter(ResponseFormat).validate_json(raw)
        except (json.JSONDecodeError, ValidationError) as e:
            raise ValueError(
                f"Invalid response format in file {file_path}: {e}"
            ) from e

    # Parse inline JSON
    try:
        return TypeAdapter(ResponseFormat).validate_json(arg)
    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(f"Invalid response format: {e}") from e


def get_default_trace_path() -> str:
    """Get the default trace output path."""
    workspace_path = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if workspace_path:
        return os.path.join(workspace_path, "profile.nsys-rep")
    return "./profile.nsys-rep"


def assert_nvidia_gpu() -> None:
    """Raise an exception if no NVIDIA GPUs are available."""
    with GPUDiagContext() as ctx:
        stats = ctx.get_stats()
        if not stats:
            raise RuntimeError(
                "No GPUs detected. The --trace flag currently only works with NVIDIA GPUs."
            )
        if not any(gpu_name.startswith("nv") for gpu_name in stats):
            raise RuntimeError(
                "The --trace flag currently only works with NVIDIA GPUs. "
                f"Found GPUs: {list(stats.keys())}"
            )


@contextlib.contextmanager
def under_nsys_tracing(
    output_path: str, session_name: str | None = None
) -> Generator[None, None, None]:
    """Run some code under nsys tracing."""
    start_cmd = ["nsys", "start", "-o", output_path, "--force-overwrite=true"]
    stop_cmd = ["nsys", "stop"]
    if session_name:
        start_cmd.extend(["--session", session_name])
        stop_cmd.extend(["--session", session_name])
    logger.info(f"Starting nsys trace: {shlex.join(start_cmd)}")
    subprocess.run(start_cmd, check=True)
    try:
        yield
    finally:
        logger.info(f"Stopping nsys trace: {shlex.join(stop_cmd)}")
        subprocess.run(stop_cmd, check=True)


async def get_request(
    input_requests: Sequence[SampledRequest],
    request_rate: float,
    timing_data: dict[str, list[float]],
    burstiness: float = 1.0,
) -> AsyncGenerator[SampledRequest, None]:
    """
    Asynchronously generates requests at a specified rate
    with OPTIONAL burstiness.

    Args:
        input_requests:
            A list of input requests, each represented as a SampledRequest.
        request_rate:
            The rate at which requests are generated (requests/s).
        burstiness (optional):
            The burstiness factor of the request generation.
            Only takes effect when request_rate is not inf.
            Default value is 1, which follows a Poisson process.
            Otherwise, the request intervals follow a gamma distribution.
            A lower burstiness value (0 < burstiness < 1) results
            in more bursty requests, while a higher burstiness value
            (burstiness > 1) results in a more uniform arrival of requests.
        timing_data:
            Dictionary where timing data will be collected with keys:
            - 'intervals': List of actual time intervals between requests
    """

    # Calculate scale parameter theta to maintain the desired request_rate.
    assert burstiness > 0, (
        f"A positive burstiness factor is expected, but given {burstiness}."
    )
    theta = 1.0 / (request_rate * burstiness)

    # Initialize timing data collection - always enabled
    timing_data.setdefault("intervals", [])

    start_time = time.perf_counter()
    last_request_time = start_time

    for request in input_requests:
        current_time = time.perf_counter()

        # Record timestamp when request is yielded
        if last_request_time != start_time:
            actual_interval = current_time - last_request_time
            timing_data["intervals"].append(actual_interval)

        yield request

        # Update last_request_time for next iteration
        last_request_time = current_time

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue

        # Sample the request interval from the gamma distribution.
        # If burstiness is 1, it follows exponential distribution.
        interval = np.random.gamma(shape=burstiness, scale=theta)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


def build_single_turn_request_input(
    *,
    benchmark_task: BenchmarkTask,
    request: SampledRequest,
    model_id: str,
    lora_id: str | None,
    api_url: str,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    max_output_len: int | None,
    run_prefix: str | None = None,
    run_prefix_len: int = 0,
) -> BaseRequestFuncInput:
    request_model_id = model_id if lora_id is None else lora_id
    if benchmark_task == "text-generation":
        max_tokens = min(
            filter(None, (request.output_len, max_output_len)),
            default=None,
        )
        prompt = request.prompt_formatted
        prompt_len = request.prompt_len
        if run_prefix:
            prompt = _prepend_run_prefix_to_formatted_prompt(prompt, run_prefix)
            prompt_len = prompt_len + run_prefix_len
        return RequestFuncInput(
            model=request_model_id,
            session_id=None,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            prompt=prompt,
            images=request.encoded_images,
            api_url=api_url,
            prompt_len=prompt_len,
            max_tokens=max_tokens,
            ignore_eos=request.ignore_eos,
            response_format=request.response_format,
        )
    if benchmark_task in PIXEL_GENERATION_TASKS:
        if not isinstance(request, PixelGenerationSampledRequest):
            raise TypeError(
                "pixel-generation benchmark requires PixelGenerationSampledRequest."
            )
        prompt = request.prompt_formatted
        if run_prefix and isinstance(prompt, str):
            prompt = run_prefix + prompt
        return PixelGenerationRequestFuncInput(
            model=request_model_id,
            session_id=None,
            prompt=prompt,
            input_image_paths=request.input_image_paths,
            api_url=api_url,
            image_options=request.image_options,
        )
    raise ValueError(f"Unsupported benchmark task: {benchmark_task}")


async def chat_session_driver(
    model_id: str,
    api_url: str,
    request_driver: RequestDriver,
    request_counter: RequestCounter,
    chat_session: ChatSession,
    max_chat_len: int,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    ignore_first_turn_stats: bool = False,
    benchmark_should_end_time: int | None = None,
    randomize_session_start: bool = False,
    run_prefix: str | None = None,
    run_prefix_len: int = 0,
) -> list[RequestFuncOutput]:
    request_func_input = RequestFuncInput(
        model=model_id,
        session_id=str(chat_session.id),
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
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


async def run_single_turn_benchmark(
    *,
    input_requests: Sequence[SampledRequest],
    benchmark_task: BenchmarkTask,
    request_rate: float,
    burstiness: float,
    timing_data: dict[str, list[float]] | None,
    semaphore: contextlib.AbstractAsyncContextManager[None],
    benchmark_should_end_time: int | None,
    request_driver: RequestDriver,
    model_id: str,
    api_url: str,
    max_output_len: int | None,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    lora_manager: LoRABenchmarkManager | None,
    run_prefix: str | None = None,
    run_prefix_len: int = 0,
) -> list[BaseRequestFuncOutput]:
    """Run single-turn benchmark scenario."""
    if timing_data is None:
        timing_data = {}

    async def limited_request_func(
        request_func_input: BaseRequestFuncInput,
    ) -> BaseRequestFuncOutput:
        async with semaphore:
            if (
                benchmark_should_end_time is not None
                and time.perf_counter_ns() >= benchmark_should_end_time
            ):
                return request_func_input.get_output_type()(
                    cancelled=True, request_submit_time=time.perf_counter()
                )
            return await request_driver.request(request_func_input)

    tasks: list[asyncio.Task[BaseRequestFuncOutput]] = []
    request_idx = 0
    async for request in get_request(
        input_requests, request_rate, timing_data, burstiness
    ):
        # If we've hit the time limit, then don't issue any more requests
        if benchmark_should_end_time is not None:
            if time.perf_counter_ns() >= benchmark_should_end_time:
                break

        # Determine which LoRA to use for this request
        lora_id = None
        if lora_manager:
            lora_id = lora_manager.get_lora_for_request(request_idx)

        request_func_input = build_single_turn_request_input(
            benchmark_task=benchmark_task,
            request=request,
            model_id=model_id,
            lora_id=lora_id,
            api_url=api_url,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_len=max_output_len,
            run_prefix=run_prefix,
            run_prefix_len=run_prefix_len,
        )
        tasks.append(
            asyncio.create_task(limited_request_func(request_func_input))
        )
        request_idx += 1

    outputs = await asyncio.gather(*tasks)

    return outputs


async def prime_prefix_turns(
    sessions: Sequence[ChatSession],
    request_driver: RequestDriver,
    model_id: str,
    api_url: str,
    max_chat_len: int,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
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
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
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
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
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
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
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


class _ConcurrentTurnsRequestDriver(RequestDriver):
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
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
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

    request_driver = _ConcurrentTurnsRequestDriver(
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
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
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


def create_benchmark_pbar(disable_tqdm: bool, samples: Samples) -> tqdm | None:
    """Create a progress bar for benchmark runs.

    Args:
        disable_tqdm: Whether to disable the progress bar.
        samples: Samples that will be benchmarked with.

    Returns:
        A tqdm progress bar instance or None if disabled.
    """
    if disable_tqdm:
        return None

    if isinstance(samples, RequestSamples):
        # single-turn chat scenario
        return tqdm(total=len(samples.requests))
    else:
        # multi-turn chat scenario
        num_qa_turns = [session.num_turns for session in samples.chat_sessions]
        return tqdm(total=sum(num_qa_turns))


async def run_single_test_prompt(
    benchmark_task: BenchmarkTask,
    model_id: str,
    api_url: str,
    samples: Samples,
    request_driver: RequestDriver,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    max_output_len: int | None,
    run_prefix: str | None = None,
    run_prefix_len: int = 0,
) -> None:
    logger.info("Starting initial single prompt test run...")
    if isinstance(samples, ChatSamples):
        test_question = samples.chat_sessions[0].messages[0]
        test_answer = samples.chat_sessions[0].messages[1]
        test_request = SampledRequest(
            prompt_formatted=[
                ChatMessage(
                    role="user",
                    content=[TextContentBlock(text=test_question.content)],
                )
            ],
            prompt_len=test_question.num_tokens,
            output_len=test_answer.num_tokens,
            encoded_images=[],
            ignore_eos=True,
        )
        # Chat samples define their own target output length per turn.
        test_max_output_len = None
    else:
        test_request = samples.requests[0]
        test_max_output_len = max_output_len

    test_input = build_single_turn_request_input(
        benchmark_task=benchmark_task,
        request=test_request,
        model_id=model_id,
        lora_id=None,
        api_url=api_url,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_output_len=test_max_output_len,
        run_prefix=run_prefix,
        run_prefix_len=run_prefix_len,
    )
    test_output = await request_driver.request(test_input)
    if not test_output.success:
        raise ValueError(
            "Initial test run failed - Please make sure benchmark"
            " arguments are correctly specified. Error:"
            f" {test_output.error}"
        )
    else:
        logger.info(
            "Initial test run completed. Starting main benchmark run..."
        )


async def prime_shared_contexts(
    model_id: str,
    api_url: str,
    samples: Samples,
    request_driver: RequestDriver,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    run_prefix: str | None = None,
    run_prefix_len: int = 0,
) -> None:
    """Warm up prefix caching by sending each shared context for prefilling."""
    warmup_entries = samples.shared_contexts

    if not warmup_entries:
        logger.warning(
            "shared_contexts is empty; the prefix cache could not be primed."
            " Check that --random-sys-prompt-ratio > 0 and input lengths are"
            " sufficient to produce a non-trivial shared context."
        )
        return

    logger.info(
        f"Warming prefix cache with {len(warmup_entries)}"
        " unique shared context(s)..."
    )

    is_chat = isinstance(samples, ChatSamples)
    warmup_inputs: list[RequestFuncInput] = []
    for entry in warmup_entries:
        warmup_prompt: str | list[ChatMessage]
        if is_chat:
            warmup_prompt = [
                ChatMessage(
                    role="user",
                    content=[TextContentBlock(text=entry.text)],
                )
            ]
        else:
            warmup_prompt = entry.text

        if run_prefix:
            warmup_prompt = _prepend_run_prefix_to_formatted_prompt(
                warmup_prompt, run_prefix
            )

        warmup_inputs.append(
            RequestFuncInput(
                model=model_id,
                session_id=None,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                prompt=warmup_prompt,
                images=[],
                api_url=api_url,
                prompt_len=entry.num_tokens + run_prefix_len,
                max_tokens=1,
                ignore_eos=True,
            )
        )

    warmup_results: list[BaseRequestFuncOutput | None] = [None] * len(
        warmup_inputs
    )

    async def _run_warmup_index(idx: int, inp: RequestFuncInput) -> None:
        warmup_results[idx] = await request_driver.request(inp)

    warmup_start = time.perf_counter()
    async with TaskGroup() as tg:
        for idx, inp in enumerate(warmup_inputs):
            tg.create_task(_run_warmup_index(idx, inp))
    warmup_elapsed_s = time.perf_counter() - warmup_start
    for sys_idx, inp in enumerate(warmup_inputs):
        result = warmup_results[sys_idx]
        if result is None:
            raise RuntimeError(
                f"Warmup task {sys_idx} did not produce a result (this is a bug)"
            )
        if not result.success:
            raise ValueError(
                f"Shared context warmup request failed at index {sys_idx}:"
                f" (prompt: (SKIPPED), prompt_len: {inp.prompt_len}),"
                f" error: {result.error}"
            )

    logger.info(
        "Prefix cache warmup completed and took %.2f seconds.",
        warmup_elapsed_s,
    )


async def benchmark(
    args: ServingBenchmarkConfig,
    session: BenchmarkSession,
    max_concurrency: int | None,
    request_rate: float,
) -> TextGenerationBenchmarkResult | PixelGenerationBenchmarkResult:
    """Run a single benchmark invocation.

    ``session.orig_skip_first`` / ``session.orig_skip_last`` are the
    user-supplied values (``None`` = auto-derive from *max_concurrency*).
    """
    backend: Backend = args.backend

    skip_first = session.orig_skip_first
    skip_last = session.orig_skip_last
    if request_rate != float("inf"):
        # Finite rate → steady drip with no ramp-up / ramp-down artifacts,
        # so skip nothing (PERF-878).
        if skip_first is None:
            skip_first = 0
        if skip_last is None:
            skip_last = 0
    elif max_concurrency is not None and max_concurrency > 1:
        if skip_first is None:
            skip_first = max_concurrency
            logger.info(
                f"Auto-setting skip_first_n_requests={skip_first}"
                f" (max_concurrency={max_concurrency})"
            )
        if skip_last is None:
            skip_last = max_concurrency
            logger.info(
                f"Auto-setting skip_last_n_requests={skip_last}"
                f" (max_concurrency={max_concurrency})"
            )
    # max_concurrency=1 → sequential requests, no ramp-up to trim.
    # max_concurrency=None → no cap; default to 0.
    # Both leave auto values unset → fall through to 0 below.
    if skip_first is None:
        skip_first = 0
    if skip_last is None:
        skip_last = 0

    if args.warm_shared_prefix:
        if args.dataset_name not in ("random", "synthetic"):
            raise ValueError(
                f"--warm-shared-prefix is not supported for dataset"
                f" '{args.dataset_name}'. Only random/synthetic datasets have a"
                " defined shared prefix to cache."
            )
        if args.random_sys_prompt_ratio <= 0:
            raise ValueError(
                "--warm-shared-prefix requires --random-sys-prompt-ratio > 0."
            )

    logger.info("Starting benchmark run")
    assert args.num_prompts is not None

    if (
        args.ignore_first_turn_stats
        and skip_first
        and not args.warmup_to_steady_state
    ):
        # Without --warmup-to-steady-state, sessions all start at turn 0,
        # so --ignore-first-turn-stats already drops the same head requests
        # that --skip-first-n-requests would target. Combining them just
        # trims deeper into the run than the user asked for.
        # With --warmup-to-steady-state, sessions begin at randomized turn
        # offsets, so the two features filter different requests and we
        # want them to compose.
        logger.warning(
            "--ignore-first-turn-stats and --skip-first-n-requests both set"
            " without --warmup-to-steady-state. --ignore-first-turn-stats"
            " already drops every session's first turn, so"
            " --skip-first-n-requests would trim deeper than expected."
            " Ignoring --skip-first-n-requests."
        )
        skip_first = 0

    # Benchmark LoRA loading if manager provided
    if session.lora_manager:
        logger.info("Starting LoRA loading benchmark...")
        await session.lora_manager.benchmark_loading(
            api_url=session.base_url,
        )

    # Generate a single run-level unique prefix so all requests in this run
    # share the same constant prefix. This prevents cross-run KV-cache
    # pollution while preserving within-run system-prompt prefix caching
    # (requests with the same system prompt still share a common token prefix).
    run_prefix: str | None = None
    run_prefix_len: int = 0
    if args.force_unique_runs:
        if session.benchmark_task == "image-to-image":
            raise ValueError(
                "--force-unique-runs is not supported for image-to-image:"
                " the primary input is the image, not text, and systems may"
                " cache vision embeddings independently, so we can't guarantee"
                " uniqueness across benchmark runs."
            )
        run_prefix = f"{uuid4()}: "
        if session.benchmark_task not in PIXEL_GENERATION_TASKS:
            # prompt_len is not tracked for pixel generation tasks, so
            # run_prefix_len is not needed there.
            assert session.tokenizer is not None
            run_prefix_len = len(
                session.tokenizer.encode(run_prefix, add_special_tokens=False)
            )

    request_driver_class: type[RequestDriver] = get_request_driver_class(
        session.api_url, task=session.benchmark_task
    )
    # Create a request driver instance without pbar for test prompt
    # (pbar will be set later for the actual benchmark runs)
    test_request_driver: RequestDriver = request_driver_class(
        tokenizer=session.tokenizer
    )

    if args.warm_shared_prefix:
        await prime_shared_contexts(
            model_id=session.model_id,
            api_url=session.api_url,
            samples=session.samples,
            request_driver=test_request_driver,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            run_prefix=run_prefix,
            run_prefix_len=run_prefix_len,
        )

    if not args.skip_test_prompt:
        await run_single_test_prompt(
            benchmark_task=session.benchmark_task,
            model_id=session.model_id,
            api_url=session.api_url,
            samples=session.samples,
            request_driver=test_request_driver,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_output_len=args.max_output_len,
            run_prefix=run_prefix,
            run_prefix_len=run_prefix_len,
        )

    if args.burstiness == 1.0:
        distribution = "Poisson process"
    else:
        distribution = "Gamma distribution"

    logger.info(f"Input request rate: {request_rate}")
    logger.info(f"Burstiness factor: {args.burstiness} ({distribution})")
    logger.info(f"Maximum request concurrency: {max_concurrency}")

    semaphore: contextlib.AbstractAsyncContextManager[None]
    if max_concurrency:
        semaphore = asyncio.Semaphore(max_concurrency)
    else:
        semaphore = contextlib.nullcontext()

    with contextlib.ExitStack() as benchmark_stack:
        gpu_recorder: GPUBackgroundRecorder | None = None
        spec_decode_metrics_before: SpecDecodeMetrics | None = None
        spec_decode_metrics_after: SpecDecodeMetrics | None = None
        if args.collect_gpu_stats:
            try:
                from max.diagnostics.gpu import BackgroundRecorder
            except ImportError:
                logger.warning(
                    "max.diagnostics not available, skipping GPU stats"
                    " collection"
                )
            else:
                gpu_recorder = benchmark_stack.enter_context(
                    BackgroundRecorder()
                )

        cpu_collector = None
        if args.collect_cpu_stats:
            try:
                pids = collect_pids_for_port(
                    int(urlparse(session.api_url).port or 8000)
                )
                cpu_collector = benchmark_stack.enter_context(
                    CPUMetricsCollector(pids)
                )
            except Exception:
                logger.warning(
                    "Cannot access max-serve PIDs, skipping CPU stats"
                    " collection"
                )

        # Start nsys trace if enabled (before timing to exclude trace overhead)
        if session.trace_path is not None:
            benchmark_stack.enter_context(
                under_nsys_tracing(session.trace_path, args.trace_session)
            )

        # Create pbar for actual benchmark runs
        pbar = create_benchmark_pbar(
            disable_tqdm=args.disable_tqdm, samples=session.samples
        )

        # Create base driver and wrap with ProgressBarRequestDriver if pbar is provided
        request_driver = base_driver = request_driver_class(
            tokenizer=session.tokenizer
        )
        if pbar is not None:
            request_driver = ProgressBarRequestDriver(request_driver, pbar)

        # Prime prefix turns before the benchmark timer starts. Only the
        # initial concurrent population keeps its prefix_turns; sessions
        # arriving mid-benchmark get reset to 0 and don't need priming.
        # Bound: kv-cache-stress uses max_concurrent_conversations;
        # multiturn uses max_concurrency (may be None for unbounded, in
        # which case all sessions keep prefix_turns and are all primed).
        if isinstance(session.samples, ChatSamples):
            assert session.tokenizer is not None
            prime_bound = (
                args.max_concurrent_conversations
                if args.max_concurrent_conversations is not None
                else max_concurrency
            )
            await prime_prefix_turns(
                sessions=session.samples.chat_sessions,
                request_driver=base_driver,
                model_id=session.model_id,
                api_url=session.api_url,
                max_chat_len=session.tokenizer.model_max_length,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_sessions=prime_bound,
            )

        # Capture baseline server metrics after priming so priming requests
        # don't affect the delta calculation.
        baseline_endpoints: Mapping[str, ParsedMetrics] = {}
        if args.collect_server_stats:
            try:
                baseline_endpoints = collect_benchmark_metrics(
                    args.metrics_urls, backend, session.base_url
                )
                logger.info("Captured baseline server metrics")
            except Exception as e:
                logger.warning(
                    f"Failed to capture baseline server metrics: {e}"
                )

        if session.benchmark_task == "text-generation":
            spec_decode_metrics_before = fetch_spec_decode_metrics(
                backend, session.base_url
            )

        # Marker consumed by utils/benchmarking/serving/analyze_batch_logs.py
        # to slice the batch log by concurrency and exclude warmup/test-prompt
        # phases.
        logger.info(
            f"=== BATCH LOG MARKER: Benchmark started "
            f"(max_concurrency={max_concurrency}, "
            f"request_rate={request_rate}) ==="
        )
        benchmark_start_time = time.perf_counter_ns()
        if args.max_benchmark_duration_s is None:
            benchmark_should_end_time = None
        else:
            benchmark_should_end_time = benchmark_start_time + int(
                args.max_benchmark_duration_s * 1e9
            )

        all_outputs: Sequence[BaseRequestFuncOutput]
        outputs_by_session: dict[str, list[RequestFuncOutput]] | None = None
        if isinstance(session.samples, RequestSamples):
            if args.max_concurrent_conversations is not None:
                raise ValueError(
                    "--max-concurrent-conversations is only valid for "
                    "multi-turn workloads. Set --num-chat-sessions to "
                    "enable multi-turn mode."
                )
            # single-turn chat scenario
            all_outputs = await run_single_turn_benchmark(
                input_requests=session.samples.requests,
                benchmark_task=session.benchmark_task,
                request_rate=request_rate,
                burstiness=args.burstiness,
                timing_data=None,
                semaphore=semaphore,
                benchmark_should_end_time=benchmark_should_end_time,
                request_driver=request_driver,
                model_id=session.model_id,
                api_url=session.api_url,
                max_output_len=args.max_output_len,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                lora_manager=session.lora_manager,
                run_prefix=run_prefix,
                run_prefix_len=run_prefix_len,
            )
        elif args.max_concurrent_conversations is not None:
            # KV-cache stress benchmark: two independent concurrency knobs.
            # max_concurrent_conversations caps active session workers;
            # max_concurrency (semaphore) caps in-flight turns globally.
            if (
                max_concurrency is not None
                and max_concurrency > args.max_concurrent_conversations
            ):
                raise ValueError(
                    f"--max-concurrency ({max_concurrency}) must be <= "
                    f"--max-concurrent-conversations "
                    f"({args.max_concurrent_conversations}): to stress the "
                    "server's KV-cache, more sessions must be open than "
                    "turns in-flight."
                )
            assert session.tokenizer is not None
            assert isinstance(args.max_concurrent_conversations, int)
            outputs_by_session = await run_kv_cache_stress_benchmark(
                chat_sessions=session.samples.chat_sessions,
                max_requests=args.num_prompts,
                max_concurrent_conversations=args.max_concurrent_conversations,
                semaphore=semaphore,
                benchmark_should_end_time=benchmark_should_end_time,
                request_driver=request_driver,
                model_id=session.model_id,
                api_url=session.api_url,
                tokenizer=session.tokenizer,
                ignore_first_turn_stats=args.ignore_first_turn_stats,
                lora_manager=session.lora_manager,
                warmup_delay_ms=args.chat_warmup_delay_ms,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                randomize_session_start=args.randomize_session_start,
                warmup_to_steady_state=args.warmup_to_steady_state,
                warmup_oversample_factor=args.warmup_oversample_factor,
                num_chat_sessions=args.num_chat_sessions or 0,
                seed=args.seed,
                run_prefix=run_prefix,
                run_prefix_len=run_prefix_len,
            )
            all_outputs = [
                out for outs in outputs_by_session.values() for out in outs
            ]
        else:
            # multi-turn chat scenario
            outputs_by_session = await run_multiturn_benchmark(
                chat_sessions=session.samples.chat_sessions,
                max_requests=args.num_prompts,
                semaphore=semaphore,
                benchmark_should_end_time=benchmark_should_end_time,
                request_driver=request_driver,
                model_id=session.model_id,
                api_url=session.api_url,
                tokenizer=session.tokenizer,
                ignore_first_turn_stats=args.ignore_first_turn_stats,
                lora_manager=session.lora_manager,
                warmup_delay_ms=args.chat_warmup_delay_ms,
                max_concurrency=max_concurrency,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                randomize_session_start=args.randomize_session_start,
                warmup_to_steady_state=args.warmup_to_steady_state,
                warmup_oversample_factor=args.warmup_oversample_factor,
                num_chat_sessions=args.num_chat_sessions or 0,
                seed=args.seed,
                run_prefix=run_prefix,
                run_prefix_len=run_prefix_len,
            )
            all_outputs = [
                out for outs in outputs_by_session.values() for out in outs
            ]

        # Close pbar if it was created
        if pbar is not None:
            pbar.close()

        benchmark_duration = (
            time.perf_counter_ns() - benchmark_start_time
        ) / 1e9

    if session.benchmark_task == "text-generation":
        spec_decode_metrics_after = fetch_spec_decode_metrics(
            backend, session.base_url
        )
    spec_decode_stats = None
    if (
        spec_decode_metrics_before is not None
        and spec_decode_metrics_after is not None
    ):
        spec_decode_stats = calculate_spec_decode_stats(
            spec_decode_metrics_before,
            spec_decode_metrics_after,
        )

    if args.print_inputs_and_outputs:
        if session.benchmark_task == "text-generation":
            assert session.tokenizer is not None
            print("Generated output text:")
            for req_id, output in enumerate(all_outputs):
                assert isinstance(output, RequestFuncOutput)
                output_len = compute_output_len(session.tokenizer, output)
                print(
                    {
                        "req_id": req_id,
                        "output_len": output_len,
                        "output": output.generated_text,
                    }
                )
        elif session.benchmark_task in PIXEL_GENERATION_TASKS:
            print("Generated pixel generation outputs:")
            for req_id, output in enumerate(all_outputs):
                assert isinstance(output, PixelGenerationRequestFuncOutput)
                print(
                    {
                        "req_id": req_id,
                        "num_generated_outputs": output.num_generated_outputs,
                        "latency_s": output.latency,
                        "success": output.success,
                        "error": output.error,
                    }
                )

    if session.lora_manager:
        await session.lora_manager.benchmark_unloading(
            api_url=session.base_url,
        )

    gpu_metrics: list[dict[str, GPUStats]] | None = None
    if args.collect_gpu_stats and gpu_recorder is not None:
        gpu_metrics = gpu_recorder.stats

    cpu_metrics_result: CPUMetrics | None = None
    if cpu_collector is not None:
        cpu_metrics_result = cpu_collector.get_stats()

    # Collect server-side metrics from Prometheus endpoint (with delta from baseline)
    endpoint_metrics: Mapping[str, ParsedMetrics] = {}
    if args.collect_server_stats:
        try:
            endpoint_metrics = collect_benchmark_metrics(
                args.metrics_urls,
                backend,
                session.base_url,
                baseline=baseline_endpoints,
            )
            logger.info("Collected server metrics (final)")
        except Exception as e:
            logger.warning(f"Failed to collect server metrics: {e}")

    achieved_request_rate = 0.0

    result: PixelGenerationBenchmarkResult | TextGenerationBenchmarkResult
    if session.benchmark_task in PIXEL_GENERATION_TASKS:
        result = build_pixel_generation_result(
            outputs=all_outputs,
            benchmark_duration=benchmark_duration,
            gpu_metrics=gpu_metrics,
            cpu_metrics=cpu_metrics_result,
            max_concurrency=max_concurrency,
            collect_gpu_stats=args.collect_gpu_stats,
            metrics_by_endpoint=endpoint_metrics,
        )
    else:
        text_result = build_text_generation_result(
            outputs=all_outputs,
            benchmark_duration=benchmark_duration,
            tokenizer=session.tokenizer,
            gpu_metrics=gpu_metrics,
            cpu_metrics=cpu_metrics_result,
            skip_first_n_requests=skip_first,
            skip_last_n_requests=skip_last,
            max_concurrency=max_concurrency,
            max_concurrent_conversations=args.max_concurrent_conversations,
            collect_gpu_stats=args.collect_gpu_stats,
            metrics_by_endpoint=endpoint_metrics,
            spec_decode_stats=spec_decode_stats,
        )
        if outputs_by_session is not None:
            result = dataclasses.replace(
                text_result,
                session_server_stats={
                    sid: [
                        dataclasses.asdict(out.server_token_stats)
                        for out in outs
                    ]
                    for sid, outs in sorted(
                        outputs_by_session.items(),
                        key=lambda kv: _session_sort_key(kv[0]),
                    )
                },
            )
        else:
            result = dataclasses.replace(
                text_result,
                aggregate_server_stats=[
                    dataclasses.asdict(out.server_token_stats)
                    for out in all_outputs
                    if isinstance(out, RequestFuncOutput)
                ],
            )
    if session.lora_manager is not None:
        result.lora_metrics = session.lora_manager.metrics

    print_benchmark_summary(
        metrics=result.metrics,
        request_rate=request_rate,
        max_concurrency=max_concurrency,
        achieved_request_rate=achieved_request_rate,
        collect_gpu_stats=args.collect_gpu_stats,
        collect_cpu_stats=args.collect_cpu_stats,
        spec_decode_stats=spec_decode_stats,
        lora_manager=session.lora_manager,
    )

    ok, validation_errors = result.validate_metrics()
    if not ok:
        for err in validation_errors:
            logger.error(f"Benchmark result validation failed: {err}")
        logger.info("finished benchmark run: Failed.")
        sys.exit(1)

    logger.info("finished benchmark run: Success.")
    return result


def validate_task_and_endpoint(
    benchmark_task: BenchmarkTask, endpoint: Endpoint
) -> None:
    if benchmark_task == "text-generation":
        if endpoint in ("/v1/responses", "/v1/images/generations"):
            raise ValueError(
                f"--benchmark-task text-generation does not support "
                f"--endpoint {endpoint}"
            )
    elif benchmark_task in PIXEL_GENERATION_TASKS:
        if endpoint not in PIXEL_GENERATION_ENDPOINTS:
            raise ValueError(
                f"--benchmark-task {benchmark_task} requires --endpoint"
                f" to be one of {sorted(PIXEL_GENERATION_ENDPOINTS)},"
                f" got {endpoint!r}"
            )


def _apply_workload_to_config(
    config: ServingBenchmarkConfig, workload: Mapping[str, object]
) -> None:
    """Set workload YAML values as fields on *config*.

    Keys are converted from kebab-case to snake_case.  Path objects are
    stringified and env vars in string values are expanded.

    Fields already in `config.model_fields_set` (i.e. explicitly provided
    by the caller, whether via CLI args or direct construction) are left
    unchanged so that CLI values always take precedence over workload YAML.
    """
    for k, v in workload.items():
        field_name = k.replace("-", "_")
        if field_name not in ServingBenchmarkConfig.model_fields:
            logger.warning(f"Ignoring unknown workload key: {k}")
            continue
        if field_name in config.model_fields_set:
            logger.info(
                f"CLI flag --{k} takes precedence over workload YAML"
                f" (CLI: {getattr(config, field_name)!r},"
                f" workload: {v!r})"
            )
            continue
        if isinstance(v, Path):
            v = str(v)
        elif isinstance(v, str):
            v = os.path.expandvars(v)
        logger.info(f"Applying workload YAML value: --{k}={v!r}")
        setattr(config, field_name, v)


def flush_prefix_cache(
    backend: Backend, host: str, port: int, dry_run: bool
) -> None:
    """Flush the serving engine's prefix cache via HTTP POST."""
    if backend not in CACHE_RESET_ENDPOINT_MAP:
        raise ValueError(
            f"Cannot flush prefix cache for {backend} backend: this backend"
            " does not support prefix cache flush."
        )
    import requests as _http_requests  # lazy - avoid hard dep for non-sweep use

    api_url = f"http://{host}:{port}{CACHE_RESET_ENDPOINT_MAP[backend]}"
    if dry_run:
        logger.info(f"Dry-run flush: POST {api_url}")
        return
    response = _http_requests.post(api_url)
    if response.status_code == 400:
        logger.warning(
            f"Prefix caching is not enabled on backend {backend} at {api_url};"
            " skipping cache flush."
        )
    elif response.status_code == 404:
        logger.warning(
            f"Prefix cache reset is not supported at {api_url} (HTTP 404);"
            " skipping cache flush."
        )
    elif response.status_code != 200:
        # Mammoth's proxy wraps engine 404s in a 502 with per-endpoint statuses
        # in the JSON body; treat unanimous 404s the same as a direct 404 above
        # (e.g. vLLM builds without /reset_prefix_cache exposed).
        try:
            body = response.json() if response.content else None
        except ValueError:
            body = None
        results = body.get("results") if isinstance(body, dict) else None
        if (
            isinstance(results, list)
            and results
            and all(
                isinstance(r, dict) and r.get("statusCode") == 404
                for r in results
            )
        ):
            logger.warning(
                f"Prefix cache reset is not supported at {api_url} "
                "(proxy reported 404 from all engine endpoints);"
                " skipping cache flush."
            )
            return
        raise RuntimeError(
            f"Failed to flush prefix cache for backend {backend} at {api_url}: "
            f"status={response.status_code} body={response.text}"
        )


@dataclass
class BenchmarkRunResult:
    """Result of one (max_concurrency, request_rate) benchmark configuration.

    Yielded by :func:`main_with_parsed_args` — one entry per (mc, rr) combo
    after median selection across ``num_iters`` iterations.
    """

    max_concurrency: int | None
    request_rate: float
    num_prompts: int
    result: (
        TextGenerationBenchmarkResult | PixelGenerationBenchmarkResult | None
    ) = None


@dataclass
class BenchmarkSession:
    """Resolved, session-level state shared across all sweep iterations.

    Created once after argument parsing / dataset loading in
    :func:`main_with_parsed_args` and threaded into each
    :func:`benchmark` call.
    """

    benchmark_task: BenchmarkTask
    endpoint: Endpoint
    api_url: str
    base_url: str
    model_id: str
    tokenizer_id: str
    tokenizer: PreTrainedTokenizerBase | None
    samples: Samples
    lora_manager: LoRABenchmarkManager | None
    trace_path: str | None
    orig_skip_first: int | None
    orig_skip_last: int | None


def _session_sort_key(sid: str) -> tuple[int, int, str]:
    """Sort numeric session ids first by integer value, then anonymous ids."""
    try:
        return (0, int(sid), "")
    except ValueError:
        return (1, 0, sid)


def main_with_parsed_args(
    args: ServingBenchmarkConfig,
) -> Iterator[BenchmarkRunResult]:
    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d %(levelname)s: %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    logger.info(args)

    if args.model is None:
        raise ValueError("--model is required when running benchmark")

    # ---- Workload YAML ----
    if args.workload_config:
        with open(args.workload_config) as workload_file:
            workload = yaml.safe_load(workload_file)
        # Resolve relative paths against the YAML's directory.
        for key in ("dataset-path", "output-lengths"):
            if workload.get(key) is not None:
                if is_castable_to_int(str(workload[key])):
                    continue
                path = Path(os.path.expandvars(workload[key]))
                if not path.is_absolute():
                    path = Path(args.workload_config).parent / path
                workload[key] = path
        # Resolve max_concurrency: CLI > YAML.
        yaml_max_concurrency = workload.pop("max-concurrency", None)
        if yaml_max_concurrency is not None and args.max_concurrency is None:
            args.max_concurrency = str(yaml_max_concurrency)
        # Resolve num_prompts: CLI > YAML > default (deferred).
        cli_num_prompts = args.num_prompts is not None
        yaml_num_prompts = workload.pop("num-prompts", None)
        if not cli_num_prompts:
            if yaml_num_prompts is not None:
                args.num_prompts = int(yaml_num_prompts)
        # Resolve max_benchmark_duration_s: CLI > YAML.
        w_duration = workload.pop("max-benchmark-duration-s", None)
        if w_duration is not None and args.max_benchmark_duration_s is None:
            args.max_benchmark_duration_s = int(w_duration)
        _apply_workload_to_config(args, workload)
        args.skip_test_prompt = True

    # Warn + default when nothing constrains run length (common to both paths).
    has_prompts = args.num_prompts is not None
    has_duration = args.max_benchmark_duration_s is not None
    has_multiplier = args.num_prompts_multiplier is not None
    # The multiplier dynamically computes num_prompts per-mc, but only
    # when no explicit duration also constrains the run.
    multiplier_will_resolve = has_multiplier and not has_duration
    if not has_prompts and not has_duration and not has_multiplier:
        logger.warning(
            "Neither --num-prompts nor --max-benchmark-duration-s is"
            " specified. Defaulting to --num-prompts 1000 and"
            " --max-benchmark-duration-s 300"
        )
        args.num_prompts = 1000
        args.max_benchmark_duration_s = 300
    elif not has_prompts and not multiplier_will_resolve:
        args.num_prompts = 1000

    # ---- Parse sweep ranges ----
    concurrency_range = parse_comma_separated(args.max_concurrency, int_or_none)
    request_rate_range = parse_comma_separated(args.request_rate, float)

    # When num_prompts_multiplier is active AND no explicit num_prompts or
    # duration constrains the run, dynamically compute num_prompts per
    # concurrency level.
    use_dynamic_num_prompts = (
        args.num_prompts_multiplier is not None
        and args.num_prompts is None
        and args.max_benchmark_duration_s is None
    )
    if use_dynamic_num_prompts:
        assert args.num_prompts_multiplier is not None
        max_mc = max(
            (mc for mc in concurrency_range if mc is not None), default=1
        )
        args.num_prompts = args.num_prompts_multiplier * max_mc
        # When using num_prompts_multiplier without explicit duration, default to
        # 300s timeout per MC config to prevent indefinitely long benchmark runs.
        logger.info(
            "Using --num-prompts-multiplier without --max-benchmark-duration-s."
            " Defaulting to 300s timeout per max-concurrency configuration."
        )
        args.max_benchmark_duration_s = 300

    # ``--dry-run`` falls through — handled after samples build.

    random.seed(args.seed)
    np.random.seed(args.seed)
    set_ulimit()
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    benchmark_task: BenchmarkTask = args.benchmark_task
    endpoint: Endpoint = args.endpoint

    # Auto-select the correct endpoint for pixel generation based on the
    # backend. Each pixel-gen backend requires a specific endpoint (e.g.,
    # sglang needs /v1/images/generations, vllm needs
    # /v1/chat/completions). We auto-select when the current endpoint
    # doesn't match this backend's expected pixel-gen endpoint.
    if benchmark_task in PIXEL_GENERATION_TASKS:
        backend_key = args.backend.removesuffix("-chat")
        if backend_key in PIXEL_GEN_DEFAULT_ENDPOINT:
            expected = PIXEL_GEN_DEFAULT_ENDPOINT[backend_key]
            if endpoint != expected:
                logger.info(
                    "Auto-selected endpoint %s for backend %s"
                    " (pixel generation task)",
                    expected,
                    args.backend,
                )
                endpoint = expected
        else:
            raise ValueError(
                f"Backend {args.backend!r} does not have a default"
                f" pixel-generation endpoint. Explicitly pass --endpoint"
                f" with one of {sorted(PIXEL_GENERATION_ENDPOINTS)}."
            )

    validate_task_and_endpoint(benchmark_task, endpoint)
    # chat is only meaningful for text-generation (enables chat template
    # formatting). For pixel generation via /v1/chat/completions
    # (vllm pixel gen), the pixel-gen code path ignores this flag.
    chat = endpoint == "/v1/chat/completions"

    if args.base_url is not None:
        base_url = args.base_url
    else:
        base_url = f"http://{args.host}:{args.port}"

    api_url = f"{base_url}{endpoint}"
    tokenizer: PreTrainedTokenizerBase | None = None

    if benchmark_task == "text-generation":
        logger.info(f"getting tokenizer. api url: {api_url}")
        tokenizer = get_tokenizer(
            tokenizer_id,
            model_max_length=args.model_max_length,
            trust_remote_code=args.trust_remote_code,
        )

    samples = sample_requests(
        args=args,
        benchmark_task=benchmark_task,
        tokenizer=tokenizer,
        chat=chat,
    )

    # Inject response_format into all sampled requests if specified
    if args.response_format is not None:
        response_format = parse_response_format(args.response_format)
        if isinstance(samples, RequestSamples):
            for request in samples.requests:
                request.response_format = response_format
            logger.info(
                f"Injected response_format into {len(samples.requests)} requests"
            )
        else:
            logger.warning(
                "response_format is only supported for single-turn benchmarks, "
                "ignoring for multi-turn chat sessions"
            )

    if args.print_workload_stats:
        print_workload_stats(samples)

    if args.print_inputs_and_outputs:
        print_input_prompts(samples)

    # Samples are ready; wait for the server before issuing any requests.
    if not args.dry_run:
        wait_for_server_ready(
            args.host, args.port, timeout_s=args.server_ready_timeout_s
        )

    # ---- Dry run: build dataset + show warmup-sampling preview ----
    if args.dry_run:
        if not args.print_workload_stats:
            print_workload_stats(samples)
        if isinstance(samples, ChatSamples) and args.warmup_to_steady_state:
            rng = np.random.default_rng(args.seed or 0)
            for mc in concurrency_range:
                warmup_count = (
                    args.max_concurrent_conversations
                    or mc
                    or len(samples.chat_sessions)
                )
                print_section(
                    title=f" Warmup sampling preview (max_concurrency={mc}) ",
                    char="=",
                )
                _, report = pick_warmup_population(
                    samples.chat_sessions,
                    warmup_count,
                    warmup_to_steady_state=True,
                    warmup_oversample_factor=args.warmup_oversample_factor,
                    main_pool_target=args.num_chat_sessions or 0,
                    rng=rng,
                )
                if report is not None:
                    log_warmup_sampling_report(report)
        for mc in concurrency_range:
            for rr in request_rate_range:
                print(
                    f"Dry run: model={args.model}"
                    f" host={args.host} port={args.port}"
                    f" endpoint={args.endpoint}"
                    f" max_concurrency={mc}"
                    f" request_rate={rr}"
                    f" num_prompts={args.num_prompts}"
                    f" max_benchmark_duration_s="
                    f"{args.max_benchmark_duration_s}"
                )
                yield BenchmarkRunResult(
                    max_concurrency=mc,
                    request_rate=rr,
                    num_prompts=args.num_prompts or 0,
                )
        return

    lora_manager = None
    if args.lora_paths:
        num_requests = (
            len(samples.requests)
            if isinstance(samples, RequestSamples)
            else len(samples.chat_sessions)
        )

        lora_manager = LoRABenchmarkManager(
            lora_paths=args.lora_paths,
            num_requests=num_requests,
            traffic_ratios=args.per_lora_traffic_ratio
            if args.per_lora_traffic_ratio
            else None,
            uniform_ratio=args.lora_uniform_traffic_ratio,
            seed=args.seed,
            max_concurrent_lora_ops=args.max_concurrent_lora_ops,
        )
        lora_manager.log_traffic_distribution()

    # Handle trace flag (once, before loop)
    trace_path = None
    if args.trace:
        assert_nvidia_gpu()
        trace_path = (
            args.trace_file if args.trace_file else get_default_trace_path()
        )
        logger.info(f"Tracing enabled, output: {trace_path}")

    session = BenchmarkSession(
        benchmark_task=benchmark_task,
        endpoint=endpoint,
        api_url=api_url,
        base_url=base_url,
        model_id=model_id,
        tokenizer_id=tokenizer_id,
        tokenizer=tokenizer,
        samples=samples,
        lora_manager=lora_manager,
        trace_path=trace_path,
        orig_skip_first=args.skip_first_n_requests,
        orig_skip_last=args.skip_last_n_requests,
    )

    # ---- Sweep loop ----
    for mc in concurrency_range:
        if use_dynamic_num_prompts:
            assert args.num_prompts_multiplier is not None
            assert mc is not None
            args.num_prompts = args.num_prompts_multiplier * mc
            logger.info(
                f"Using num_prompts = {args.num_prompts_multiplier}"
                f" * {mc} = {args.num_prompts}"
            )

        for rr in request_rate_range:
            # Temporarily write the per-iteration values so that downstream
            # code reading args.max_concurrency / args.request_rate sees the
            # correct scalar value.
            args.max_concurrency = str(mc) if mc is not None else None
            args.request_rate = str(rr)

            iteration_results: list[
                TextGenerationBenchmarkResult | PixelGenerationBenchmarkResult
            ] = []
            for _iteration in range(args.num_iters):
                if args.flush_prefix_cache:
                    flush_prefix_cache(
                        args.backend, args.host, args.port, args.dry_run
                    )

                args.seed = int(np.random.randint(0, 10000))

                result = asyncio.run(benchmark(args, session, mc, rr))
                iteration_results.append(result)

            # Median selection when running multiple iterations.
            if len(iteration_results) > 1:
                throughputs = np.asarray(
                    [r.metrics.request_throughput for r in iteration_results]
                )
                idx = argmedian(throughputs)
            else:
                idx = 0
            best_result = iteration_results[idx]

            # JSON result file (for the median iteration).
            save_result_json(
                args.result_filename,
                args,
                best_result,
                benchmark_task=session.benchmark_task,
                model_id=session.model_id,
                tokenizer_id=session.tokenizer_id,
                request_rate=rr,
            )

            # Output lengths recording (for the median iteration).
            save_output_lengths(args, best_result, session.benchmark_task)

            yield BenchmarkRunResult(
                mc, rr, args.num_prompts or 0, result=best_result
            )


def _extract_metadata_args(
    args: list[str],
) -> tuple[list[str], list[str]]:
    """Extract --metadata values from args before passing to cyclopts.

    cyclopts interprets bare ``key=value`` tokens as keyword assignments. When
    a token like ``enable_prefix_caching=True`` matches a real model field, it
    is routed to that field rather than consumed as a ``--metadata`` list item,
    leaving subsequent tokens as orphaned positionals (which then fail).

    This function peels off all space-separated values after ``--metadata``
    (until the next ``--flag``) and returns them separately so cyclopts never
    sees them.

    Returns:
        A 2-tuple of (clean_args, metadata_values).
    """
    clean_args: list[str] = []
    metadata_values: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--metadata":
            i += 1
            while i < len(args) and not args[i].startswith("-"):
                metadata_values.append(args[i])
                i += 1
        else:
            clean_args.append(args[i])
            i += 1
    return clean_args, metadata_values


def parse_args(
    args: Sequence[str] | None = None,
    *,
    app_name: str = "benchmark_serving",
    description: str = BENCHMARK_SERVING_ARGPARSER_DESCRIPTION,
) -> ServingBenchmarkConfig:
    """Parse command line arguments into a ServingBenchmarkConfig.

    Args:
        args: Command line arguments to parse. If None, parse from sys.argv.
        app_name: Name shown in --help output.
        description: Description shown in --help output.
    """
    raw_args = list(sys.argv[1:] if args is None else args)

    clean_args, metadata_values = _extract_metadata_args(raw_args)

    parsed_configs: list[ServingBenchmarkConfig] = []

    app = App(
        name=app_name,
        help=description,
        help_formatter="plain",
        config=[Env(prefix="MODULAR_")],
        result_action="return_value",
    )

    @app.default
    def _capture(
        config: Annotated[
            ServingBenchmarkConfig, Parameter(name="*")
        ] = ServingBenchmarkConfig(),
    ) -> None:
        parsed_configs.append(config)

    app(clean_args)
    if not parsed_configs:
        raise SystemExit(0)
    config = parsed_configs[0]
    if metadata_values:
        config.metadata = metadata_values
    return config
