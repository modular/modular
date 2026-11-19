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
"""Fakes shared by the benchmark unit tests."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import NamedTuple
from unittest.mock import MagicMock

from max.benchmark.benchmark_shared.config import SamplingConfig
from max.benchmark.benchmark_shared.datasets.types import (
    ChatMessage,
    ChatSession,
    OpenAIImage,
    ResponseFormat,
    SampledRequest,
    SessionMessage,
)
from max.benchmark.benchmark_shared.metrics import (
    InteractivityMetrics,
    TextGenAggregates,
)
from max.benchmark.benchmark_shared.request import (
    BaseRequestFuncInput,
    RequestDriver,
    RequestFuncInput,
    RequestFuncOutput,
)
from pytest_mock import MockerFixture


class FakeTokenizer:
    """Picklable stand-in for `PreTrainedTokenizerBase` used by unit tests.

    `fake_loader` constructs one per spawn worker; the parent uses an
    instance of this class directly. Behavior is deterministic so tests
    can assert on outputs (no Mock call-count plumbing).
    """

    name_or_path = "_fake_"
    vocab_size = 1000
    unk_token_id = None
    all_special_ids: frozenset[int] = frozenset({0, 1, 2})

    def __init__(self, model_max_length: int = 4096) -> None:
        self.model_max_length = model_max_length

    def encode(
        self, text: str, add_special_tokens: bool = False, **_: object
    ) -> list[int]:
        # Length-aware so multiturn-fit tests that scale messages to a
        # target token count round-trip through encode→decode→encode and
        # see a meaningful encoded length.
        return list(range(max(4, len(text))))

    def decode(
        self, ids: list[int], skip_special_tokens: bool = False, **_: object
    ) -> str:
        return "Z" * len(ids)

    def convert_tokens_to_ids(self, token: str) -> int:
        return 223


def fake_loader(
    name_or_path: str,
    model_max_length: int | None,
    trust_remote_code: bool,
    revision: str | None,
) -> FakeTokenizer:
    return FakeTokenizer(model_max_length=model_max_length or 4096)


class CapturingDriver(RequestDriver):
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


def make_session(session_id: int, num_user_turns: int = 3) -> ChatSession:
    messages: list[SessionMessage] = []
    for i in range(num_user_turns):
        messages.append(
            SessionMessage(source="user", content=f"turn {i}", num_tokens=10)
        )
        messages.append(
            SessionMessage(source="assistant", content="", num_tokens=5)
        )
    return ChatSession(id=session_id, messages=messages)


def write_blob(path: Path, **fields: object) -> None:
    """Writes ``fields`` as a JSON object, standing in for a benchmark result."""
    path.write_text(json.dumps(fields))


def make_text_aggregates(**overrides: object) -> TextGenAggregates:
    """Returns the aggregates of a 100-request, 60 s run, with ``overrides``."""
    fields: dict[str, object] = {
        "duration": 60.0,
        "completed": 100,
        "failures": 0,
        "request_throughput": 1.66,
        "total_input": 5000,
        "total_output": 8000,
        "nonempty_response_chunks": 8000,
        "max_input": 100,
        "max_output": 200,
        "max_total": 300,
        "global_cached_token_rate": 0.42,
    }
    fields.update(overrides)
    return TextGenAggregates.model_validate(fields)


class SampleInteractivity(NamedTuple):
    """One distinct `InteractivityMetrics` per interactivity basis."""

    content_bounded: InteractivityMetrics
    full_response: InteractivityMetrics
    e2e_normalized: InteractivityMetrics


def make_sample_interactivity() -> SampleInteractivity:
    """Returns interactivity metrics whose values differ in every cell."""
    return SampleInteractivity(
        content_bounded=InteractivityMetrics(
            mean=120.0,
            p50=120.0,
            p75=115.0,
            p90=110.0,
            p95=105.0,
            p99=100.0,
        ),
        full_response=InteractivityMetrics(
            mean=110.0,
            p50=110.0,
            p75=106.0,
            p90=102.0,
            p95=98.0,
            p99=94.0,
        ),
        e2e_normalized=InteractivityMetrics(
            mean=100.0,
            p50=100.0,
            p75=96.0,
            p90=92.0,
            p95=88.0,
            p99=84.0,
        ),
    )


def make_sampled_request(
    prompt: str | list[ChatMessage] = "hi",
    *,
    prompt_len: int = 1,
    output_len: int | None = 64,
    ignore_eos: bool = True,
) -> SampledRequest:
    """Returns an image-free request that forces 64 output tokens by default."""
    return SampledRequest(
        prompt_formatted=prompt,
        prompt_len=prompt_len,
        output_len=output_len,
        encoded_images=[],
        ignore_eos=ignore_eos,
    )


def make_request_input(
    api_url: str,
    *,
    model: str = "test-model",
    prompt: str | list[ChatMessage] = "Test prompt",
    prompt_len: int = 10,
    max_tokens: int = 100,
    images: list[OpenAIImage] | None = None,
    ignore_eos: bool = False,
    response_format: ResponseFormat | None = None,
    tools: list[dict[str, object]] | None = None,
) -> RequestFuncInput:
    """Returns a text request with default sampling."""
    return RequestFuncInput(
        model=model,
        session_id=None,
        sampling=SamplingConfig(),
        prompt=prompt,
        images=[] if images is None else images,
        api_url=api_url,
        prompt_len=prompt_len,
        max_tokens=max_tokens,
        ignore_eos=ignore_eos,
        response_format=response_format,
        tools=tools,
    )


def mock_chat_stream(
    mocker: MockerFixture, chunks: Sequence[bytes]
) -> MagicMock:
    """Patches aiohttp so a chat request streams ``chunks`` back.

    Returns the mocked client, whose ``post`` records the sent payload.
    """
    mocker.patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    session_class = mocker.patch(
        "max.benchmark.benchmark_shared.request.aiohttp.ClientSession"
    )
    client = session_class.return_value.__aenter__.return_value

    async def body() -> AsyncIterator[bytes]:
        for chunk in chunks:
            yield chunk
        yield b"data: [DONE]\n\n"

    response = mocker.AsyncMock()
    response.status = 200
    response.content = body()
    post_ctx = mocker.AsyncMock()
    post_ctx.__aenter__ = mocker.AsyncMock(return_value=response)
    post_ctx.__aexit__ = mocker.AsyncMock(return_value=None)
    client.post = mocker.Mock(return_value=post_ctx)
    return client
