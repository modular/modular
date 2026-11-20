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
from collections.abc import AsyncIterator, Iterable, Sequence
from pathlib import Path
from typing import Any, NamedTuple
from unittest.mock import MagicMock, Mock

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


# One JSON request body as a test sees it: an object whose values are
# arbitrarily nested, which is what ``json.loads`` hands back.
SentBody = dict[str, Any]


def sent_json(post: Mock) -> SentBody:
    """The JSON body a mocked aiohttp ``post`` carried."""
    return sent_bodies(post)[-1]


def sent_bodies(post: Mock) -> list[SentBody]:
    """The JSON bodies a mocked aiohttp ``post`` carried, in call order."""
    return [json.loads(call.kwargs["data"]) for call in post.call_args_list]


class StreamBody:
    """Stands in for aiohttp's body reader, consuming as it is read."""

    def __init__(self, chunks: Iterable[bytes]) -> None:
        self.chunks = list(chunks)

    def iter_any(self) -> AsyncIterator[bytes]:
        """Yields the chunks given, so boundaries stay as sent."""
        return _BodyChunks(self)

    def iter_chunked(self, nbytes: int) -> AsyncIterator[bytes]:
        """Yields the rest the same way; a test chunk is already small."""
        return _BodyChunks(self)


class _BodyChunks:
    """Drains a ``StreamBody``, like aiohttp's own stream iterator.

    An iterator rather than a generator, so a caller that stops reading early
    leaves nothing for the garbage collector to report as an unclosed
    generator.
    """

    def __init__(self, body: StreamBody) -> None:
        self._body = body

    def __aiter__(self) -> _BodyChunks:
        return self

    async def __anext__(self) -> bytes:
        if not self._body.chunks:
            raise StopAsyncIteration
        return self._body.chunks.pop(0)


def write_engine_result_set(path: Path) -> None:
    """Writes a minimal engine ResultSetInContext document."""
    path.write_text(
        json.dumps(
            {
                "run_context": {"config_id": None, "git_commit": "abc"},
                "results": [
                    {
                        "iteration_config": {
                            "batch_size": 4,
                            "context_len": 0,
                            "input_len": 128,
                            "output_len": 64,
                        },
                        "result": {
                            "total_latency_ms": {
                                "mean": 10.0,
                                "std": 1.0,
                                "p50": 9.0,
                                "p90": 12.0,
                                "p95": 13.0,
                                "p99": 15.0,
                                "unit": "ms",
                                "confidence_info": None,
                            },
                            "prefill_latency_ms": {"mean": 3.0, "p90": 3.5},
                            "decode_latency_ms": {"mean": 7.0, "p90": 8.0},
                            "tokens_per_second": 500.0,
                            "input_tokens_per_second": 300.0,
                            "output_tokens_per_second": 200.0,
                            "gpu": {
                                "peak_memory_mib": 40000.0,
                                "avg_utilization_percent": 90.0,
                                "kernel_time_ms": 5.0,
                                "device_count": 1,
                                "per_gpu_metrics": None,
                            },
                        },
                    }
                ],
            }
        )
    )


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
    prompt_carries_image: bool = False,
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
        prompt_carries_image=prompt_carries_image,
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

    response = mocker.AsyncMock()
    response.status = 200
    response.content = StreamBody([*chunks, b"data: [DONE]\n\n"])
    post_ctx = mocker.AsyncMock()
    post_ctx.__aenter__ = mocker.AsyncMock(return_value=response)
    post_ctx.__aexit__ = mocker.AsyncMock(return_value=None)
    client.post = mocker.Mock(return_value=post_ctx)
    return client
