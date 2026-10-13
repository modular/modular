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
"""An error before a stream's 200 is committed is an HTTP status; after, it is
a frame inside the stream."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Iterator
from unittest.mock import Mock, patch

import pytest
from fastapi.responses import JSONResponse
from max.pipelines.context import GenerationStatus
from max.pipelines.context.exceptions import InputError
from max.pipelines.modeling.types import RequestID
from max.serve.pipelines.llm import TokenGeneratorOutput
from max.serve.router.openai_routes import (
    OpenAICompletionResponseGenerator,
    _start_stream,
)


@pytest.fixture(autouse=True)
def patch_openai_metrics() -> Iterator[None]:
    with (
        patch("max.serve.router.openai_routes.record_request_start"),
        patch("max.serve.router.openai_routes.record_request_end"),
    ):
        yield


def _request() -> Mock:
    request = Mock()
    request.request_id = RequestID("test")
    request.model_name = "test-model"
    request.timestamp_ns = 1
    return request


def _chunk(text: str) -> TokenGeneratorOutput:
    return TokenGeneratorOutput(
        status=GenerationStatus.ACTIVE, decoded_tokens=text, token_count=1
    )


async def _collect(
    stream: AsyncGenerator[str | JSONResponse, None],
) -> list[str | JSONResponse]:
    return [chunk async for chunk in stream]


# Before the 200, only these two types become an HTTP status here; after it,
# every failure is reported inside the stream.
HTTP_ERRORS: list[tuple[type[Exception], int]] = [
    (InputError, 400),
    (ValueError, 500),
]
STREAM_ERRORS: list[tuple[type[Exception], int, str]] = [
    (InputError, 400, "boom"),
    (ValueError, 500, "boom"),
    (RuntimeError, 500, "Internal server error."),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected_status", "expected_message"),
    STREAM_ERRORS,
    ids=["input-error", "value-error", "other"],
)
async def test_failure_after_a_payload_stays_inside_the_stream(
    error: type[Exception], expected_status: int, expected_message: str
) -> None:
    async def tokens() -> AsyncGenerator[TokenGeneratorOutput, None]:
        yield _chunk("hello")
        raise error("boom")

    generator = OpenAICompletionResponseGenerator(Mock(model_name="m"))
    error_response, stream = await _start_stream(
        generator._stream(_request(), tokens())
    )
    frames = await _collect(stream)

    assert error_response is None, "the 200 is committed once a chunk is out"
    last = frames[-1]
    assert isinstance(last, str), (
        f"a response object here reaches the client as its repr, got {last!r}"
    )
    reported = json.loads(last)["error"]
    assert reported["message"] == expected_message
    assert reported["code"] == str(expected_status)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected_status"),
    HTTP_ERRORS,
    ids=["input-error", "value-error"],
)
async def test_failure_before_any_payload_is_an_http_status(
    error: type[Exception], expected_status: int
) -> None:
    async def tokens() -> AsyncGenerator[TokenGeneratorOutput, None]:
        raise error("boom")
        yield  # unreachable; makes this an async generator

    generator = OpenAICompletionResponseGenerator(Mock(model_name="m"))
    error_response, _ = await _start_stream(
        generator._stream(_request(), tokens())
    )

    assert isinstance(error_response, JSONResponse)
    assert error_response.status_code == expected_status


@pytest.mark.asyncio
async def test_other_failure_before_any_payload_propagates() -> None:
    # The request middleware answers it with a generic 500, so the exception's
    # own text never reaches the client.
    async def tokens() -> AsyncGenerator[TokenGeneratorOutput, None]:
        raise RuntimeError("internal detail")
        yield  # unreachable; makes this an async generator

    generator = OpenAICompletionResponseGenerator(Mock(model_name="m"))
    with pytest.raises(RuntimeError, match="internal detail"):
        await _start_stream(generator._stream(_request(), tokens()))
