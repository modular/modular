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
"""The responses route's handler, and the media seam it shares with chat."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from max.pipelines.request import OpenResponsesRequest, RequestID
from max.pipelines.request.open_responses import OpenResponsesRequestBody
from max.serve.pipelines.general_handler import GeneralPipelineHandler

pytestmark = pytest.mark.asyncio

_PNG_DATA_URI = "data:image/png;base64,aGk="


def _request(content: Any) -> OpenResponsesRequest:
    return OpenResponsesRequest(
        request_id=RequestID(value="test-request"),
        body=OpenResponsesRequestBody.model_validate(
            {
                "model": "test",
                "input": [{"role": "user", "content": content}],
            }
        ),
    )


async def _recorded_carried_media(request: OpenResponsesRequest) -> bool:
    """The ``carried_media`` flag the handler passes the recorder."""

    async def mock_stream(
        streamed_id: str, context: Any
    ) -> AsyncGenerator[tuple[list[Any], int | None], None]:
        async def _gen() -> AsyncGenerator[tuple[list[Any], int | None], None]:
            yield ([Mock()], None)

        return _gen()

    handler = Mock()
    handler.tokenizer.new_context = AsyncMock(return_value=Mock())
    handler.model_worker.stream = mock_stream
    handler.debug_logging = False

    # Same invariant as the chat path: the snapshot is read *after*
    # tokenization, which is what moved the cache's counters.
    order = Mock()
    order.attach_mock(handler.tokenizer.new_context, "new_context")
    order.attach_mock(handler._preprocess_cache_stats.record, "record")

    bound = GeneralPipelineHandler.next.__get__(handler, GeneralPipelineHandler)
    _ = [output async for output in bound(request)]

    assert [call[0] for call in order.mock_calls] == ["new_context", "record"]
    handler._preprocess_cache_stats.record.assert_called_once()
    return handler._preprocess_cache_stats.record.call_args.kwargs[
        "carried_media"
    ]


async def test_an_image_request_publishes_the_preprocess_cache_counters() -> (
    None
):
    """The responses route shares the tokenizer, so it moves the same cache.

    Its lookups were counted by nobody: the recorder hung off
    ``TokenGeneratorPipeline`` alone, so a responses-only deployment saw the
    family permanently absent and a mixed one folded this traffic's deltas
    into whichever chat request came next.
    """
    request = _request(
        [
            {"type": "input_text", "text": "what is this"},
            {"type": "input_image", "image_url": _PNG_DATA_URI},
        ]
    )

    assert await _recorded_carried_media(request) is True


async def test_a_text_only_request_stays_silent() -> None:
    """``new_context`` runs for every request, so the gate is explicit."""
    assert (
        await _recorded_carried_media(
            _request([{"type": "input_text", "text": "hello"}])
        )
        is False
    )


async def test_a_plain_string_input_stays_silent() -> None:
    """``input`` is a string or a message list; the string arm has no media."""
    request = OpenResponsesRequest(
        request_id=RequestID(value="test-request"),
        body=OpenResponsesRequestBody.model_validate(
            {"model": "test", "input": "hello"}
        ),
    )

    assert await _recorded_carried_media(request) is False
