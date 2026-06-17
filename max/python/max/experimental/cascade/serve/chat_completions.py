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
"""Chat-completion route adapter for cascade text generation."""

from __future__ import annotations

import time
from collections.abc import AsyncIterable, AsyncIterator, Mapping, Sequence
from typing import Any

from fastapi import APIRouter, HTTPException
from max.experimental.cascade.pipelines.textgen import (
    ChatMessages,
    GenerateRequest,
    TextGenInterface,
)
from max.serve.schemas.openai import (
    ChatCompletionResponseChoice,
    ChatCompletionResponseMessage,
    ChatCompletionStreamResponseChoice,
    ChatCompletionStreamResponseDelta,
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
)
from sse_starlette.sse import EventSourceResponse


def _normalize_message_content(
    content: str | Sequence[Mapping[str, Any]],
) -> str:
    """Flatten a chat message content payload into plain text."""
    if isinstance(content, str):
        return content

    text_parts: list[str] = []
    unsupported_types: list[str] = []
    for part in content:
        if part.get("type") == "text" and part.get("text") is not None:
            text_parts.append(part["text"])
        else:
            unsupported_types.append(part.get("type", "unknown"))

    if unsupported_types:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported chat message content part types: "
                + ", ".join(sorted(set(unsupported_types)))
            ),
        )

    return "".join(text_parts)


async def _stream_response(
    response: AsyncIterable[str],
    model: str,
) -> AsyncIterator[str]:
    async for text_chunk in response:
        chunk = CreateChatCompletionStreamResponse(
            id="chatcmpl-cascade",
            created=int(time.time()),
            model=model,
            object="chat.completion.chunk",
            choices=[
                ChatCompletionStreamResponseChoice(
                    index=0,
                    delta=ChatCompletionStreamResponseDelta(
                        content=text_chunk,
                    ),
                    finish_reason=None,
                )
            ],
        )
        yield chunk.model_dump_json()
    yield "[DONE]"


def build_router(
    pipeline: TextGenInterface,
) -> APIRouter:
    """Build OpenAI-style chat-completion routes for a pipeline."""
    router = APIRouter()

    @router.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        request: CreateChatCompletionRequest,
    ) -> CreateChatCompletionResponse | EventSourceResponse:
        messages: ChatMessages = [
            {
                "role": message.get("role", ""),
                "content": _normalize_message_content(
                    message.get("content") or ""
                ),
            }
            for message in request.messages
        ]
        req = GenerateRequest(
            ignore_eos=request.ignore_eos,
        )
        if request.max_tokens is not None:
            req.num_tokens = request.max_tokens
        response = pipeline.generate_text(req, messages)

        if request.stream:
            return EventSourceResponse(
                _stream_response(response, model=request.model),
            )

        chunks = [chunk async for chunk in response]
        return CreateChatCompletionResponse(
            id="chatcmpl-cascade",
            created=int(time.time()),
            model=request.model,
            object="chat.completion",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatCompletionResponseMessage(
                        role="assistant",
                        content="".join(chunks),
                    ),
                    finish_reason="stop",
                )
            ],
        )

    return router
