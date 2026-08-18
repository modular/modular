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
from collections.abc import Mapping, Sequence
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from max.experimental.cascade.core import Runtime
from max.experimental.cascade.interfaces.textgen import (
    ChatMessages,
    GenerateRequest,
    TextGenInterface,
)
from max.experimental.cascade.serve.openai_chat_pipeline import (
    OpenAIChatCompletionPipeline,
)
from max.serve.schemas.openai import (
    ChatCompletionResponseChoice,
    ChatCompletionResponseMessage,
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
)


def _convert_stop(stop: str | Sequence[str] | None) -> list[str] | None:
    """Normalize the OpenAI ``stop`` field into a list of strings."""
    if stop is None:
        return None
    if isinstance(stop, str):
        return [stop]
    return list(stop)


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


async def build_router(
    pipeline: TextGenInterface,
    runtime: Runtime,
) -> APIRouter:
    """Build OpenAI-style chat-completion routes for a text generator.

    The routes pair ``pipeline`` with an :class:`OpenAIChatCompletionPipeline`
    so the per-token SSE serialization runs on the CPU worker pool instead of
    the API event loop. That wrapper is an implementation detail of this
    adapter, so callers hand over a plain :class:`TextGenInterface`.

    ``runtime`` is the one ``pipeline`` is already deployed on; only the
    wrapper's own formatter worker is deployed here.
    """
    chat = OpenAIChatCompletionPipeline(pipeline)
    await chat.deploy(runtime)

    router = APIRouter()

    @router.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        request: CreateChatCompletionRequest,
    ) -> CreateChatCompletionResponse | StreamingResponse:
        messages: ChatMessages = [
            {
                "role": message.get("role", ""),
                "content": _normalize_message_content(
                    message.get("content") or ""
                ),
            }
            for message in request.messages
        ]
        # Forward every request-configurable text-gen field OpenAI exposes.
        # The passthrough fields share ``GenerateRequest``'s ``None`` defaults,
        # so forwarding them verbatim leaves unset ones on the model default.
        req = GenerateRequest(
            ignore_eos=request.ignore_eos,
            top_k=request.top_k,
            top_p=request.top_p,
            min_p=request.min_p,
            thinking_temperature=request.thinking_temperature,
            seed=request.seed,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            repetition_penalty=request.repetition_penalty,
            stop=_convert_stop(request.stop),
            stop_token_ids=request.stop_token_ids,
        )
        # Fields whose ``GenerateRequest`` default differs from "unset" are only
        # overridden when the client supplies a value. ``max_completion_tokens``
        # supersedes the legacy ``max_tokens``.
        max_new_tokens = (
            request.max_completion_tokens
            if request.max_completion_tokens is not None
            else request.max_tokens
        )
        if max_new_tokens is not None:
            req.num_tokens = max_new_tokens
        if request.min_tokens is not None:
            req.min_new_tokens = request.min_tokens
        if request.temperature is not None:
            req.temperature = request.temperature
        if request.stream:
            # The wrapper emits fully-framed OpenAI SSE bytes (formatting is
            # offloaded to a worker), so the route just forwards them.
            return StreamingResponse(
                chat.stream_chat_sse(
                    req,
                    messages,
                    request.model,
                    "chatcmpl-cascade",
                    int(time.time()),
                ),
                media_type="text/event-stream",
            )

        chunks = [chunk async for chunk in chat.generate_text(req, messages)]
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
