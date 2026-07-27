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
"""OpenAI chat-completion SSE framing.

Encoding a text delta into an OpenAI ``chat.completion.chunk`` SSE frame is the
per-token hot path: it runs a pydantic ``model_dump_json`` for every streamed
chunk, which is why :class:`OpenAIChatFormatter` runs it on the CPU worker pool
rather than the API event loop. Hand-rolling the JSON instead would be cheaper
still, but the schema surface only grows from here (tool calls, usage,
logprobs), and offloading already takes the cost off the loop.
"""

from __future__ import annotations

from max.serve.schemas.openai import (
    ChatCompletionStreamResponseChoice,
    ChatCompletionStreamResponseDelta,
    CreateChatCompletionStreamResponse,
)

# Terminal frame of an OpenAI streaming response. Pre-encoded because it never
# varies per request.
DONE_SSE: bytes = b"data: [DONE]\n\n"


def format_chat_chunk_sse(
    text_chunk: str,
    model: str,
    request_id: str,
    created: int,
) -> bytes:
    r"""Encode a text delta as an OpenAI ``chat.completion.chunk`` SSE frame.

    Args:
        text_chunk: Incremental generated text for this delta.
        model: Model name echoed back to the client.
        request_id: Stable ``chatcmpl-`` id shared across the response's frames.
        created: Unix timestamp shared across the response's frames.

    Returns:
        The UTF-8 ``data: {json}\n\n`` server-sent-event frame bytes.
    """
    chunk = CreateChatCompletionStreamResponse(
        id=request_id,
        created=created,
        model=model,
        object="chat.completion.chunk",
        choices=[
            ChatCompletionStreamResponseChoice(
                index=0,
                delta=ChatCompletionStreamResponseDelta(content=text_chunk),
                finish_reason=None,
            )
        ],
    )
    return b"data: " + chunk.model_dump_json().encode("utf-8") + b"\n\n"
