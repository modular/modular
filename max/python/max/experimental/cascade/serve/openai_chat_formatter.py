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
"""Cascade worker that serializes a text stream into OpenAI chat SSE bytes."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator

from max.experimental.cascade.core import Worker, worker_method
from max.experimental.cascade.serve.openai_sse import (
    DONE_SSE,
    format_chat_chunk_sse,
)


class OpenAIChatFormatter(Worker):
    """Encode a detokenized text stream into OpenAI chat-completion SSE frames.

    OpenAI streaming responses run a pydantic ``model_dump_json`` per token,
    which otherwise executes on the single GIL-bound API event loop. Running it
    in a CPU worker moves that per-chunk cost onto the round-robin worker pool,
    so it parallelizes across concurrent requests and stops pacing the decode
    stream. Chaining it after the detokenizer keeps the token deltas flowing
    worker-to-worker; the API process only forwards finished byte frames.
    """

    def __init__(self) -> None:
        super().__init__(deploy_hints=["cpu"])

    @worker_method()
    async def format_stream(
        self,
        text_iter: AsyncIterable[str],
        model: str,
        request_id: str,
        created: int,
    ) -> AsyncIterator[bytes]:
        """Forward a text-delta stream as OpenAI SSE frame bytes."""
        async for text in text_iter:
            yield format_chat_chunk_sse(text, model, request_id, created)
        yield DONE_SSE
