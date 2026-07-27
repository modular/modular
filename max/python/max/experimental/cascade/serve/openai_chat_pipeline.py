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
"""Serving-layer pipeline that adds OpenAI chat streaming to any text generator.

Wraps a :class:`TextGenInterface` and pairs it with an
:class:`OpenAIChatFormatter` worker so the OpenAI ``chat.completion.chunk``
serialization runs on the CPU worker pool rather than the API event loop. The
wrapped generator's text stream is forwarded to the formatter worker-to-worker
(via its :meth:`~TextGenInterface.open_text_stream` handle), so the per-token
JSON encoding never round-trips the orchestrator. Keeping this in ``serve``
leaves the generic text-gen interface and pipelines free of any OpenAI-wire or
serving concerns.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator

from max.experimental.cascade.core import pipeline_method
from max.experimental.cascade.interfaces.pipeline import CascadePipeline
from max.experimental.cascade.interfaces.textgen import (
    ChatMessages,
    GenerateRequest,
    TextGenInterface,
)
from max.experimental.cascade.serve.openai_chat_formatter import (
    OpenAIChatFormatter,
)


class OpenAIChatCompletionPipeline(CascadePipeline, TextGenInterface):
    """Wrap a text generator with worker-offloaded OpenAI chat SSE formatting."""

    def __init__(self, inner: TextGenInterface) -> None:
        """Wrap *inner*, adding a formatter worker for OpenAI chat streaming.

        Args:
            inner: An already-deployed text-generation pipeline to expose over
                OpenAI chat. :meth:`deploy` only brings up this wrapper's own
                formatter worker; ``inner`` is deployed by whoever owns it.
        """
        self.inner = inner
        self.formatter = OpenAIChatFormatter()

    async def open_text_stream(
        self,
        req: GenerateRequest,
        prompt: str | ChatMessages,
    ) -> AsyncIterable[str]:
        """Forward the wrapped generator's streaming text handle."""
        return await self.inner.open_text_stream(req, prompt)

    @pipeline_method
    async def stream_chat_sse(
        self,
        req: GenerateRequest,
        prompt: str | ChatMessages,
        model: str,
        request_id: str,
        created: int,
    ) -> AsyncIterator[bytes]:
        """Stream OpenAI chat SSE frames, formatting on the worker pool.

        The wrapped generator's text handle flows straight into the formatter
        worker, so both run worker-to-worker inside this one pipeline scope and
        the orchestrator only forwards the finished byte frames.
        """
        text_stream = await self.inner.open_text_stream(req, prompt)
        async for frame in await self.formatter.format_stream(
            text_stream, model, request_id, created
        ):
            yield frame
