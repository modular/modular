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
"""A streaming ``openai-api`` model provider for Inspect.

Inspect's OpenAI-compatible provider never streams. OpenRouter's benchmark
harness always does, so a provider sees streamed requests from it. Streaming
also keeps bytes flowing during a long answer: a non-streaming request is silent
until it finishes, so a proxy idle timeout on the path would cut a long
reasoning chain or a repetition loop and Inspect would re-sample it instead of
scoring it.

``openai-stream/<service>/<model>`` behaves like ``openai-api/<service>/<model>``
(same ``<SERVICE>_BASE_URL`` / ``<SERVICE>_API_KEY`` lookup), except that it
streams the completion and assembles the chunks into the ``ChatCompletion``
Inspect already parses.

Runs only inside the harness venv; importing this module registers the provider.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
from inspect_ai.model import GenerateConfig, modelapi
from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI
from openai.types.chat import ChatCompletion


class IncompleteStreamError(Exception):
    """The stream closed before the server sent a finish_reason."""


class OpenAIStreamingAPI(OpenAICompatibleAPI):
    async def _generate_completion(
        self, request: dict[str, Any], config: GenerateConfig
    ) -> ChatCompletion:
        stream = await self.client.chat.completions.create(
            **request, stream=True, stream_options={"include_usage": True}
        )
        completion_id = ""
        model = self.service_model_name()
        created = int(time.time())
        content: list[str] = []
        reasoning: list[str] = []
        finish_reason: str | None = None
        usage: dict[str, Any] | None = None
        async for chunk in stream:
            completion_id = chunk.id or completion_id
            model = chunk.model or model
            created = chunk.created or created
            if chunk.usage is not None:
                usage = chunk.usage.model_dump()
            for choice in chunk.choices:
                delta = choice.delta.model_dump()
                if delta.get("content"):
                    content.append(delta["content"])
                # MAX streams "reasoning"; other servers use "reasoning_content".
                piece = delta.get("reasoning") or delta.get("reasoning_content")
                if piece:
                    reasoning.append(piece)
                if delta.get("tool_calls"):
                    raise NotImplementedError(
                        "openai-stream does not assemble streamed tool calls"
                    )
                finish_reason = choice.finish_reason or finish_reason

        # A stream cut mid-generation looks like a short, finished answer
        # unless it is rejected here.
        if finish_reason is None:
            raise IncompleteStreamError(
                f"stream {completion_id or '<no id>'} closed after "
                f"{len(content)} content and {len(reasoning)} reasoning chunks "
                "without a finish_reason"
            )

        message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(content),
        }
        if reasoning:
            message["reasoning_content"] = "".join(reasoning)
        return ChatCompletion.model_validate(
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": usage,
            }
        )

    def should_retry(self, ex: BaseException) -> bool:
        # A connection dropped mid-stream surfaces as a raw httpx error rather
        # than the openai SDK's APIConnectionError.
        if isinstance(ex, (IncompleteStreamError, httpx.TransportError)):
            return True
        return super().should_retry(ex)


@modelapi(name="openai-stream")
def openai_stream() -> type[OpenAIStreamingAPI]:
    return OpenAIStreamingAPI
