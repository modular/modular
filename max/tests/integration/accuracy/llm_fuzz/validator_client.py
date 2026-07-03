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
"""OpenAI SDK-based client for correctness validation scenarios.

Wraps the openai Python SDK to provide high-level helpers for testing
tool calling, structured output, and streaming protocol compliance.
This is the complement to FuzzClient — FuzzClient sends broken requests
to crash servers, ValidatorClient sends valid requests to check correctness.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from helpers import collect_stream

if TYPE_CHECKING:
    from client import RunConfig

try:
    import openai

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


@dataclass
class ValidatorConfig:
    """Configuration for the validator client, derived from RunConfig."""

    base_url: str
    api_key: str = "EMPTY"
    model: str = "default"
    timeout: float = 600.0
    max_workers: int = 16


class ValidatorClient:
    """OpenAI SDK wrapper for correctness validation scenarios."""

    def __init__(self, config: ValidatorConfig):
        if not HAS_OPENAI:
            raise RuntimeError(
                "openai package required for validation scenarios. "
                "Install with: pip install openai"
            )
        self.config = config
        self._client = openai.OpenAI(
            api_key=config.api_key or "EMPTY",
            base_url=config.base_url.rstrip("/") + "/v1",
            timeout=config.timeout,
        )

    @property
    def openai(self) -> openai.OpenAI:
        """Direct access to the underlying OpenAI client."""
        return self._client

    @property
    def model(self) -> str:
        return self.config.model

    def close(self) -> None:
        self._client.close()

    def detect_model(self) -> str | None:
        """Fetch the first available model from the server."""
        try:
            models = self._client.models.list()
            if models.data:
                return models.data[0].id
        except Exception:
            pass
        return None

    def is_available(self) -> bool:
        """Check if the server is reachable."""
        try:
            self._client.models.list()
            return True
        except Exception:
            return False

    def chat(
        self,
        messages: list[Any],
        *,
        model: str | None = None,
        max_tokens: int = 2048,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Send a chat completion request."""
        params: dict[str, Any] = {
            "model": model or self.config.model,
            "messages": messages,
            "max_tokens": max_tokens,
            **kwargs,
        }
        if stream:
            return self._client.chat.completions.create(**params, stream=True)
        return self._client.chat.completions.create(**params)

    def chat_stream(
        self,
        messages: list[Any],
        *,
        model: str | None = None,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send a streaming chat request and collect all chunks into a dict."""
        stream = self.chat(
            messages, model=model, max_tokens=max_tokens, stream=True, **kwargs
        )
        return collect_stream(stream)

    def tc_chat(
        self,
        messages: list[Any],
        tools: list[dict[str, Any]],
        *,
        model: str | None = None,
        max_tokens: int = 2048,
        tool_choice: str | dict[str, Any] = "auto",
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Send a tool-calling chat request."""
        return self.chat(
            messages,
            model=model,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            stream=stream,
            **kwargs,
        )

    def tc_chat_stream(
        self,
        messages: list[Any],
        tools: list[dict[str, Any]],
        *,
        model: str | None = None,
        max_tokens: int = 2048,
        tool_choice: str | dict[str, Any] = "auto",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send a streaming tool-calling request and collect chunks."""
        stream = self.tc_chat(
            messages,
            tools,
            model=model,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
            stream=True,
            **kwargs,
        )
        return collect_stream(stream)

    def so_chat(
        self,
        messages: list[Any],
        schema: dict[str, Any],
        *,
        schema_name: str = "response",
        model: str | None = None,
        max_tokens: int = 2048,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Send a structured output (json_schema) chat request."""
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": schema,
            },
        }
        return self.chat(
            messages,
            model=model,
            max_tokens=max_tokens,
            response_format=response_format,
            stream=stream,
            **kwargs,
        )

    def so_chat_stream(
        self,
        messages: list[Any],
        schema: dict[str, Any],
        *,
        schema_name: str = "response",
        model: str | None = None,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send a streaming structured output request and collect chunks."""
        stream = self.so_chat(
            messages,
            schema,
            schema_name=schema_name,
            model=model,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )
        return collect_stream(stream)

    def concurrent_run(
        self,
        fn: Callable[..., Any],
        args_list: list[Any],
        *,
        max_workers: int | None = None,
    ) -> list[tuple[int, Any, str | None]]:
        """Run fn(args) concurrently, returning (index, result, error) tuples."""
        workers = max_workers or self.config.max_workers
        results: list[tuple[int, Any, str | None]] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(fn, *args): i for i, args in enumerate(args_list)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    results.append((idx, result, None))
                except Exception as e:
                    results.append((idx, None, str(e)))
        results.sort(key=lambda x: x[0])
        return results


def make_validator_config(run_config: RunConfig) -> ValidatorConfig:
    """Create a ValidatorConfig from a RunConfig."""
    return ValidatorConfig(
        base_url=run_config.base_url,
        api_key=run_config.api_key,
        model=run_config.model,
        timeout=max(run_config.timeout * 20, 600.0),
        max_workers=min(run_config.max_concurrency, 16),
    )
