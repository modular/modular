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
"""Request types and interface for cascade text-generation pipelines.

Kept in ``interfaces`` (below both ``workers`` and the concrete ``pipelines``)
so workers can consume the request types and ``serve`` can route on the
interface without either depending on concrete pipeline implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, AsyncIterator
from typing import TypeAlias

from max.experimental.cascade.core import pipeline_method
from pydantic import BaseModel


class GenerateRequest(BaseModel):
    """Generation parameters for a single text-generation request.

    Mirrors the request-configurable sampling fields that MAX Serve accepts on
    its OpenAI text-generation routes, so the chat-completion adapter can
    forward everything a client sends straight through to the model worker.
    Every field except ``num_tokens``, ``min_new_tokens`` and ``ignore_eos``
    defaults to ``None`` meaning "use the model / server default", matching how
    ``SamplingParamsInput`` is layered on top of a model's ``GenerationConfig``.
    """

    # Generation-length control.
    num_tokens: int = 10
    """Maximum number of new tokens to generate (maps to ``max_new_tokens``)."""

    min_new_tokens: int = 0
    """Minimum number of new tokens to generate before stopping."""

    ignore_eos: bool = False
    """If ``True``, keep generating past the EOS token until the token budget
    (or a stop condition) is hit."""

    # Core sampling controls.
    temperature: float = 1.0
    """Sampling temperature. Set to ``0.0`` for greedy (deterministic) decoding."""

    top_k: int | None = None
    """Keep only the ``top_k`` most probable tokens when sampling."""

    top_p: float | None = None
    """Nucleus-sampling cumulative-probability threshold."""

    min_p: float | None = None
    """Minimum token probability relative to the most likely token."""

    thinking_temperature: float | None = None
    """Temperature override for tokens inside a ``<think>...</think>`` block."""

    seed: int | None = None
    """Random seed for reproducible sampling. ``None`` picks a random seed."""

    # Penalties.
    frequency_penalty: float | None = None
    """Penalty applied proportionally to a token's frequency so far."""

    presence_penalty: float | None = None
    """Flat penalty applied to tokens that have already appeared."""

    repetition_penalty: float | None = None
    """Factor by which logits of repeated tokens are divided."""

    # Stopping criteria.
    stop: list[str] | None = None
    """Strings that terminate generation when produced."""

    stop_token_ids: list[int] | None = None
    """Token ids that terminate generation when produced."""


ChatMessages: TypeAlias = list[dict[str, str]]


class TextGenInterface(ABC):
    """Standard interface for a text-generation model.

    The composable primitive is :meth:`open_text_stream`, which returns the
    detokenized text as a streaming worker handle. A composing pipeline (for
    example the serving-layer OpenAI chat wrapper) can forward that handle to
    another worker so post-processing runs worker-to-worker without routing
    every token through the orchestrator. :meth:`generate_text` is a thin
    default that just materializes the handle for direct consumption.
    """

    @abstractmethod
    async def open_text_stream(
        self,
        req: GenerateRequest,
        prompt: str | ChatMessages,
    ) -> AsyncIterable[str]:
        """Open a streaming text-generation handle for a text or chat prompt.

        Returns a worker stream handle (not a materialized iterator) so a
        composing pipeline can hand it to another worker. Must be called within
        an active pipeline scope (for example from a ``@pipeline_method``); the
        returned handle stays valid for that scope's lifetime.
        """
        ...

    @pipeline_method
    async def generate_text(
        self,
        req: GenerateRequest,
        prompt: str | ChatMessages,
    ) -> AsyncIterator[str]:
        """Generate a streaming text response to a text or chat prompt."""
        async for text in await self.open_text_stream(req, prompt):
            yield text
