# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import AsyncGenerator, Generic, Optional, TypeVar

from transformers import AutoTokenizer

from max.serve.scheduler.queues import BatchMultiplexQueue

Context = TypeVar("Context")


@dataclass(frozen=True)
class TokenGeneratorRequest:
    id: str
    prompt: str
    max_new_tokens: Optional[int] = None

    def __str__(self):
        txt = f"Id: {self.id}, Prompt: [{self.prompt[:40]}]"
        if self.max_new_tokens:
            txt += f", MaxNewTokens: {self.max_new_tokens}"
        return txt


@dataclass
class TokenGeneratorPipeline(Generic[Context]):
    """Base class for LLM pipelines."""

    model: max.pipelines.interfaces.TokenGenerator[Context]
    tokenizer: Optional[AutoTokenizer] = None

    tokens_queue: BatchMultiplexQueue = field(
        default_factory=BatchMultiplexQueue
    )
    context_queue: BatchMultiplexQueue = field(
        default_factory=BatchMultiplexQueue
    )
    _background_tasks: set = field(default_factory=set)

    # TODO(SERV-180) - Temporarily setting batch size to 1 to unblock benchmarking
    # as the underlying llama3 pipeline currently only support BS=1.
    max_batch_size: int = 1

    async def next_token(
        self,
        request: TokenGeneratorRequest,
    ) -> AsyncGenerator[str, None]:
        """Generates tokens for each provided request ending with sentinel `None` tokens.
        """
        # The first token is part of a context-encoding batch.
        # This goes away once we support ragged tensors.
        context = await self.model.new_context(
            request.prompt, max_new_tokens=request.max_new_tokens
        )
        async for token in self.tokens_queue.stream(request.id, context):
            yield token

    async def all_tokens(self, request: TokenGeneratorRequest) -> list[str]:
        return [token async for token in self.next_token(request)]

    async def __aenter__(self):
        # This can go away once we have ragged tensors
        loop = asyncio.get_running_loop()
        # This worker only does context encoding. Turned off temporarily.
        context_encoder = loop.create_task(
            self.context_queue.dynamic_batching_worker(
                self.next_token,
                self.max_batch_size,
            )
        )
        token_generator = loop.create_task(
            self.tokens_queue.continuous_batching_worker(
                self.model.next_token,
                max_batch_size=self.max_batch_size,
            )
        )
        self._background_tasks |= {context_encoder, token_generator}
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        for task in self._background_tasks:
            task.cancel()
        # TODO: also cancel any `queue.get()` tasks
