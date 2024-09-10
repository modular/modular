# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import AsyncGenerator, Generic, Optional, TypeVar

from max.serve.scheduler.queues import BatchMultiplexQueue
from transformers import AutoTokenizer

Context = TypeVar("Context")


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
        self, request_id: str, prompt: str
    ) -> AsyncGenerator[str, None]:
        """Generates tokens for each provided request ending with sentinel `None` tokens.
        """
        # The first token is part of a context-encoding batch.
        # This goes away once we support ragged tensors.
        context = await self.context_queue.submit(request_id, prompt)
        async for token in self.tokens_queue.stream(request_id, context):
            yield token
            if token is None:
                break

    async def all_tokens(self, request_id: str, prompt: str) -> list[str]:
        return [
            token
            async for token in self.next_token(request_id, prompt)
            if token is not None
        ]

    async def create_context(self, request: dict[str, str]):
        assert len(request) == 1
        request_id, request_prompt = next(iter(request.items()))
        context = await self.model.new_context(request_prompt)
        assert request_id == request_id
        return {request_id: context}

    async def __aenter__(self):
        # This can go away once we have ragged tensors
        loop = asyncio.get_running_loop()
        context_encoder = loop.create_task(
            self.context_queue.dynamic_batching_worker(
                self.create_context,
                self.max_batch_size,
            )
        )
        token_generator = loop.create_task(
            self.tokens_queue.continuous_batching_worker(
                self.model.next_token,
                complete=(lambda token: token is None),
                max_batch_size=self.max_batch_size,
            )
        )
        self._background_tasks |= {context_encoder, token_generator}

    async def __aexit__(self, exc_type, exc_value, traceback):
        for task in self._background_tasks:
            task.cancel()
        # TODO: also cancel any `queue.get()` tasks
