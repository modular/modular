# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import AsyncGenerator, Generic, Tuple, TypeVar

from max.serve.scheduler.queues import BatchMultiplexQueue

Context = TypeVar("Context")


@dataclass
class TokenGeneratorPipeline(Generic[Context]):
    """Base class for LLM pipelines."""

    model: max.pipelines.interfaces.TokenGenerator[Context]
    max_batch_size: int = 32

    tokens_queue: BatchMultiplexQueue = field(
        default_factory=BatchMultiplexQueue
    )
    context_queue: BatchMultiplexQueue = field(
        default_factory=BatchMultiplexQueue
    )
    _background_tasks: set = field(default_factory=set)

    async def next_token(
        self, requests: dict[str, Context]
    ) -> AsyncGenerator[Tuple[str, str], None]:
        """Generates tokens for each provided request ending with sentinel `None` tokens.
        """
        # TODO: The selection of requests should be configurable.
        for rid, context in requests.items():
            # The first token is part of a context-encoding batch.
            # This goes away once we support ragged tensors.
            yield rid, await self.context_queue.submit(context)
            async for token in self.tokens_queue.stream(context):
                yield rid, token
                if not token:
                    break

    async def __aenter__(self):
        # This can go away once we have ragged tensors
        loop = asyncio.get_running_loop()
        context_encoder = loop.create_task(
            self.context_queue.dynamic_batching_worker(
                self.model.next_token,
                self.max_batch_size,
            )
        )
        token_generator = loop.create_task(
            self.tokens_queue.continuous_batching_worker(
                self.model.next_token,
                complete=(lambda token: not token),
                max_batch_size=self.max_batch_size,
            )
        )
        self._background_tasks |= {context_encoder, token_generator}

    async def __aexit__(self, exc_type, exc_value, traceback):
        for task in self._background_tasks:
            task.cancel()
        # TODO: also cancel any `queue.get()` tasks
