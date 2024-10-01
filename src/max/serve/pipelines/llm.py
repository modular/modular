# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import asyncio
import logging
import os
import signal
from dataclasses import dataclass, field
from typing import AsyncGenerator, Generic, Optional, TypeVar

from transformers import AutoTokenizer

from max.pipelines import TokenGenerator

from max.serve.scheduler.queues import BatchMultiplexQueue

# TODO (SI-582) unify logging infra
logging.basicConfig(
    level=logging.INFO,
    encoding="utf-8",
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
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

    model: TokenGenerator[Context]
    tokenizer: Optional[AutoTokenizer] = None
    max_batch_size: int = 1

    max_queue_wait_s: float = 0.001

    # non-configurable parameters

    tokens_queue: BatchMultiplexQueue = field(
        default_factory=BatchMultiplexQueue
    )
    context_queue: BatchMultiplexQueue = field(
        default_factory=BatchMultiplexQueue
    )
    _background_tasks: set = field(default_factory=set)

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
        max_queue_wait_s = (
            0 if self.max_batch_size == 1 else self.max_queue_wait_s
        )
        # This worker only does context encoding. Turned off temporarily.
        context_encoder = loop.create_task(
            self.context_queue.dynamic_batching_worker(
                self.next_token,
                self.max_batch_size,
                max_queue_wait_s=max_queue_wait_s,
            ),
            name="dynamic_batching_worker",
        )
        token_generator = loop.create_task(
            self.tokens_queue.continuous_batching_worker(
                self.model.next_token,
                max_batch_size=self.max_batch_size,
                max_queue_wait_s=max_queue_wait_s,
            ),
            name="continuous_batching_worker",
        )
        logger.info("Created workers")

        def log_task_done(task: asyncio.Task):
            # TODO - pipe in a logger to TokenGeneratorPipeline and log here
            # TODO - should gracefully shut down here.
            logger.info("task completed: %s", task.get_name())
            for t in self._background_tasks:
                if not t.done():
                    t.cancel("terminating task")
            e = task.exception()
            if e:
                logger.error("Task completed with error. Stopping", exc_info=e)
                # Shut server down. Sending SIGTERM is ugly, but simplifies the intenral plumbing.
                os.kill(os.getpid(), signal.SIGTERM)

        context_encoder.add_done_callback(log_task_done)
        token_generator.add_done_callback(log_task_done)

        self._background_tasks |= {context_encoder, token_generator}
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        for task in self._background_tasks:
            task.cancel()
        logger.info("Exiting serving pipeline context")
        # TODO: also cancel any `queue.get()` tasks
