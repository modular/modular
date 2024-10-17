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
from typing import AsyncGenerator, Generic, Mapping, Optional
from transformers import PreTrainedTokenizerBase
from max.pipelines import TokenGenerator, TokenGeneratorContext
from max.serve.telemetry.stopwatch import StopWatch

from max.serve.scheduler.queues import (
    BatchMultiplexQueue,
    BatchQueueConfig,
    BatchingStrategy,
)


@dataclass(frozen=True)
class TokenGeneratorTimers:
    context_creation: StopWatch = field(default_factory=StopWatch)
    context_encoding: StopWatch = field(default_factory=StopWatch)
    token_generation: StopWatch = field(default_factory=StopWatch)
    # Started on creation
    total: StopWatch = field(default_factory=StopWatch.start)


@dataclass(frozen=True)
class TokenGeneratorRequest:
    id: str
    prompt: str
    max_new_tokens: Optional[int] = None
    timers: TokenGeneratorTimers = field(default_factory=TokenGeneratorTimers)

    def __str__(self):
        txt = f"Id: {self.id}, Prompt: [{self.prompt[:40]}]"
        if self.max_new_tokens:
            txt += f", MaxNewTokens: {self.max_new_tokens}"
        return txt


@dataclass(frozen=True)
class TokenGeneratorPipelineConfig:
    """Example config
    {
        "context_encoding": {
            "strategy": "dynamic",
            "size": 1,
            "timeout": 0.1
        },
        "token_generation": {
            "strategy": "continuous",
            "size": 64,
            "timeout": 0.0
            }
        }
    }
    """

    token_generation: BatchQueueConfig
    context_encoding: Optional[BatchQueueConfig] = None

    @classmethod
    def dynamic_homogenous(
        cls, batch_size: int, batch_timeout=0.1
    ) -> TokenGeneratorPipelineConfig:
        """The dynamic-homogenous config uses a single queue.
        Requests are dequed into a batch and the entire batch is
        executed until all requests are completed.
        """
        token_generation_config = BatchQueueConfig(
            strategy=BatchingStrategy.DYNAMIC_IMMUTABLE,
            size=batch_size,
            timeout=batch_timeout,
        )
        config = cls(token_generation=token_generation_config)
        return config

    @classmethod
    def continuous_heterogenous(
        cls, tg_batch_size: int, ce_batch_size: int, ce_batch_timeout=0.1
    ) -> TokenGeneratorPipelineConfig:
        """The continuous-hetrogenous config creates 2 queues.
        Context-encoding is done via dynamic batching.
        Token-generation is done via continuous batching.
        """
        token_generation_config = BatchQueueConfig(
            strategy=BatchingStrategy.CONTINUOUS,
            size=tg_batch_size,
            timeout=0.0,
        )
        context_encoding_config = BatchQueueConfig(
            strategy=BatchingStrategy.DYNAMIC,
            size=ce_batch_size,
            timeout=ce_batch_timeout,
        )
        config = cls(
            context_encoding=context_encoding_config,
            token_generation=token_generation_config,
        )
        return config


class TokenGeneratorPipeline(Generic[TokenGeneratorContext]):  # type: ignore
    """Base class for LLM pipelines."""

    def __init__(
        self,
        config: TokenGeneratorPipelineConfig,
        model: TokenGenerator[TokenGeneratorContext],
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.queues: list[BatchMultiplexQueue] = []
        self._background_tasks: set[asyncio.Task] = set()
        self.token_gen_queue = BatchMultiplexQueue(
            "token_generation",
            config.token_generation,
            self.model.next_token,
            completed_fn=TokenGeneratorPipeline.completed_token_generation_requests,
        )
        self.queues.append(self.token_gen_queue)
        # Create a context-encoding queue if specified.
        self.context_enc_queue: Optional[BatchMultiplexQueue] = None
        if config.context_encoding:
            self.context_enc_queue = BatchMultiplexQueue(
                name="context_encoding",
                config=config.context_encoding,
                executor_fn=self.model.next_token,
                completed_fn=TokenGeneratorPipeline.completed_context_encoding_requests,
            )
            self.queues.append(self.context_enc_queue)

    async def next_token(
        self,
        request: TokenGeneratorRequest,
    ) -> AsyncGenerator[str, None]:
        """Generates and streams tokens for the provided request."""
        timers = request.timers
        self.logger.debug(
            "%s: Started: Elapsed: %0.2f ms",
            request.id,
            timers.total.elapsed_ms,
        )
        with timers.context_creation:
            context = await self.model.new_context(
                request.prompt, max_new_tokens=request.max_new_tokens
            )
        self.logger.debug(
            "%s: Context-Creation: %0.2f ms, Elapsed: %0.2f ms",
            request.id,
            timers.context_creation.elapsed_ms,
            timers.total.elapsed_ms,
        )
        try:
            # Use a different queue for context encoding if specified.
            # Otherwise, the same queue is used. And in case of dynamic batching,
            # any new requests which require CE will be blocked until any active
            # CE or TG requests being serviced by the dynamic-batching-queue are
            # fully processed.
            if self.context_enc_queue:
                with timers.context_encoding:
                    (token, valid) = await self.context_enc_queue.submit(
                        request.id, context
                    )
                    if valid:
                        yield token
                self.logger.debug(
                    "%s: Context-Encoding: %0.2f ms, Elapsed: %0.2f ms",
                    request.id,
                    timers.context_encoding.elapsed_ms,
                    timers.total.elapsed_ms,
                )
            with timers.token_generation:
                async for token in self.token_gen_queue.stream(
                    request.id, context
                ):
                    yield token
            self.logger.debug(
                "%s: Token-Generation: %0.2f ms, Elapsed: %0.2f ms",
                request.id,
                timers.token_generation.elapsed_ms,
                timers.total.elapsed_ms,
            )
        finally:
            await self.model.release(context)
            self.logger.debug(
                "%s: Completed: Elapsed: %0.2f ms",
                request.id,
                timers.total.elapsed_ms,
            )

    async def all_tokens(self, request: TokenGeneratorRequest) -> list[str]:
        """Generates all tokens for the provided request."""
        return [token async for token in self.next_token(request)]

    async def __aenter__(self):
        self.logger.info("Starting workers")
        assert not self._background_tasks

        def log_task_done(task: asyncio.Task):
            # TODO - should gracefully shut down here.
            self.logger.info("Task completed: %s", task.get_name())
            self._background_tasks.remove(task)
            # Cancel remaining tasks.
            for t in self._background_tasks:
                if not t.done():
                    t.cancel("Terminating task")
            if task.cancelled():
                return
            e = task.exception()
            if e:
                self.logger.error(
                    "Task completed with error. Stopping", exc_info=e
                )
                # Shut server down.
                # Sending SIGTERM is ugly, but simplifies the internal plumbing.
                os.kill(os.getpid(), signal.SIGTERM)

        for queue in self.queues:
            # TODO@gaz: Move to queue constructor once the queue has constructor.
            if (
                queue.config.strategy == BatchingStrategy.DYNAMIC
                or queue.config.strategy == BatchingStrategy.DYNAMIC_IMMUTABLE
            ):
                queue_task = asyncio.create_task(
                    queue.dynamic_batching_worker()
                )
            elif queue.config.strategy == BatchingStrategy.CONTINUOUS:
                queue_task = asyncio.create_task(
                    queue.continuous_batching_worker()
                )
            queue_task.add_done_callback(log_task_done)
            self._background_tasks.add(queue_task)

        self.logger.info("Started workers")
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        self.logger.info("Stopping workers")
        for task in self._background_tasks:
            task.cancel()
        # TODO: also cancel any `queue.get()` tasks

    @staticmethod
    def completed_context_encoding_requests(
        batch_input: Mapping[str, TokenGeneratorContext],
        batch_output: Mapping[str, str],
    ):
        # All request ids in the input are assumed to be
        # context-encoded after the first pass.
        return set(batch_input.keys())

    @staticmethod
    def completed_token_generation_requests(
        batch_input: Mapping[str, TokenGeneratorContext],
        batch_output: Mapping[str, str],
    ):
        # Request ids which were in the input batch but were not produced
        # in the output batch are assumed completed.
        return batch_input.keys() - batch_output.keys()
