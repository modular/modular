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
from functools import partial
from typing import AsyncGenerator, Callable, Generic, Mapping, Optional, TypeVar

from max.pipelines.interfaces import (
    TokenGeneratorRequest,
    TokenGeneratorTokenizer,
)
from max.serve.scheduler.queues import (
    BatchingStrategy,
    BatchMultiplexQueue,
    BatchQueueConfig,
)
from max.serve.telemetry.metrics import METRICS
from max.serve.telemetry.stopwatch import StopWatch


@dataclass(frozen=True)
class TokenGeneratorTimers:
    context_creation: StopWatch = field(default_factory=StopWatch)
    context_encoding: StopWatch = field(default_factory=StopWatch)
    token_generation: StopWatch = field(default_factory=StopWatch)
    ttft: StopWatch = field(default_factory=StopWatch)
    # Started on creation
    total: StopWatch = field(default_factory=StopWatch.start)


@dataclass(frozen=True)
class TokenGeneratorPipelineConfig:
    """
    Example config

    .. code-block:: json

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
    """

    token_generation: BatchQueueConfig
    context_encoding: Optional[BatchQueueConfig] = None

    @classmethod
    def dynamic_homogenous(
        cls, batch_size: int, batch_timeout=0.1, max_forward_steps=1
    ) -> TokenGeneratorPipelineConfig:
        """The dynamic-homogenous config uses a single queue.
        Requests are dequeued into a batch and the entire batch is
        executed until all requests are completed.
        """
        token_generation_config = BatchQueueConfig(
            strategy=BatchingStrategy.DYNAMIC_IMMUTABLE,
            size=batch_size,
            timeout=batch_timeout,
            max_forward_steps=max_forward_steps,
        )
        config = cls(token_generation=token_generation_config)
        return config

    @classmethod
    def continuous_heterogenous(
        cls,
        tg_batch_size: int,
        ce_batch_size: int,
        ce_batch_timeout=0.1,
        max_forward_steps=1,
        target_ce_batch_tokens=4096,
    ) -> TokenGeneratorPipelineConfig:
        """The continuous-hetrogenous config creates 2 queues.
        Context-encoding is done via dynamic batching.
        Token-generation is done via continuous batching.
        """
        token_generation_config = BatchQueueConfig(
            strategy=BatchingStrategy.CONTINUOUS,
            size=tg_batch_size,
            timeout=0.0,
            max_forward_steps=max_forward_steps,
        )
        context_encoding_config = BatchQueueConfig(
            strategy=BatchingStrategy.DYNAMIC,
            size=ce_batch_size,
            timeout=ce_batch_timeout,
            target_sum_seq_len=target_ce_batch_tokens,
        )
        config = cls(
            context_encoding=context_encoding_config,
            token_generation=token_generation_config,
        )
        return config


@dataclass
class TokenGeneratorStats:
    token_gen_batch_size: int = 0
    token_gen_batch_calls: int = 0


TokenGeneratorContext = TypeVar("TokenGeneratorContext")


class TokenGeneratorPipeline(Generic[TokenGeneratorContext]):  # type: ignore
    """Base class for LLM pipelines."""

    def context_enc_non_empty(self) -> bool:
        if self.debug_logging:
            self.logger.debug(
                "Called context_enc_non_empty with %s",
                self.context_enc_queue.in_queue.qsize() if self.context_enc_queue else -1,  # type: ignore
            )
        if self.context_enc_queue:
            return (
                self.context_enc_queue.in_queue.qsize()
                > 0
                # and self.token_gen.in_queue.qsize() < 32
            )
        return False

    def context_enc_full(self) -> bool:
        """Check if the context queue has a full batch in it."""
        if self.debug_logging:
            self.logger.debug(
                "Called context_enc_full with %s",
                self.context_enc_queue.in_queue.qsize() if self.context_enc_queue else -1,  # type: ignore
            )
        if self.context_enc_queue:
            return self.context_enc_queue.in_queue.qsize() >= 32
        return False

    def __init__(
        self,
        config: TokenGeneratorPipelineConfig,
        model_name: str,
        tokenizer: TokenGeneratorTokenizer,
        tg_yield_to_ce: bool = True,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("%s: Constructed", model_name)

        self.model_name = model_name
        self.tokenizer = tokenizer
        self.config = config
        self.stats = TokenGeneratorStats()
        self.debug_logging = self.logger.isEnabledFor(logging.DEBUG)

        self.queues: list[BatchMultiplexQueue] = []
        self.token_gen_queue = BatchMultiplexQueue(
            "token_generation",
            self.model_name,
            config.token_generation,
            completed_fn=self.completed_token_generation_requests,
            should_yield=self.context_enc_non_empty if tg_yield_to_ce else None,
        )
        self.queues.append(self.token_gen_queue)
        # Create a context-encoding queue if specified.
        self.context_enc_queue: Optional[BatchMultiplexQueue] = None
        if config.context_encoding:
            self.context_enc_queue = BatchMultiplexQueue(
                "context_encoding",
                self.model_name,
                config.context_encoding,
                completed_fn=self.completed_context_encoding_requests,
            )
            self.queues.append(self.context_enc_queue)
        self.max_queue_size = max(q.config.size for q in self.queues)

        self.request_semaphore = asyncio.BoundedSemaphore(self.max_queue_size)
        self.tokenizer_semaphore = asyncio.BoundedSemaphore(2)
        self.request_indices = set(range(self.max_queue_size))

        self._timers: dict[str, TokenGeneratorTimers] = {}
        self._background_tasks: set[asyncio.Task] = set()

    async def create_request(self, id: str, **kwargs) -> TokenGeneratorRequest:
        METRICS.reqsQueued(1)
        await self.request_semaphore.acquire()
        METRICS.reqsQueued(-1)
        METRICS.reqsRunning(1)
        index = self.request_indices.pop()
        request = TokenGeneratorRequest(id=id, index=index, **kwargs)
        self._timers[id] = TokenGeneratorTimers()
        return request

    def _complete_request(self, request: TokenGeneratorRequest):
        del self._timers[request.id]
        self.request_indices.add(request.index)
        METRICS.reqsRunning(-1)
        self.request_semaphore.release()

    async def next_token(
        self,
        request: TokenGeneratorRequest,
    ) -> AsyncGenerator[str, None]:
        """Generates and streams tokens for the provided request."""
        timers = self._timers[request.id]
        timers.ttft.start_ns = request.req_recv_time_ns
        self.logger.debug(
            "%s [%d]: Started: Elapsed: %0.2f ms",
            request.id,
            request.index,
            timers.total.elapsed_ms,
        )

        # try to defer to shipping a context encoding batch
        while self.context_enc_full():
            await asyncio.sleep(0.0)

        # the semaphore acts like a dumb queue to prevent all of the coros
        # from getting queued in the work loop all at once
        # todo: replace with a proper work queue for tokenization
        async with self.tokenizer_semaphore:
            with timers.context_creation:
                context = await self.tokenizer.new_context(request)
        if self.debug_logging:
            self.logger.debug(
                "%s [%d]: Context-Creation: %0.2f ms, Elapsed: %0.2f ms",
                request.id,
                request.index,
                timers.context_creation.elapsed_ms,
                timers.total.elapsed_ms,
            )
        # TODO(MAXCORE-137): TokenGeneratorContext currently does not enforce
        # a seq_len property.
        if hasattr(context, "seq_len"):
            METRICS.inputTokens(context.seq_len)

        tokenIdx = 0
        try:
            # Use a different queue for context encoding if specified.
            # Otherwise, the same queue is used. And in case of dynamic batching,
            # any new requests which require CE will be blocked until any active
            # CE or TG requests being serviced by the dynamic-batching-queue are
            # fully processed.
            if self.context_enc_queue:
                with timers.context_encoding:
                    encoded_token, valid = await self.context_enc_queue.submit(
                        request.id, context
                    )
                    if valid:
                        # TODO: Put this off the main thread.
                        METRICS.ttft(timers.ttft.elapsed_ms)
                        tokenIdx += 1
                        yield await self.tokenizer.decode(
                            context, encoded_token
                        )
                    else:
                        return
                if self.debug_logging:
                    self.logger.debug(
                        (
                            "%s [%d]: Context-Encoding: %0.2f ms, Elapsed:"
                            " %0.2f ms"
                        ),
                        request.id,
                        request.index,
                        timers.context_encoding.elapsed_ms,
                        timers.total.elapsed_ms,
                    )
            METRICS.inputTime(timers.total.elapsed_s)
            with timers.token_generation:
                async for encoded_token in self.token_gen_queue.stream(
                    request.id, context
                ):
                    # TODO: Put this off the main thread.
                    if tokenIdx == 0:
                        METRICS.ttft(timers.ttft.elapsed_ms)
                    tokenIdx += 1
                    yield await self.tokenizer.decode(context, encoded_token)
            if self.debug_logging:
                self.logger.debug(
                    "%s [%d]: Token-Generation: %0.2f ms, Elapsed: %0.2f ms",
                    request.id,
                    request.index,
                    timers.token_generation.elapsed_ms,
                    timers.total.elapsed_ms,
                )
            METRICS.outputTime(timers.token_generation.elapsed_ms)
        finally:
            self._complete_request(request)
            if self.debug_logging:
                self.logger.debug(
                    "%s [%d]: Completed: Elapsed: %0.2f ms",
                    request.id,
                    request.index,
                    timers.total.elapsed_ms,
                )

            METRICS.outputTokens(tokenIdx)

    async def all_tokens(self, request: TokenGeneratorRequest) -> list[str]:
        """Generates all tokens for the provided request."""
        return [token async for token in self.next_token(request)]

    async def __aenter__(self):
        self.logger.info("%s: Starting workers:", self.model_name)
        assert not self._background_tasks

        for queue in self.queues:
            # TODO@gaz: Move to queue constructor once the queue has constructor.
            if (
                queue.config.strategy == BatchingStrategy.DYNAMIC
                or queue.config.strategy == BatchingStrategy.DYNAMIC_IMMUTABLE
            ):
                self.create_background_task(queue.dynamic_batching_worker)
            elif queue.config.strategy == BatchingStrategy.CONTINUOUS:
                self.create_background_task(queue.continuous_batching_worker)

        # Add global fanout worker.
        self.create_background_task(queue.response_fanout_worker)

        self.logger.info(
            "%s: Started workers: %d tasks",
            self.model_name,
            len(self._background_tasks),
        )
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        self.logger.info("%s: Stopping workers", self.model_name)
        for task in self._background_tasks:
            task.cancel()
        # await asyncio.sleep(0.1)
        # TODO: also cancel any `queue.get()` tasks

    def create_background_task(self, fn: Callable):
        task_name = fn.__name__
        task = asyncio.create_task(fn())
        task.add_done_callback(partial(self.log_task_done, task_name=task_name))
        self._background_tasks.add(task)
        self.logger.info(
            "%s: Task Added: %s, %s, %d total",
            self.model_name,
            task_name,
            type(fn),
            len(self._background_tasks),
        )

    def log_task_done(self, task: asyncio.Task, task_name: str):
        # TODO - should gracefully shut down here.
        self._background_tasks.remove(task)
        self.logger.info(
            "%s: Task completed: %s, %d remaining",
            self.model_name,
            task_name,
            len(self._background_tasks),
        )
        # Cancel remaining tasks.
        for t in self._background_tasks:
            if not t.done():
                t.cancel("Terminating task")
        if task.cancelled():
            return
        e = task.exception()
        if e:
            self.logger.error("Task completed with error. Stopping", exc_info=e)
            # Shut server down.
            # Sending SIGTERM is ugly, but simplifies the internal plumbing.
            os.kill(os.getpid(), signal.SIGTERM)

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
