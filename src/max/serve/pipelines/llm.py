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
from typing import (
    Any,
    AsyncGenerator,
    Generic,
    Mapping,
    Optional,
    TypeVar,
    Union,
)

import numpy as np
from max.pipelines.interfaces import (
    TokenGeneratorRequest,
    TokenGeneratorTokenizer,
)
from max.serve.scheduler.queues import (
    BatchingStrategy,
    BatchMultiplexQueue,
    BatchQueueConfig,
)
from max.serve.telemetry.stopwatch import StopWatch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


@dataclass(frozen=True)
class TokenGeneratorTimers:
    context_creation: StopWatch = field(default_factory=StopWatch)
    context_encoding: StopWatch = field(default_factory=StopWatch)
    token_generation: StopWatch = field(default_factory=StopWatch)
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


@dataclass
class TokenGeneratorStats:
    token_gen_batch_size: int = 0
    token_gen_batch_calls: int = 0


TokenGeneratorContext = TypeVar("TokenGeneratorContext")


class TokenGeneratorPipeline(Generic[TokenGeneratorContext]):  # type: ignore
    """Base class for LLM pipelines."""

    def call_wrapper(self, batch):
        self.stats.token_gen_batch_size = self.stats.token_gen_batch_size + len(
            batch
        )
        self.stats.token_gen_batch_calls = self.stats.token_gen_batch_calls + 1
        self.logger.info(
            (
                "Executing token-gen with, %s, step, %s, average, %s,"
                " context-encoding-inq, %s, context-enc-outq, %s,"
                " token-gen-inq, %s, token-gen-outq, %s"
            ),
            len(batch),
            self.stats.token_gen_batch_calls,
            self.stats.token_gen_batch_size / self.stats.token_gen_batch_calls,
            self.context_enc_queue.in_queue.qsize() if self.context_enc_queue else -1,
            len(
                self.context_enc_queue.out_queues
            ) if self.context_enc_queue else -1,
            self.token_gen_queue.in_queue.qsize(),
            len(self.token_gen_queue.out_queues),
        )
        return batch

    def context_enc_non_empty(self) -> bool:
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

    def __init__(
        self,
        config: TokenGeneratorPipelineConfig,
        model_name: str,
        tokenizer: TokenGeneratorTokenizer,
        tg_yield_to_ce: bool = True,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.config = config
        self.stats = TokenGeneratorStats()

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
        self.request_indices = set(range(self.max_queue_size))

        self._timers: dict[str, TokenGeneratorTimers] = {}
        self._background_tasks: set[asyncio.Task] = set()

    async def create_request(self, id: str, **kwargs) -> TokenGeneratorRequest:
        await self.request_semaphore.acquire()
        index = self.request_indices.pop()
        request = TokenGeneratorRequest(id=id, index=index, **kwargs)
        self._timers[id] = TokenGeneratorTimers()
        return request

    def _complete_request(self, request: TokenGeneratorRequest):
        del self._timers[request.id]
        self.request_indices.add(request.index)
        self.request_semaphore.release()

    async def next_token(
        self,
        request: TokenGeneratorRequest,
    ) -> AsyncGenerator[str, None]:
        """Generates and streams tokens for the provided request."""
        timers = self._timers[request.id]
        self.logger.debug(
            "%s [%d]: Started: Elapsed: %0.2f ms",
            request.id,
            request.index,
            timers.total.elapsed_ms,
        )
        with timers.context_creation:
            context = await self.tokenizer.new_context(request)
        self.logger.debug(
            "%s [%d]: Context-Creation: %0.2f ms, Elapsed: %0.2f ms",
            request.id,
            request.index,
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
                    encoded_token, valid = await self.context_enc_queue.submit(
                        request.id, context
                    )
                    if valid:
                        # TODO: Put this off the main thread.
                        yield await self.tokenizer.decode(
                            context, encoded_token
                        )
                    else:
                        return
                self.logger.debug(
                    "%s [%d]: Context-Encoding: %0.2f ms, Elapsed: %0.2f ms",
                    request.id,
                    request.index,
                    timers.context_encoding.elapsed_ms,
                    timers.total.elapsed_ms,
                )
            with timers.token_generation:
                async for encoded_token in self.token_gen_queue.stream(
                    request.id, context
                ):
                    # TODO: Put this off the main thread.
                    yield await self.tokenizer.decode(context, encoded_token)
            self.logger.debug(
                "%s [%d]: Token-Generation: %0.2f ms, Elapsed: %0.2f ms",
                request.id,
                request.index,
                timers.token_generation.elapsed_ms,
                timers.total.elapsed_ms,
            )
        finally:
            self._complete_request(request)
            self.logger.debug(
                "%s [%d]: Completed: Elapsed: %0.2f ms",
                request.id,
                request.index,
                timers.total.elapsed_ms,
            )

    async def all_tokens(self, request: TokenGeneratorRequest) -> list[str]:
        """Generates all tokens for the provided request."""
        return [token async for token in self.next_token(request)]

    async def __aenter__(self):
        self.logger.info("Starting workers")
        assert not self._background_tasks

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

            self.add_background_task(queue_task)

        # Add global fanout worker.
        self.add_background_task(
            asyncio.create_task(queue.response_fanout_worker())
        )

        self.logger.info("Started workers")
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        self.logger.info("Stopping workers")
        for task in self._background_tasks:
            task.cancel()
        # TODO: also cancel any `queue.get()` tasks

    def add_background_task(self, task: asyncio.Task):
        task.add_done_callback(self.log_task_done)
        self._background_tasks.add(task)

    def log_task_done(self, task: asyncio.Task):
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


class IdentityTokenGeneratorTokenizer(
    Generic[TokenGeneratorContext],
    TokenGeneratorTokenizer[TokenGeneratorContext, str],
):
    async def encode(self, prompt: str) -> str:
        return prompt

    async def decode(
        self,
        context: TokenGeneratorContext,
        encoded: Any,
    ) -> str:
        if isinstance(encoded, str):
            return encoded
        return ""


class PreTrainedTokenGeneratorTokenizer(
    Generic[TokenGeneratorContext],
    TokenGeneratorTokenizer[TokenGeneratorContext, np.ndarray],
):
    def __init__(
        self,
        delegate: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, None],
    ) -> None:
        self.delegate = delegate

    async def encode(self, prompt: str) -> np.ndarray:
        if self.delegate:
            return np.array(self.delegate.encode(prompt))
        return np.ones([0])

    async def decode(
        self,
        context: TokenGeneratorContext,
        encoded: np.ndarray,
    ) -> str:
        if self.delegate:
            return self.delegate.decode(encoded)
        return ""
